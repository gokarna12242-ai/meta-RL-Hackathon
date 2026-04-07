"""
Baseline inference script for the Data Cleaning Environment.

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks
and reports quality scores.  Falls back to a scripted heuristic policy
when no LLM API key is available so the run is always reproducible.

Environment variables (read at startup):
    API_BASE_URL  — LLM API endpoint        (default: https://api.openai.com/v1)
    MODEL_NAME    — model identifier         (default: gpt-4o-mini)
    HF_TOKEN      — Hugging Face API token   (mandatory for LLM mode)

Output format:
    [START] task=<task> env=data_clean_env model=<model>
    [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> rewards=<r1,r2,...,rn>

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback

# ---------------------------------------------------------------------------
# We run the environment *in-process* (no server needed for the baseline).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_clean_env.server.environment import (
    DataCleanEnvironment,
    TASKS,
)
from data_clean_env.models import DataCleanAction

# ---------------------------------------------------------------------------
# Config — required env vars per hackathon guidelines
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# LLM client (optional — falls back to scripted policy if unavailable)
# ---------------------------------------------------------------------------
USE_LLM = False
client = None

if HF_TOKEN:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        USE_LLM = True
    except ImportError:
        pass

if not USE_LLM:
    print("INFO: No HF_TOKEN or openai package unavailable — using scripted policy.", file=sys.stderr)

# ---------------------------------------------------------------------------
# System prompt for LLM-based agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert data cleaning agent. You interact with a Data Cleaning Environment
to fix data quality issues in tabular datasets.

You receive observations containing:
- dataset_preview: first rows of the dataset
- column_info: column types, null counts, unique counts
- issues_found: list of detected quality issues
- quality_score: current score (0.0 - 1.0)
- available_commands: list of commands you can use

You MUST respond with a single JSON object with these fields:
{
  "command": "<command_name>",
  "column": "<column_name_or_null>",
  "params": { ... }
}

Available commands:
- inspect (column=optional) — view dataset info
- fill_missing (column=required, params={strategy:"mean"|"median"|"mode"|"value", value:<v>})
- cast_type (column=required, params={dtype:"int"|"float"|"str"|"datetime"})
- remove_duplicates — remove duplicate rows
- fix_format (column=required, params={pattern:"<regex>", replacement:"<str>"})
- filter_outliers (column=required, params={method:"iqr"|"zscore", threshold:<n>})
- drop_rows (params={condition:"<pandas_query>"})
- rename_column (column=required, params={new_name:"<name>"})
- standardize (column=required, params={mapping:{"old":"new",...}})
- submit — submit for final grading

Strategy: First inspect the data, identify issues, fix them one by one, then submit.
Always respond with ONLY the JSON object, no extra text.
"""


# ===================================================================== #
#  Scripted fallback policies — one per task (deterministic, no LLM)
# ===================================================================== #

SCRIPTED_POLICIES: dict[str, list[dict]] = {
    "easy": [
        {"command": "inspect", "column": None, "params": {}},
        {"command": "fix_format", "column": "revenue",
         "params": {"pattern": r"^\$", "replacement": ""}},
        {"command": "cast_type", "column": "revenue",
         "params": {"dtype": "float"}},
        {"command": "cast_type", "column": "age",
         "params": {"dtype": "int"}},
        {"command": "fill_missing", "column": "age",
         "params": {"strategy": "median"}},
        {"command": "fill_missing", "column": "email",
         "params": {"strategy": "value", "value": "unknown@example.com"}},
        {"command": "submit", "column": None, "params": {}},
    ],
    "medium": [
        {"command": "inspect", "column": None, "params": {}},
        {"command": "remove_duplicates", "column": None, "params": {}},
        {"command": "standardize", "column": "region",
         "params": {"mapping": {
             "north": "North", "NORTH": "North", "N.": "North",
             "south": "South", "S.": "South",
             "east": "East",
             "west": "West", "W.": "West",
         }}},
        {"command": "fill_missing", "column": "unit_price",
         "params": {"strategy": "median"}},
        {"command": "filter_outliers", "column": "quantity",
         "params": {"method": "zscore", "threshold": 3.0}},
        {"command": "submit", "column": None, "params": {}},
    ],
    "hard": [
        {"command": "inspect", "column": None, "params": {}},
        {"command": "remove_duplicates", "column": None, "params": {}},
        {"command": "standardize", "column": "department",
         "params": {"mapping": {
             "engineering": "Engineering", "Eng": "Engineering",
             "ENGINEERING": "Engineering", "Engg": "Engineering",
             "marketing": "Marketing", "Mktg": "Marketing", "MARKETING": "Marketing",
             "sales": "Sales", "SALES": "Sales",
             "hr": "HR", "Human Resources": "HR", "H.R.": "HR",
             "finance": "Finance", "FIN": "Finance", "FINANCE": "Finance",
         }}},
        {"command": "filter_outliers", "column": "salary",
         "params": {"method": "iqr", "threshold": 2.0}},
        {"command": "fill_missing", "column": "salary",
         "params": {"strategy": "median"}},
        {"command": "fill_missing", "column": "performance_score",
         "params": {"strategy": "median"}},
        {"command": "fill_missing", "column": "department",
         "params": {"strategy": "mode"}},
        {"command": "fill_missing", "column": "phone",
         "params": {"strategy": "value", "value": "+1-555-0000"}},
        {"command": "submit", "column": None, "params": {}},
    ],
}


# ===================================================================== #
#  Helpers
# ===================================================================== #

def parse_agent_response(response_text: str) -> dict:
    """Extract JSON action from LLM response."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"command": "inspect", "column": None, "params": {}}


def _action_str(action: DataCleanAction) -> str:
    """Compact display string for an action."""
    s = action.command
    if action.column:
        s += f"({action.column})"
    return s


# ===================================================================== #
#  Run a single task
# ===================================================================== #

def run_task(task_id: str, seed: int = 42) -> tuple[bool, int, list[float]]:
    """Run the agent on *task_id*.

    Returns (success, steps_taken, list_of_rewards).
    """
    env = DataCleanEnvironment(task_id=task_id, seed=seed)
    obs = None
    step = 0
    rewards: list[float] = []
    last_error: str | None = None

    model_label = MODEL_NAME if USE_LLM else "scripted"
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}] if USE_LLM else []
    scripted = SCRIPTED_POLICIES.get(task_id, SCRIPTED_POLICIES["easy"])
    script_idx = 0

    try:
        obs = env.reset()

        # --- [START] line -------------------------------------------------
        print(f"[START] task={task_id} env=data_clean_env model={model_label}")

        while obs is not None and not obs.done and step < obs.max_steps:
            # ---- decide next action ----
            if USE_LLM:
                user_msg = (
                    f"Step {obs.step_count}/{obs.max_steps} | Quality: {obs.quality_score:.4f}\n\n"
                    f"Dataset preview:\n{obs.dataset_preview}\n\n"
                    f"Column info:\n{obs.column_info}\n\n"
                    f"Issues found:\n" + "\n".join(f"  - {i}" for i in obs.issues_found) + "\n\n"
                    f"Choose your next action (respond with JSON only):"
                )
                messages.append({"role": "user", "content": user_msg})

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=256,
                    )
                    agent_text = response.choices[0].message.content or ""
                except Exception as e:
                    agent_text = '{"command": "submit"}'

                messages.append({"role": "assistant", "content": agent_text})
                action_dict = parse_agent_response(agent_text)

                # sliding window to keep context manageable
                if len(messages) > 20:
                    messages = messages[:1] + messages[-14:]
            else:
                # scripted policy
                if script_idx < len(scripted):
                    action_dict = scripted[script_idx]
                    script_idx += 1
                else:
                    action_dict = {"command": "submit"}

            action = DataCleanAction(
                command=action_dict.get("command", "inspect"),
                column=action_dict.get("column"),
                params=action_dict.get("params", {}),
            )

            # ---- step env ----
            obs = env.step(action)
            step += 1
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)

            # Check for error in message
            error_str = "null"
            if obs.message and ("error" in obs.message.lower() or "not found" in obs.message.lower()
                                or "unknown" in obs.message.lower() or "forbidden" in obs.message.lower()):
                error_str = obs.message.replace("\n", " ").strip()
                last_error = error_str

            # --- [STEP] line ---
            print(
                f"[STEP] step={step} "
                f"action={_action_str(action)} "
                f"reward={reward:.2f} "
                f"done={'true' if obs.done else 'false'} "
                f"error={error_str}"
            )

            if obs.done:
                break

    except Exception:
        # Make sure we always emit [END]
        traceback.print_exc(file=sys.stderr)

    success = obs.quality_score >= 0.5 if obs is not None else False

    # --- [END] line ---
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} "
        f"rewards={rewards_str}"
    )

    return success, step, rewards


# ===================================================================== #
#  Main
# ===================================================================== #

def main():
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        success, steps, rewards = run_task(task_id)
        results[task_id] = {
            "success": success,
            "steps": steps,
            "total_reward": sum(rewards),
        }

    return results


if __name__ == "__main__":
    main()
