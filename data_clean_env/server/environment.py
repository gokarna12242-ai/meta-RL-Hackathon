"""
Data Cleaning Environment — core logic.

Simulates real-world tabular data cleaning.  The agent receives a messy
dataset and must issue cleaning commands (inspect, fill_missing, cast_type,
remove_duplicates, fix_format, filter_outliers, drop_rows, rename_column,
submit) to reach a target quality score.

Three built-in tasks (easy / medium / hard) provide a difficulty ramp.
Graders score 0.0–1.0 based on the quality of the cleaned dataset.
"""

from __future__ import annotations

import copy
import io
import math
import random
import re
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import DataCleanAction, DataCleanObservation, DataCleanState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS_EASY = 15
MAX_STEPS_MEDIUM = 25
MAX_STEPS_HARD = 40

AVAILABLE_COMMANDS = """\
Available commands:
  inspect                           — Show dataset info, column types, null counts.
  fill_missing  column=<col>  params={"strategy":"mean"|"median"|"mode"|"value","value":<v>}
                                    — Fill missing values in <col>.
  cast_type     column=<col>  params={"dtype":"int"|"float"|"str"|"datetime"}
                                    — Cast column to a new type.
  remove_duplicates                 — Remove duplicate rows.
  fix_format    column=<col>  params={"pattern":"<regex>","replacement":"<str>"}
                                    — Regex-based string cleaning on <col>.
  filter_outliers column=<col> params={"method":"iqr"|"zscore","threshold":<n>}
                                    — Remove outlier rows in <col>.
  drop_rows     params={"condition":"<pandas-query>"}
                                    — Drop rows matching a condition.
  rename_column column=<col>  params={"new_name":"<name>"}
                                    — Rename a column.
  standardize   column=<col>  params={"mapping":{"old1":"new1",...}}
                                    — Map inconsistent values to standard ones.
  submit                            — Submit cleaned dataset for grading.
"""


# ====================================================================== #
#  Dataset generators — one per task
# ====================================================================== #

def _generate_easy_dataset(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Easy task: small customer dataset with missing values & wrong types."""
    rng = random.Random(seed)

    n = 20
    names = [f"Customer_{i}" for i in range(1, n + 1)]
    ages = [rng.randint(18, 80) for _ in range(n)]
    emails = [f"user{i}@example.com" for i in range(1, n + 1)]
    revenues = [round(rng.uniform(100, 5000), 2) for _ in range(n)]
    signup_dates = [f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}" for _ in range(n)]

    df = pd.DataFrame({
        "name": names,
        "age": ages,
        "email": emails,
        "revenue": revenues,
        "signup_date": signup_dates,
    })

    # Inject issues
    dirty = df.copy()

    # 1. Missing values in age (3 rows)
    for idx in rng.sample(range(n), 3):
        dirty.loc[idx, "age"] = None

    # 2. Age stored as string
    dirty["age"] = dirty["age"].astype(object)
    for idx in range(n):
        val = dirty.loc[idx, "age"]
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            dirty.loc[idx, "age"] = str(int(val))

    # 3. Missing email (2 rows)
    for idx in rng.sample(range(n), 2):
        dirty.loc[idx, "email"] = None

    # 4. Revenue as string with $ prefix
    dirty["revenue"] = dirty["revenue"].apply(lambda x: f"${x}")

    target = df.copy()
    target["age"] = target["age"].astype(int)
    target["revenue"] = target["revenue"].astype(float)

    issues = {
        "missing_age": 3,
        "age_wrong_type": True,
        "missing_email": 2,
        "revenue_wrong_format": True,
    }
    return dirty, target, issues


def _generate_medium_dataset(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Medium task: sales dataset with duplicates, outliers, mixed formats."""
    rng = random.Random(seed)

    n = 50
    products = ["Widget", "Gadget", "Doohickey", "Thingamajig", "Whatchamacallit"]
    regions = ["North", "South", "East", "West"]

    rows = []
    for i in range(n):
        rows.append({
            "order_id": 1000 + i,
            "product": rng.choice(products),
            "quantity": rng.randint(1, 100),
            "unit_price": round(rng.uniform(5.0, 500.0), 2),
            "region": rng.choice(regions),
            "order_date": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "customer_id": rng.randint(100, 200),
        })

    df = pd.DataFrame(rows)
    dirty = df.copy()

    # 1. Duplicate rows (5)
    dup_indices = rng.sample(range(n), 5)
    dups = dirty.iloc[dup_indices].copy()
    dirty = pd.concat([dirty, dups], ignore_index=True)

    # 2. Inconsistent region names
    inconsistent_map = {"North": ["north", "NORTH", "N."], "South": ["south", "S."], "East": ["east"], "West": ["west", "W."]}
    for idx in range(len(dirty)):
        region = dirty.loc[idx, "region"]
        if region in inconsistent_map and rng.random() < 0.3:
            dirty.loc[idx, "region"] = rng.choice(inconsistent_map[region])

    # 3. Missing values in unit_price (4 rows)
    for idx in rng.sample(range(len(dirty)), 4):
        dirty.loc[idx, "unit_price"] = None

    # 4. Outliers in quantity (3 extreme values)
    outlier_indices = rng.sample(range(len(dirty)), 3)
    for idx in outlier_indices:
        dirty.loc[idx, "quantity"] = rng.randint(9000, 99999)

    # 5. Date format inconsistency
    for idx in range(len(dirty)):
        if rng.random() < 0.2:
            date_str = dirty.loc[idx, "order_date"]
            parts = date_str.split("-")
            dirty.loc[idx, "order_date"] = f"{parts[1]}/{parts[2]}/{parts[0]}"

    target = df.copy()
    issues = {
        "duplicates": 5,
        "inconsistent_regions": True,
        "missing_unit_price": 4,
        "outlier_quantity": 3,
        "inconsistent_dates": True,
    }
    return dirty, target, issues


def _generate_hard_dataset(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Hard task: HR dataset with complex issues — referential integrity,
    business rule violations, encoding issues, multi-column dependencies."""
    rng = random.Random(seed)

    n = 80
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    titles = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]

    rows = []
    for i in range(n):
        dept = rng.choice(departments)
        title = rng.choice(titles)
        base_salary = {"Junior": 50000, "Mid": 70000, "Senior": 90000,
                       "Lead": 110000, "Manager": 130000, "Director": 160000}[title]
        salary = base_salary + rng.randint(-5000, 15000)
        hire_year = rng.randint(2015, 2024)
        perf = round(rng.uniform(1.0, 5.0), 1)
        rows.append({
            "emp_id": f"E{1000 + i}",
            "name": f"Employee_{i}",
            "department": dept,
            "title": title,
            "salary": salary,
            "hire_date": f"{hire_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "performance_score": perf,
            "manager_id": f"E{1000 + rng.randint(0, n-1)}",
            "email": f"emp{i}@company.com",
            "phone": f"+1-555-{rng.randint(1000,9999)}",
        })

    df = pd.DataFrame(rows)
    dirty = df.copy()

    # 1. Duplicate rows (8)
    dup_indices = rng.sample(range(n), 8)
    dups = dirty.iloc[dup_indices].copy()
    dirty = pd.concat([dirty, dups], ignore_index=True)
    total = len(dirty)

    # 2. Missing values scattered across columns
    for col in ["salary", "performance_score", "department", "phone"]:
        miss_count = rng.randint(3, 6)
        for idx in rng.sample(range(total), min(miss_count, total)):
            dirty.loc[idx, col] = None

    # 3. Inconsistent department names
    dept_variants = {
        "Engineering": ["engineering", "Eng", "ENGINEERING", "Engg"],
        "Marketing": ["marketing", "Mktg", "MARKETING"],
        "Sales": ["sales", "SALES"],
        "HR": ["hr", "Human Resources", "H.R."],
        "Finance": ["finance", "FIN", "FINANCE"],
    }
    for idx in range(total):
        dept = dirty.loc[idx, "department"]
        if dept in dept_variants and rng.random() < 0.35:
            dirty.loc[idx, "department"] = rng.choice(dept_variants[dept])

    # 4. Salary outliers (negative or absurdly high)
    for idx in rng.sample(range(total), 4):
        dirty.loc[idx, "salary"] = rng.choice([-5000, 0, 999999, 1500000])

    # 5. Performance scores out of range (should be 1.0-5.0)
    for idx in rng.sample(range(total), 3):
        dirty.loc[idx, "performance_score"] = rng.choice([-1.0, 0.0, 7.5, 10.0])

    # 6. Phone format inconsistency
    for idx in range(total):
        phone = dirty.loc[idx, "phone"]
        if phone is not None and rng.random() < 0.3:
            # strip formatting
            digits = re.sub(r"[^0-9]", "", str(phone))
            dirty.loc[idx, "phone"] = digits  # raw digits, no formatting

    # 7. Date format issues
    for idx in range(total):
        if rng.random() < 0.25:
            d = dirty.loc[idx, "hire_date"]
            if d and isinstance(d, str) and "-" in d:
                parts = d.split("-")
                dirty.loc[idx, "hire_date"] = f"{parts[2]}/{parts[1]}/{parts[0]}"

    # 8. Email formatting issues
    for idx in range(total):
        if rng.random() < 0.15:
            dirty.loc[idx, "email"] = dirty.loc[idx, "email"].upper() if dirty.loc[idx, "email"] else None

    # 9. Self-referencing manager_id
    for idx in rng.sample(range(total), 3):
        dirty.loc[idx, "manager_id"] = dirty.loc[idx, "emp_id"]

    target = df.copy()
    issues = {
        "duplicates": 8,
        "missing_values": True,
        "inconsistent_departments": True,
        "salary_outliers": 4,
        "performance_out_of_range": 3,
        "phone_format": True,
        "date_format": True,
        "email_case": True,
        "self_referencing_manager": 3,
    }
    return dirty, target, issues


# ====================================================================== #
#  Quality scoring helpers
# ====================================================================== #

def _compute_quality_score(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score 0.0–1.0 how close *current* is to *target*.

    Components:
    - Row count match (15%)
    - Column match (10%)
    - Missing value reduction (15%)
    - Row-level matching via best-effort alignment (60%)
    """
    score = 0.0

    # Row count similarity
    if len(target) > 0:
        row_ratio = 1.0 - abs(len(current) - len(target)) / max(len(target), 1)
        score += max(row_ratio, 0.0) * 0.15

    # Column match
    target_cols = set(target.columns)
    current_cols = set(current.columns)
    if target_cols:
        col_score = len(target_cols & current_cols) / len(target_cols)
        score += col_score * 0.10

    # Missing values — compare null fraction
    current_null_frac = current.isnull().sum().sum() / max(current.size, 1)
    null_improvement = 1.0 - min(current_null_frac, 1.0)
    score += null_improvement * 0.15

    # Row-level matching on shared columns
    shared_cols = sorted(target_cols & current_cols)
    if shared_cols and len(current) > 0 and len(target) > 0:
        def normalize_val(v):
            if pd.isna(v):
                return "__NULL__"
            s = str(v).strip().lower()
            # Normalize numeric strings: remove $ , and compare as float when possible
            cleaned = s.lstrip("$").replace(",", "")
            try:
                return f"__NUM_{float(cleaned):.4f}"
            except (ValueError, TypeError):
                return s

        def row_to_key(row):
            return tuple(normalize_val(row[c]) for c in shared_cols)

        target_rows = {}
        for i in range(len(target)):
            key = row_to_key(target.iloc[i])
            target_rows[key] = target_rows.get(key, 0) + 1

        matched = 0
        target_copy = dict(target_rows)
        for i in range(len(current)):
            key = row_to_key(current.iloc[i])
            if target_copy.get(key, 0) > 0:
                matched += 1
                target_copy[key] -= 1

        # Score based on fraction of target rows matched
        row_match_score = matched / len(target)
        score += row_match_score * 0.60

    return round(min(score, 1.0), 4)


def _detect_issues(df: pd.DataFrame, task_id: str) -> List[str]:
    """Return a human-readable list of quality issues detected."""
    issues: List[str] = []

    # Missing values
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} missing values")

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows detected")

    # Type issues — numeric columns stored as object
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                numeric_count = sum(1 for v in non_null if _is_numeric_string(str(v)))
                if numeric_count / len(non_null) > 0.7:
                    issues.append(f"Column '{col}' looks numeric but is stored as text")

    if not issues:
        issues.append("No obvious issues detected — consider submitting")
    return issues


def _is_numeric_string(s: str) -> bool:
    s = s.strip().lstrip("$").replace(",", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def _df_preview(df: pd.DataFrame, n: int = 8) -> str:
    buf = io.StringIO()
    df.head(n).to_string(buf, index=False)
    return buf.getvalue()


def _df_info(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", ""]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        uniq = int(df[col].nunique())
        lines.append(f"  {col:25s}  dtype={dtype:10s}  nulls={nulls}  unique={uniq}")
    return "\n".join(lines)


# ====================================================================== #
#  Task registry
# ====================================================================== #

TASKS = {
    "easy": {
        "id": "easy",
        "description": (
            "Clean a small customer dataset (20 rows, 5 columns). "
            "Issues: missing values in age & email, age stored as string, "
            "revenue column has '$' prefix. Fix all issues and submit."
        ),
        "generator": _generate_easy_dataset,
        "max_steps": MAX_STEPS_EASY,
    },
    "medium": {
        "id": "medium",
        "description": (
            "Clean a sales orders dataset (~55 rows, 7 columns). "
            "Issues: duplicate rows, inconsistent region names (e.g. 'north' vs 'North'), "
            "missing unit prices, extreme outliers in quantity, mixed date formats. "
            "Fix all issues and submit."
        ),
        "generator": _generate_medium_dataset,
        "max_steps": MAX_STEPS_MEDIUM,
    },
    "hard": {
        "id": "hard",
        "description": (
            "Clean an HR employee dataset (~88 rows, 10 columns). "
            "Issues: duplicates, missing values across multiple columns, inconsistent "
            "department names (e.g. 'Eng' vs 'Engineering'), salary outliers (negative/extreme), "
            "performance scores out of 1-5 range, phone format inconsistency, date format mix, "
            "email casing, self-referencing manager IDs. Fix all issues and submit."
        ),
        "generator": _generate_hard_dataset,
        "max_steps": MAX_STEPS_HARD,
    },
}


# ====================================================================== #
#  Environment
# ====================================================================== #

class DataCleanEnvironment(Environment):
    """Data cleaning RL environment.

    The agent receives a messy dataset and must issue cleaning commands
    to improve data quality, then submit for grading.
    """

    def __init__(self, task_id: str = "easy", seed: int = 42):
        self._task_id = task_id
        self._seed = seed
        self._state = DataCleanState()
        self._df: Optional[pd.DataFrame] = None
        self._target: Optional[pd.DataFrame] = None
        self._initial_issues: dict = {}
        self._max_steps = TASKS[task_id]["max_steps"]
        self._done = False
        self._prev_quality = 0.0

    # ---- OpenEnv interface ------------------------------------------------

    def reset(self) -> Observation:
        task = TASKS.get(self._task_id)
        if task is None:
            task = TASKS["easy"]
            self._task_id = "easy"

        generator = task["generator"]
        dirty, target, issues = generator(self._seed)

        self._df = dirty.copy()
        self._target = target.copy()
        self._initial_issues = issues
        self._max_steps = task["max_steps"]
        self._done = False

        quality = _compute_quality_score(self._df, self._target)
        self._prev_quality = quality

        self._state = DataCleanState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self._task_id,
            quality_score=quality,
        )

        detected = _detect_issues(self._df, self._task_id)

        return DataCleanObservation(
            dataset_preview=_df_preview(self._df),
            column_info=_df_info(self._df),
            message=f"Task '{self._task_id}' loaded. {task['description']}",
            quality_score=quality,
            issues_found=detected,
            task_id=self._task_id,
            task_description=task["description"],
            step_count=0,
            max_steps=self._max_steps,
            available_commands=AVAILABLE_COMMANDS,
            done=False,
            reward=0.0,
        )

    def step(self, action: Action) -> Observation:
        if not isinstance(action, DataCleanAction):
            raise ValueError(f"Expected DataCleanAction, got {type(action)}")

        if self._done:
            return self._make_obs("Episode already finished. Call reset().", reward=0.0, done=True)

        self._state.step_count += 1

        if self._state.step_count > self._max_steps:
            self._done = True
            quality = _compute_quality_score(self._df, self._target)
            return self._make_obs(
                f"Max steps ({self._max_steps}) exceeded. Final quality: {quality:.4f}",
                reward=quality * 0.5,  # partial credit
                done=True,
            )

        cmd = action.command.strip().lower()
        col = action.column
        params = action.params or {}

        # Dispatch commands
        handler = {
            "inspect": self._cmd_inspect,
            "fill_missing": self._cmd_fill_missing,
            "cast_type": self._cmd_cast_type,
            "remove_duplicates": self._cmd_remove_duplicates,
            "fix_format": self._cmd_fix_format,
            "filter_outliers": self._cmd_filter_outliers,
            "drop_rows": self._cmd_drop_rows,
            "rename_column": self._cmd_rename_column,
            "standardize": self._cmd_standardize,
            "submit": self._cmd_submit,
        }.get(cmd)

        if handler is None:
            return self._make_obs(
                f"Unknown command '{cmd}'. Use 'inspect' to see available commands.",
                reward=-0.01,
            )

        return handler(col, params)

    @property
    def state(self) -> DataCleanState:
        return self._state

    # ---- Observation helper -----------------------------------------------

    def _make_obs(self, message: str, reward: float = 0.0, done: bool = False) -> DataCleanObservation:
        quality = _compute_quality_score(self._df, self._target) if self._df is not None else 0.0
        self._state.quality_score = quality
        if done:
            self._done = True
        detected = _detect_issues(self._df, self._task_id) if self._df is not None else []

        return DataCleanObservation(
            dataset_preview=_df_preview(self._df) if self._df is not None else "",
            column_info=_df_info(self._df) if self._df is not None else "",
            message=message,
            quality_score=quality,
            issues_found=detected,
            task_id=self._task_id,
            task_description=TASKS.get(self._task_id, {}).get("description", ""),
            step_count=self._state.step_count,
            max_steps=self._max_steps,
            available_commands=AVAILABLE_COMMANDS,
            done=done,
            reward=reward,
        )

    # ---- Reward shaping ---------------------------------------------------

    def _quality_delta_reward(self) -> float:
        """Reward = improvement in quality since last step."""
        quality = _compute_quality_score(self._df, self._target)
        delta = quality - self._prev_quality
        self._prev_quality = quality
        # Scale up to make signal more useful
        return round(delta * 5.0, 4)

    # ---- Command implementations ------------------------------------------

    def _cmd_inspect(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if col and col in self._df.columns:
            info = f"Column '{col}':\n"
            info += f"  dtype: {self._df[col].dtype}\n"
            info += f"  nulls: {self._df[col].isnull().sum()}\n"
            info += f"  unique: {self._df[col].nunique()}\n"
            info += f"  sample values: {list(self._df[col].dropna().head(5))}\n"
            if self._df[col].dtype in ("int64", "float64"):
                info += f"  min: {self._df[col].min()}, max: {self._df[col].max()}, mean: {self._df[col].mean():.2f}\n"
            msg = info
        else:
            msg = "Dataset overview:\n" + _df_info(self._df)
        return self._make_obs(msg, reward=0.0)

    def _cmd_fill_missing(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        strategy = params.get("strategy", "mean")
        null_before = int(self._df[col].isnull().sum())
        if null_before == 0:
            return self._make_obs(f"No missing values in '{col}'.", reward=-0.01)

        try:
            if strategy == "mean":
                numeric = pd.to_numeric(self._df[col], errors="coerce")
                fill_val = numeric.mean()
                self._df[col] = numeric.fillna(fill_val)
            elif strategy == "median":
                numeric = pd.to_numeric(self._df[col], errors="coerce")
                fill_val = numeric.median()
                self._df[col] = numeric.fillna(fill_val)
            elif strategy == "mode":
                mode_val = self._df[col].mode()
                if len(mode_val) > 0:
                    self._df[col] = self._df[col].fillna(mode_val.iloc[0])
            elif strategy == "value":
                fill_val = params.get("value", "")
                self._df[col] = self._df[col].fillna(fill_val)
            else:
                return self._make_obs(f"Unknown strategy '{strategy}'.", reward=-0.01)
        except Exception as e:
            return self._make_obs(f"Error filling '{col}': {e}", reward=-0.02)

        null_after = int(self._df[col].isnull().sum())
        filled = null_before - null_after
        reward = self._quality_delta_reward()
        return self._make_obs(
            f"Filled {filled} missing values in '{col}' using strategy='{strategy}'.",
            reward=reward,
        )

    def _cmd_cast_type(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        dtype = params.get("dtype", "float")
        try:
            if dtype == "int":
                self._df[col] = pd.to_numeric(self._df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").astype("Int64")
            elif dtype == "float":
                self._df[col] = pd.to_numeric(self._df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
            elif dtype == "str":
                self._df[col] = self._df[col].astype(str)
            elif dtype == "datetime":
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce", format="mixed")
            else:
                return self._make_obs(f"Unsupported dtype '{dtype}'.", reward=-0.01)
        except Exception as e:
            return self._make_obs(f"Error casting '{col}' to {dtype}: {e}", reward=-0.02)

        reward = self._quality_delta_reward()
        return self._make_obs(f"Cast '{col}' to {dtype}.", reward=reward)

    def _cmd_remove_duplicates(self, col: Optional[str], params: dict) -> DataCleanObservation:
        before = len(self._df)
        self._df = self._df.drop_duplicates().reset_index(drop=True)
        after = len(self._df)
        removed = before - after
        reward = self._quality_delta_reward()
        return self._make_obs(f"Removed {removed} duplicate rows.", reward=reward)

    def _cmd_fix_format(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        pattern = params.get("pattern")
        replacement = params.get("replacement", "")
        if not pattern:
            return self._make_obs("Missing 'pattern' parameter.", reward=-0.01)

        try:
            mask = self._df[col].notna()
            self._df.loc[mask, col] = self._df.loc[mask, col].astype(str).str.replace(
                pattern, replacement, regex=True
            )
        except Exception as e:
            return self._make_obs(f"Error in fix_format: {e}", reward=-0.02)

        reward = self._quality_delta_reward()
        return self._make_obs(f"Applied regex fix on '{col}'.", reward=reward)

    def _cmd_filter_outliers(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        method = params.get("method", "iqr")
        threshold = float(params.get("threshold", 1.5))

        try:
            numeric = pd.to_numeric(self._df[col], errors="coerce")
            before = len(self._df)

            if method == "iqr":
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask = (numeric >= lower) & (numeric <= upper) | numeric.isna()
            elif method == "zscore":
                mean = numeric.mean()
                std = numeric.std()
                if std == 0:
                    mask = pd.Series([True] * len(self._df))
                else:
                    z = (numeric - mean).abs() / std
                    mask = (z <= threshold) | numeric.isna()
            else:
                return self._make_obs(f"Unknown method '{method}'.", reward=-0.01)

            self._df = self._df[mask].reset_index(drop=True)
            removed = before - len(self._df)
        except Exception as e:
            return self._make_obs(f"Error filtering outliers: {e}", reward=-0.02)

        reward = self._quality_delta_reward()
        return self._make_obs(f"Removed {removed} outlier rows from '{col}'.", reward=reward)

    def _cmd_drop_rows(self, col: Optional[str], params: dict) -> DataCleanObservation:
        condition = params.get("condition")
        if not condition:
            return self._make_obs("Missing 'condition' parameter.", reward=-0.01)

        # Sanitize: only allow simple pandas query expressions
        # Block dangerous patterns
        forbidden = ["import", "exec", "eval", "os.", "sys.", "__", "open(", "subprocess"]
        condition_lower = condition.lower()
        for f in forbidden:
            if f in condition_lower:
                return self._make_obs(f"Forbidden pattern in condition: '{f}'", reward=-0.05)

        try:
            before = len(self._df)
            mask = self._df.eval(condition)
            self._df = self._df[~mask].reset_index(drop=True)
            removed = before - len(self._df)
        except Exception as e:
            return self._make_obs(f"Error in drop_rows: {e}", reward=-0.02)

        reward = self._quality_delta_reward()
        return self._make_obs(f"Dropped {removed} rows matching condition.", reward=reward)

    def _cmd_rename_column(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        new_name = params.get("new_name")
        if not new_name:
            return self._make_obs("Missing 'new_name' parameter.", reward=-0.01)

        self._df = self._df.rename(columns={col: new_name})
        reward = self._quality_delta_reward()
        return self._make_obs(f"Renamed '{col}' to '{new_name}'.", reward=reward)

    def _cmd_standardize(self, col: Optional[str], params: dict) -> DataCleanObservation:
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        mapping = params.get("mapping")
        if not mapping or not isinstance(mapping, dict):
            return self._make_obs("Missing or invalid 'mapping' parameter (should be dict).", reward=-0.01)

        try:
            # Apply case-insensitive mapping
            def apply_mapping(val):
                if pd.isna(val):
                    return val
                val_str = str(val).strip()
                for old_val, new_val in mapping.items():
                    if val_str.lower() == str(old_val).strip().lower():
                        return new_val
                return val_str

            self._df[col] = self._df[col].apply(apply_mapping)
        except Exception as e:
            return self._make_obs(f"Error in standardize: {e}", reward=-0.02)

        reward = self._quality_delta_reward()
        return self._make_obs(f"Standardized values in '{col}'.", reward=reward)

    def _cmd_submit(self, col: Optional[str], params: dict) -> DataCleanObservation:
        quality = _compute_quality_score(self._df, self._target)
        self._done = True

        # Final reward is the quality score itself
        return self._make_obs(
            f"Dataset submitted! Final quality score: {quality:.4f}",
            reward=quality,
            done=True,
        )
