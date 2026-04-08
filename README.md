---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🧹 Data Cleaning Environment

> **OpenEnv RL Hackathon Submission** — A complete, spec-compliant environment for training and evaluating RL agents on real-world tabular data cleaning tasks.

An OpenEnv environment that simulates **real-world tabular data cleaning** — the task that occupies ~80% of a data scientist's time. An RL agent receives a messy dataset and must issue cleaning commands to improve data quality, then submit for grading.

## Why Data Cleaning?

| Aspect | Detail |
|--------|--------|
| **Real-world utility** | Data cleaning is the #1 bottleneck in every ML pipeline. Training agents to automate it has immediate production value. |
| **Rich action space** | 11 distinct cleaning commands with parameters — far more nuanced than binary or discrete actions. |
| **Dense reward signal** | 5-component quality scoring with column-level partial credit; shaped reward improves incrementally after each successful fix. |
| **Difficulty progression** | Three tasks: simple (5-col, 20-row) → complex (10-col, 88-row with referential integrity). |

---

## Project Structure

```
meta-RL-Hackathon/
├── Dockerfile                      # Root Dockerfile — builds the HF Space server
├── README.md                       # This file (also doubles as HF Space card)
├── inference.py                    # Baseline inference script (hackathon entry point)
├── openenv.yaml                    # Root OpenEnv manifest for validation and deployment
├── pyproject.toml                  # Root package metadata for the OpenEnv environment
├── test_local.py                   # Local validation test suite (no server needed)
├── .dockerignore
├── .gitignore
├── server/                         # Root server wrapper for OpenEnv validation
│   ├── __init__.py
│   └── app.py                      # Wrapper exposing the nested FastAPI app
└── data_clean_env/                 # OpenEnv environment package
    ├── __init__.py                 # Package exports (DataCleanEnv, DataCleanAction, ...)
    ├── models.py                   # Typed Pydantic models: Action, Observation, State
    ├── client.py                   # DataCleanEnv — EnvClient WebSocket wrapper
    ├── openenv.yaml                # Nested OpenEnv manifest
    ├── pyproject.toml              # Package metadata & dependencies
    └── server/
        ├── __init__.py
        ├── app.py                  # FastAPI application (create_app)
        └── environment.py          # DataCleanEnvironment — all core logic & graders
```

---

## Prerequisites

- Python 3.10 or 3.11
- pip
- Docker (for container builds)
- Git

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gokarna12242-ai/meta-RL-Hackathon.git
cd meta-RL-Hackathon
```

### 2. Install Dependencies

**Option A — install the environment package in editable mode** (recommended for development):

```bash
pip install "openenv-core[core]>=0.2.2"
pip install -e data_clean_env/
```

**Option B — install dependencies directly** (lighter, no package install):

```bash
pip install "openenv-core[core]>=0.2.2" fastapi "uvicorn[standard]" pydantic pandas openai
```

---

## Configuration

`inference.py` reads three environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for inference |
| `HF_TOKEN` / `OPENAI_API_KEY` | **Yes** | — | API token (mandatory — raises `ValueError` if missing) |

**`HF_TOKEN` (or `OPENAI_API_KEY`) must be set.** The script raises `ValueError` if neither is provided, per hackathon guidelines. The LLM drives the agent; if an individual API call fails, the script falls back to a deterministic scripted action for that step only.

Set variables on Linux/macOS:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
```

Set variables on Windows (PowerShell):

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN     = "your-token-here"
```

---

## Running the Server

The environment exposes a FastAPI server that the client wrapper connects to via WebSocket.

### Local (no Docker)

```bash
# From the repo root
uvicorn data_clean_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`. Health check: `GET /health`.

### Docker (root Dockerfile)

```bash
# Build
docker build -t data-clean-env:latest .

# Run
docker run -d -p 8000:8000 --name data-clean-env data-clean-env:latest

# Verify
curl http://localhost:8000/health

# Logs
docker logs data-clean-env

# Stop
docker stop data-clean-env && docker rm data-clean-env
```

---

## Running Inference

`inference.py` runs the agent (LLM or scripted) against all three tasks and prints results in the required hackathon format.

```bash
python inference.py
```

**Expected output format:**

```
[START] task=easy env=data_clean_env model=scripted
[STEP] step=1 action=inspect reward=0.01 done=false error=null
[STEP] step=2 action=fix_format(revenue) reward=0.00 done=false error=null
...
[END] success=true steps=7 score=0.905 rewards=0.01,0.00,0.00,0.00,0.01,0.01,0.91
```

One `[START]` line per task, one `[STEP]` line per environment step, one `[END]` line when the episode finishes.

---

## Running Tests

The local validation suite runs all three tasks and edge cases entirely in-process — no server, no Docker, no API keys needed.

```bash
python test_local.py
```

**Expected output:**

```
============================================================
  Data Cleaning Environment -- Local Validation
============================================================

[TEST] Easy task ...   Submit OK. Final quality=0.9050
[TEST] Medium task ... Submit OK. Final quality=0.9449
[TEST] Hard task ...   Submit OK. Final quality=0.8908
[TEST] Edge cases ...  All edge cases passed!

  easy    : 0.9050  [PASS]
  medium  : 0.9449  [PASS]
  hard    : 0.8908  [PASS]
  average : 0.9136

  ALL TESTS PASSED!
```

---

## Using the Client API

With the server running, use the async or sync `DataCleanEnv` client:

### Async

```python
import asyncio
from data_clean_env import DataCleanEnv, DataCleanAction

async def main():
    async with DataCleanEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(result.observation.message)

        result = await env.step(DataCleanAction(command="inspect"))
        print(result.observation.column_info)

        result = await env.step(DataCleanAction(
            command="fill_missing",
            column="age",
            params={"strategy": "median"}
        ))
        print(f"Quality: {result.observation.quality_score:.4f}")

asyncio.run(main())
```

### Sync

```python
from data_clean_env import DataCleanEnv, DataCleanAction

with DataCleanEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(DataCleanAction(command="inspect"))
    print(result.observation.column_info)
```

### In-process (no server)

```python
from data_clean_env.server.environment import DataCleanEnvironment
from data_clean_env.models import DataCleanAction

env = DataCleanEnvironment(task_id="easy", seed=42)
obs = env.reset()
obs = env.step(DataCleanAction(command="inspect"))
print(obs.quality_score)
```

---

## Action Space

| Command | Column | Params | Description |
|---------|--------|--------|-------------|
| `inspect` | optional | — | Show dataset shape, dtypes, null counts, sample values |
| `fill_missing` | required | `strategy`: `mean`/`median`/`mode`/`value`; `value` (for strategy=value) | Fill null values in a column |
| `cast_type` | required | `dtype`: `int`/`float`/`str`/`datetime` | Cast column to a different type |
| `remove_duplicates` | — | — | Drop all exact duplicate rows |
| `fix_format` | required | `pattern` (regex), `replacement`, `case` (`lower`/`upper`/`title`) | Regex string replace or case normalization on a column |
| `filter_outliers` | required | `method`: `iqr`/`zscore`; `threshold` | Remove rows where column value is an outlier |
| `drop_rows` | — | `condition` (pandas query expression) | Drop rows matching a boolean condition |
| `rename_column` | required | `new_name` | Rename a column |
| `standardize` | required | `mapping`: `{"old": "new", ...}` | Map variant spellings to a canonical value |
| `clip_values` | required | `min`, `max` | Clip numeric values to a valid range (keeps rows, adjusts values) |
| `submit` | — | — | Submit dataset for final grading (ends episode) |

```python
class DataCleanAction(Action):
    command: str        # command name (see table above)
    column: str | None  # target column — required for most commands
    params: dict        # additional parameters as shown above
```

---

## Observation Space

```python
class DataCleanObservation(Observation):
    dataset_preview: str        # First 8 rows as a formatted text table
    column_info: str            # Shape, dtype, null count, unique count per column
    message: str                # Human-readable feedback from last action
    quality_score: float        # Current quality score 0.0-1.0
    issues_found: list[str]     # Auto-detected issues with actionable fix hints
    task_id: str                # "easy" | "medium" | "hard"
    task_description: str       # Full task objective text
    step_count: int             # Steps taken so far
    max_steps: int              # Episode step budget
    available_commands: str     # Full command reference with JSON examples
    suggested_actions: list[str] # Prioritized JSON action suggestions
    done: bool                  # True when episode is finished
    reward: float               # Reward for the last action
```

---

## Tasks

### Task 1: Easy — Customer Dataset
- **Size:** 20 rows × 5 columns (`name`, `age`, `email`, `revenue`, `signup_date`)
- **Issues:** 3 missing ages, 2 missing emails, `age` stored as string, `revenue` has `$` prefix
- **Max steps:** 15
- **Difficulty:** Beginner

### Task 2: Medium — Sales Orders
- **Size:** ~55 rows × 7 columns (`order_id`, `product`, `quantity`, `unit_price`, `region`, `order_date`, `customer_id`)
- **Issues:** 5 duplicate rows, inconsistent region names (`north`/`NORTH`/`N.`), 4 missing unit prices, 3 extreme quantity outliers, mixed date formats
- **Max steps:** 25
- **Difficulty:** Intermediate

### Task 3: Hard — HR Employee Dataset
- **Size:** ~88 rows × 10 columns (`emp_id`, `name`, `department`, `title`, `salary`, `hire_date`, `performance_score`, `manager_id`, `email`, `phone`)
- **Issues:** 8 duplicates, missing salary/performance/department/phone, inconsistent department names, salary outliers (negative/extreme), performance scores outside 1–5, phone format mismatch, date format mix, email casing, self-referencing manager IDs
- **Max steps:** 40
- **Difficulty:** Advanced

---

## Reward Design

| Signal | Amount | Trigger |
|--------|--------|---------|
| First inspect | `+0.005` | Encourages initial exploration |
| Quality improvement | `+delta × 5.0` | Any action that raises the quality score |
| Quality degradation | `-delta × 5.0` | Any action that lowers the quality score |
| Unknown command | `-0.01` | Command name not recognised |
| Missing parameter | `-0.01` to `-0.02` | Required column or param not provided |
| Security violation | `-0.05` | Forbidden pattern in `drop_rows` condition |
| Submit | `quality_score` | Final episode reward on explicit submit |
| Timeout | `quality × 0.5` | Episode ended by exceeding `max_steps` |

Reward is dense — every step returns a signal proportional to its effect on data quality, not just the final submit.

---

## Quality Scoring (5 Components)

| Component | Weight | Description |
|-----------|--------|-------------|
| Row count match | 10% | Penalizes wrong number of rows vs target |
| Column match | 10% | Fraction of target columns present |
| Missing value reduction | 10% | Lower null fraction = higher score |
| Exact row matching | 40% | Row-level tuple matching against target |
| Column-level value matching | 30% | Per-column multiset overlap for partial credit |

---

## Baseline Scores (Scripted Policy, seed=42)

| Task | Final Quality | Steps | Success |
|------|--------------|-------|---------|
| Easy | **0.9050** | 7 | true |
| Medium | **0.9449** | 7 | true |
| Hard | **0.8908** | 14 | true |
| **Average** | **0.9136** | — | 3 / 3 |

All scores are deterministic (fixed seed). LLM-based agents with frontier models are expected to score higher.

---

## OpenEnv Spec Compliance

| Requirement | Status |
|-------------|--------|
| `openenv.yaml` with spec metadata | ✅ `data_clean_env/openenv.yaml` |
| Typed `Action` Pydantic model | ✅ `DataCleanAction` |
| Typed `Observation` Pydantic model | ✅ `DataCleanObservation` |
| Typed `State` Pydantic model | ✅ `DataCleanState` |
| `reset()` returns clean initial observation | ✅ |
| `step(action)` returns observation + reward + done | ✅ |
| `state` property returns current state | ✅ |
| 3+ tasks with graders scoring 0.0–1.0 | ✅ easy / medium / hard |
| Dense reward function | ✅ per-step quality delta |
| FastAPI server via `create_app` | ✅ `data_clean_env/server/app.py` |
| Working Dockerfile | ✅ root `Dockerfile` |

---

## License

BSD 3-Clause License
