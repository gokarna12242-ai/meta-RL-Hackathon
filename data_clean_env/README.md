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

An OpenEnv environment that simulates **real-world tabular data cleaning** — the task that occupies ~80% of a data scientist's time. An RL agent receives a messy dataset and must issue cleaning commands to improve data quality, then submit for grading.

## Why Data Cleaning?

| Aspect | Detail |
|--------|--------|
| **Real-world utility** | Data cleaning is the #1 bottleneck in every ML pipeline. Training agents to clean data has immediate production value. |
| **Rich action space** | 10 distinct cleaning commands with parameters — far more nuanced than binary actions. |
| **Continuous reward signal** | Quality score improves incrementally with each successful fix, providing dense reward. |
| **Difficulty progression** | Three tasks from simple (5-col, 20-row) to complex (10-col, 88-row with referential integrity). |

## Quick Start

### Install

```bash
pip install openenv-core[core]>=0.2.2
pip install git+https://huggingface.co/spaces/<username>/data-clean-env
```

### Usage (async)

```python
import asyncio
from data_clean_env import DataCleanEnv, DataCleanAction

async def main():
    async with DataCleanEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(result.observation.message)
        
        # Inspect the data
        result = await env.step(DataCleanAction(command="inspect"))
        print(result.observation.column_info)
        
        # Fix missing values
        result = await env.step(DataCleanAction(
            command="fill_missing",
            column="age",
            params={"strategy": "median"}
        ))
        print(f"Quality: {result.observation.quality_score}")

asyncio.run(main())
```

### Usage (sync)

```python
from data_clean_env import DataCleanEnv, DataCleanAction

with DataCleanEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(DataCleanAction(command="inspect"))
    print(result.observation.column_info)
```

## Action Space

| Command | Parameters | Description |
|---------|-----------|-------------|
| `inspect` | `column` (optional) | View dataset info, column types, null counts |
| `fill_missing` | `column`, `params.strategy` (mean/median/mode/value) | Fill missing values |
| `cast_type` | `column`, `params.dtype` (int/float/str/datetime) | Cast column type |
| `remove_duplicates` | — | Remove duplicate rows |
| `fix_format` | `column`, `params.pattern`, `params.replacement` | Regex-based string cleaning |
| `filter_outliers` | `column`, `params.method` (iqr/zscore), `params.threshold` | Remove outlier rows |
| `drop_rows` | `params.condition` (pandas query) | Drop rows matching condition |
| `rename_column` | `column`, `params.new_name` | Rename a column |
| `standardize` | `column`, `params.mapping` (dict) | Map inconsistent values to standard ones |
| `submit` | — | Submit cleaned dataset for final grading |

### DataCleanAction

```python
class DataCleanAction(Action):
    command: str        # One of the commands above
    column: str | None  # Target column (required for most commands)
    params: dict        # Additional parameters
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `dataset_preview` | str | Text preview of current dataset (first 8 rows) |
| `column_info` | str | Column types, null counts, unique counts |
| `message` | str | Feedback from last action |
| `quality_score` | float | Current data quality (0.0–1.0) |
| `issues_found` | list[str] | Remaining quality issues |
| `task_id` | str | Current task (easy/medium/hard) |
| `task_description` | str | Task objective description |
| `step_count` | int | Current step number |
| `max_steps` | int | Maximum allowed steps |
| `available_commands` | str | Help text for commands |

## Tasks

### Task 1: Easy — Customer Dataset (20 rows, 5 columns)

**Issues to fix:**
- Missing values in `age` (3 rows) and `email` (2 rows)
- `age` column stored as string instead of integer
- `revenue` column has `$` prefix (e.g., `$1234.56`)

**Max steps:** 15 | **Expected difficulty:** Beginner

### Task 2: Medium — Sales Orders (55 rows, 7 columns)

**Issues to fix:**
- 5 duplicate rows
- Inconsistent region names (`north`, `NORTH`, `N.` → `North`)
- 4 missing unit prices
- 3 extreme outliers in quantity (9000–99999)
- Mixed date formats (`2024-01-15` vs `01/15/2024`)

**Max steps:** 25 | **Expected difficulty:** Intermediate

### Task 3: Hard — HR Employee Dataset (88 rows, 10 columns)

**Issues to fix:**
- 8 duplicate rows
- Missing values across salary, performance, department, phone
- Inconsistent department names (`Eng`, `engineering`, `ENGINEERING` → `Engineering`)
- Salary outliers (negative, zero, $1.5M)
- Performance scores outside 1.0–5.0 range
- Phone number format inconsistency
- Date format mix (`YYYY-MM-DD` vs `DD/MM/YYYY`)
- Email casing inconsistency
- Self-referencing manager IDs

**Max steps:** 40 | **Expected difficulty:** Advanced

## Reward Design

| Signal | Value | When |
|--------|-------|------|
| Quality improvement | `+Δ × 5.0` | Each action that improves quality |
| Quality decrease | `-Δ × 5.0` | Actions that hurt quality |
| Invalid command | `-0.01` | Unknown command |
| Missing parameter | `-0.01` to `-0.02` | Required parameter not provided |
| Forbidden operation | `-0.05` | Security violation in query |
| Submit | `quality_score` | Final submission reward |
| Timeout | `quality × 0.5` | Exceeded max steps |

The reward function provides **continuous signal over the full trajectory**, not just binary end-of-episode. Each cleaning action yields immediate feedback proportional to its impact on data quality.

## Setup & Development

### Local (no Docker)

```bash
cd data_clean_env
pip install -e .
uvicorn data_clean_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
cd data_clean_env
docker build -t data-clean-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 data-clean-env:latest
```

### Run baseline inference

```bash
export HF_TOKEN=your-token
export OPENAI_API_KEY=your-token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
python inference.py
```

**No API key?** The script falls back to a deterministic scripted policy — no external tokens required for a reproducible run.

## Baseline Scores

| Task | Score | Steps Used |
|------|-------|-----------|
| Easy | ~0.85 | 5–8 |
| Medium | ~0.70 | 10–15 |
| Hard | ~0.55 | 20–30 |

*(Scores from gpt-4o-mini baseline. Frontier models score higher.)*

## Project Structure

```
data_clean_env/
├── __init__.py              # Package exports
├── models.py                # DataCleanAction, DataCleanObservation, DataCleanState
├── client.py                # DataCleanEnv (EnvClient implementation)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package config & dependencies
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── environment.py       # DataCleanEnvironment (core logic)
    ├── app.py               # FastAPI application
    └── Dockerfile           # Container image
inference.py                 # Baseline inference script
```

## License

BSD 3-Clause License
