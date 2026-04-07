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
Respond with a JSON action: {"command": "<name>", "column": "<col_or_null>", "params": {}}

Commands (with examples):
  inspect — View dataset info or a specific column
    {"command": "inspect"}
    {"command": "inspect", "column": "age"}

  fill_missing — Fill null/missing values in a column
    {"command": "fill_missing", "column": "age", "params": {"strategy": "median"}}
    Strategies: mean, median, mode, value (requires "value" param)

  cast_type — Convert column to a new data type
    {"command": "cast_type", "column": "revenue", "params": {"dtype": "float"}}
    Types: int, float, str, datetime

  remove_duplicates — Remove duplicate rows
    {"command": "remove_duplicates"}

  fix_format — Regex-based string cleaning or case normalization
    {"command": "fix_format", "column": "email", "params": {"case": "lower"}}
    {"command": "fix_format", "column": "price", "params": {"pattern": "^\\\\$", "replacement": ""}}
    {"command": "fix_format", "column": "date_col", "params": {"pattern": "(\\\\d{2})/(\\\\d{2})/(\\\\d{4})", "replacement": "\\\\3-\\\\1-\\\\2"}}

  filter_outliers — Remove rows containing outlier values
    {"command": "filter_outliers", "column": "salary", "params": {"method": "iqr", "threshold": 1.5}}
    Methods: iqr, zscore

  drop_rows — Drop rows matching a pandas query condition
    {"command": "drop_rows", "params": {"condition": "age < 0"}}

  rename_column — Rename a column
    {"command": "rename_column", "column": "old_name", "params": {"new_name": "new_name"}}

  standardize — Map inconsistent categorical values to standard ones
    {"command": "standardize", "column": "region", "params": {"mapping": {"N": "North", "NORTH": "North"}}}

  clip_values — Clip numeric values to a valid min/max range (keeps rows, adjusts values)
    {"command": "clip_values", "column": "score", "params": {"min": 1.0, "max": 5.0}}

  submit — Submit the cleaned dataset for final grading
    {"command": "submit"}

Strategy: inspect → identify issues → fix one by one → submit when quality is high.
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

    # 9. Inconsistent title formatting
    title_variants = {
        "Junior": ["junior", "Jr", "Jr."],
        "Mid": ["mid", "Mid-level", "MID"],
        "Senior": ["senior", "Sr", "Sr."],
        "Lead": ["lead", "Lead Engineer", "LEAD"],
        "Manager": ["manager", "Mgr", "Managerial"],
        "Director": ["director", "Dir", "DIRECTOR"],
    }
    for idx in range(total):
        title = dirty.loc[idx, "title"]
        if title in title_variants and rng.random() < 0.25:
            dirty.loc[idx, "title"] = rng.choice(title_variants[title])

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
        "inconsistent_titles": True,
    }
    return dirty, target, issues


# ====================================================================== #
#  Quality scoring helpers
# ====================================================================== #

def _compute_quality_score(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Score 0.0–1.0 how close *current* is to *target*.

    Components:
    - Row count match (10%)
    - Column match (10%)
    - Missing value reduction (10%)
    - Row-level exact matching via best-effort alignment (40%)
    - Column-level value matching for partial credit (30%)
    """
    score = 0.0

    # Row count similarity
    if len(target) > 0:
        row_ratio = 1.0 - abs(len(current) - len(target)) / max(len(target), 1)
        score += max(row_ratio, 0.0) * 0.10

    # Column match
    target_cols = set(target.columns)
    current_cols = set(current.columns)
    if target_cols:
        col_score = len(target_cols & current_cols) / len(target_cols)
        score += col_score * 0.10

    # Missing values — compare null fraction
    current_null_frac = current.isnull().sum().sum() / max(current.size, 1)
    null_improvement = 1.0 - min(current_null_frac, 1.0)
    score += null_improvement * 0.10

    # Shared columns needed for row and column matching
    shared_cols = sorted(target_cols & current_cols)

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

    # Row-level matching on shared columns (40%)
    if shared_cols and len(current) > 0 and len(target) > 0:
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
        score += row_match_score * 0.40

    # Column-level value matching (30%) — partial credit per column
    if shared_cols and len(current) > 0 and len(target) > 0:
        col_scores = []
        for col in shared_cols:
            target_val_counts: dict = {}
            for v in target[col]:
                nv = normalize_val(v)
                target_val_counts[nv] = target_val_counts.get(nv, 0) + 1

            current_val_counts: dict = {}
            for v in current[col]:
                nv = normalize_val(v)
                current_val_counts[nv] = current_val_counts.get(nv, 0) + 1

            overlap = sum(
                min(target_val_counts[v], current_val_counts.get(v, 0))
                for v in target_val_counts
            )
            col_scores.append(overlap / len(target))

        avg_col = sum(col_scores) / len(col_scores) if col_scores else 0.0
        score += avg_col * 0.30

    return round(min(score, 1.0), 4)


def _detect_issues(df: pd.DataFrame, task_id: str) -> List[str]:
    """Return a human-readable list of quality issues detected with fix hints."""
    issues: List[str] = []

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows detected → use remove_duplicates")

    # Missing values
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            if df[col].dtype in ("int64", "float64"):
                issues.append(f"Column '{col}' has {null_count} missing values → fill_missing with strategy='median'")
            else:
                issues.append(f"Column '{col}' has {null_count} missing values → fill_missing with strategy='mode' or 'value'")

    # Type issues — numeric columns stored as object
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                numeric_count = sum(1 for v in non_null if _is_numeric_string(str(v)))
                if numeric_count / len(non_null) > 0.7:
                    # Check if values have $ prefix
                    dollar_count = sum(1 for v in non_null if str(v).strip().startswith("$"))
                    if dollar_count > len(non_null) * 0.3:
                        issues.append(f"Column '{col}' has '$' prefix on numeric values → fix_format to remove '$', then cast_type to float")
                    else:
                        issues.append(f"Column '{col}' looks numeric but is stored as text → cast_type with dtype='float' or 'int'")

        # Inconsistent string values (potential standardization needed)
        if df[col].dtype == object and col in ("region", "department", "title", "status", "category"):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                # Check for case variations
                lower_vals = set(str(v).strip().lower() for v in unique_vals)
                if len(lower_vals) < len(unique_vals):
                    issues.append(f"Column '{col}' has inconsistent values (case/abbreviation variants) → use standardize with a mapping")

        # Phone format check
        if col == "phone":
            invalid_phones = [v for v in df[col].dropna().astype(str) if not _is_standard_phone_format(v)]
            if invalid_phones:
                issues.append(f"Column 'phone' has {len(invalid_phones)} values with inconsistent formatting → fix_format with regex")

        # Title format check
        if col == "title":
            title_values = [str(v).strip() for v in df[col].dropna()]
            inconsistent = any(_is_inconsistent_title(v) for v in title_values)
            if inconsistent:
                issues.append(f"Column 'title' has inconsistent role/title formatting → use standardize with a mapping")

        # Email case check
        if col == "email":
            emails = df[col].dropna().astype(str)
            upper_emails = sum(1 for e in emails if e != e.lower())
            if upper_emails > 0:
                issues.append(f"Column 'email' has {upper_emails} values with uppercase characters → fix_format with case='lower'")

        # Date format inconsistency
        if col in ("order_date", "hire_date", "signup_date", "date"):
            date_vals = df[col].dropna().astype(str)
            slash_dates = sum(1 for v in date_vals if "/" in v)
            if slash_dates > 0:
                issues.append(f"Column '{col}' has {slash_dates} dates with inconsistent format (contains '/') → fix_format with regex to standardize to YYYY-MM-DD")

    # Outlier check for numeric columns
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
            if outliers > 0:
                issues.append(f"Column '{col}' has {int(outliers)} extreme outlier values → filter_outliers with method='iqr'")

    if not issues:
        issues.append("No obvious issues detected — consider submitting with {\"command\": \"submit\"}")
    return issues


def _suggest_next_actions(df: pd.DataFrame, task_id: str) -> List[str]:
    """Return a prioritized list of suggested JSON actions based on current data state."""
    suggestions: List[str] = []

    # Priority 1: Remove duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        suggestions.append('{"command": "remove_duplicates"}')

    # Priority 2: Fix format issues (dates, currency symbols, case)
    for col in df.columns:
        if col in ("order_date", "hire_date", "signup_date", "date"):
            date_vals = df[col].dropna().astype(str)
            slash_dates = sum(1 for v in date_vals if "/" in v)
            if slash_dates > 0:
                suggestions.append(f'{{"command": "fix_format", "column": "{col}", "params": {{"pattern": "(\\\\d{{2}})/(\\\\d{{2}})/(\\\\d{{4}})", "replacement": "\\\\3-\\\\2-\\\\1"}}}}')

        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                dollar_count = sum(1 for v in non_null if str(v).strip().startswith("$"))
                if dollar_count > len(non_null) * 0.3:
                    suggestions.append(f'{{"command": "fix_format", "column": "{col}", "params": {{"pattern": "^\\\\$", "replacement": ""}}}}')

        if col == "email":
            emails = df[col].dropna().astype(str)
            upper_emails = sum(1 for e in emails if e != e.lower())
            if upper_emails > 0:
                suggestions.append(f'{{"command": "fix_format", "column": "{col}", "params": {{"case": "lower"}}}}')

    # Priority 3: Standardize inconsistent categorical values
    for col in df.columns:
        if df[col].dtype == object and col in ("region", "department", "title", "status", "category"):
            unique_vals = df[col].dropna().unique()
            lower_vals = set(str(v).strip().lower() for v in unique_vals)
            if len(lower_vals) < len(unique_vals):
                suggestions.append(f'{{"command": "standardize", "column": "{col}", "params": {{"mapping": {{...}}}}}}')

    # Priority 4: Cast type for numeric-looking text columns
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                numeric_count = sum(1 for v in non_null if _is_numeric_string(str(v)))
                if numeric_count / len(non_null) > 0.7:
                    suggestions.append(f'{{"command": "cast_type", "column": "{col}", "params": {{"dtype": "float"}}}}')

    # Priority 5: Filter outliers
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
            if outliers > 0:
                suggestions.append(f'{{"command": "filter_outliers", "column": "{col}", "params": {{"method": "iqr", "threshold": 1.5}}}}')

    # Priority 6: Fill missing values
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            if df[col].dtype in ("int64", "float64"):
                suggestions.append(f'{{"command": "fill_missing", "column": "{col}", "params": {{"strategy": "median"}}}}')
            else:
                suggestions.append(f'{{"command": "fill_missing", "column": "{col}", "params": {{"strategy": "mode"}}}}')

    # If nothing else, suggest submit
    if not suggestions:
        suggestions.append('{"command": "submit"}')

    return suggestions


def _is_standard_phone_format(s: str) -> bool:
    return bool(re.match(r"^\+1-555-\d{4}$", s))


def _is_inconsistent_title(s: str) -> bool:
    normalized = str(s).strip().lower()
    canonical_titles = {"junior", "mid", "senior", "lead", "manager", "director"}
    alternate_titles = {
        "jr", "jr.", "mid-level", "sr", "sr.", "lead engineer",
        "mgr", "managerial", "dir",
    }
    return normalized in alternate_titles and normalized not in canonical_titles


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
        self._inspected = False

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
        self._inspected = False

        quality = _compute_quality_score(self._df, self._target)
        self._prev_quality = quality

        self._state = DataCleanState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self._task_id,
            quality_score=quality,
        )

        detected = _detect_issues(self._df, self._task_id)
        suggestions = _suggest_next_actions(self._df, self._task_id)

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
            suggested_actions=suggestions,
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
            "clip_values": self._cmd_clip_values,
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
        suggestions = _suggest_next_actions(self._df, self._task_id) if (self._df is not None and not done) else []

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
            suggested_actions=suggestions,
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
        # Give a small positive reward for the first inspect (encourages exploration)
        reward = 0.005 if not self._inspected else 0.0
        self._inspected = True

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
        return self._make_obs(msg, reward=reward)

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
        case = params.get("case")
        if not pattern and not case:
            return self._make_obs("Missing 'pattern' or 'case' parameter.", reward=-0.01)

        try:
            mask = self._df[col].notna()
            if case:
                if case == "lower":
                    self._df.loc[mask, col] = self._df.loc[mask, col].astype(str).str.lower()
                elif case == "upper":
                    self._df.loc[mask, col] = self._df.loc[mask, col].astype(str).str.upper()
                elif case == "title":
                    self._df.loc[mask, col] = self._df.loc[mask, col].astype(str).str.title()
                else:
                    return self._make_obs(f"Unknown case transformation '{case}'.", reward=-0.01)
            if pattern:
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

    def _cmd_clip_values(self, col: Optional[str], params: dict) -> DataCleanObservation:
        """Clip numeric values in a column to a valid [min, max] range."""
        if not col or col not in self._df.columns:
            return self._make_obs(f"Column '{col}' not found.", reward=-0.02)

        min_val = params.get("min")
        max_val = params.get("max")
        if min_val is None and max_val is None:
            return self._make_obs("Missing 'min' and/or 'max' parameter.", reward=-0.01)

        try:
            numeric = pd.to_numeric(self._df[col], errors="coerce")
            clipped_count = 0
            if min_val is not None:
                clipped_count += int((numeric < float(min_val)).sum())
                numeric = numeric.clip(lower=float(min_val))
            if max_val is not None:
                clipped_count += int((numeric > float(max_val)).sum())
                numeric = numeric.clip(upper=float(max_val))
            self._df[col] = numeric
        except Exception as e:
            return self._make_obs(f"Error clipping values in '{col}': {e}", reward=-0.02)

        reward = self._quality_delta_reward()
        return self._make_obs(
            f"Clipped {clipped_count} out-of-range values in '{col}' to [{min_val}, {max_val}].",
            reward=reward,
        )

    def _cmd_submit(self, col: Optional[str], params: dict) -> DataCleanObservation:
        quality = _compute_quality_score(self._df, self._target)
        self._done = True

        # Final reward is the quality score itself
        return self._make_obs(
            f"Dataset submitted! Final quality score: {quality:.4f}",
            reward=quality,
            done=True,
        )
