"""
Local validation test for the Data Cleaning Environment.

Runs all 3 tasks with a scripted policy (no LLM needed) to verify:
  1. Environment resets correctly
  2. Step executes actions properly
  3. State tracking works
  4. Rewards are calculated correctly
  5. Graders produce scores in 0.0–1.0
  6. Episode terminates on submit or max_steps
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_clean_env.models import DataCleanAction
from data_clean_env.server.environment import DataCleanEnvironment


def test_easy():
    """Scripted policy for the easy task."""
    print("\n[TEST] Easy task")
    env = DataCleanEnvironment(task_id="easy", seed=42)
    obs = env.reset()

    assert not obs.done, "Should not be done after reset"
    assert obs.quality_score >= 0.0
    assert obs.quality_score <= 1.0
    assert obs.step_count == 0
    assert obs.task_id == "easy"
    print(f"  Reset OK. Quality={obs.quality_score:.4f}, Issues={len(obs.issues_found)}")

    # Step 1: Inspect
    obs = env.step(DataCleanAction(command="inspect"))
    assert not obs.done
    print(f"  Inspect OK. Step={obs.step_count}")

    # Step 2: Fix revenue format (remove $ prefix)
    obs = env.step(DataCleanAction(
        command="fix_format",
        column="revenue",
        params={"pattern": r"^\$", "replacement": ""}
    ))
    print(f"  Fix revenue format. Quality={obs.quality_score:.4f}, Reward={obs.reward}")

    # Step 3: Cast revenue to float
    obs = env.step(DataCleanAction(
        command="cast_type",
        column="revenue",
        params={"dtype": "float"}
    ))
    print(f"  Cast revenue. Quality={obs.quality_score:.4f}")

    # Step 4: Cast age to int
    obs = env.step(DataCleanAction(
        command="cast_type",
        column="age",
        params={"dtype": "int"}
    ))
    print(f"  Cast age. Quality={obs.quality_score:.4f}")

    # Step 5: Fill missing age with median
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="age",
        params={"strategy": "median"}
    ))
    print(f"  Fill age. Quality={obs.quality_score:.4f}")

    # Step 6: Fill missing email with a placeholder
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="email",
        params={"strategy": "value", "value": "unknown@example.com"}
    ))
    print(f"  Fill email. Quality={obs.quality_score:.4f}")

    # Step 7: Submit
    obs = env.step(DataCleanAction(command="submit"))
    assert obs.done, "Should be done after submit"
    assert 0.0 <= obs.quality_score <= 1.0
    print(f"  Submit OK. Final quality={obs.quality_score:.4f}")
    return obs.quality_score


def test_medium():
    """Scripted policy for the medium task."""
    print("\n[TEST] Medium task")
    env = DataCleanEnvironment(task_id="medium", seed=42)
    obs = env.reset()
    print(f"  Reset OK. Quality={obs.quality_score:.4f}, Issues={len(obs.issues_found)}")

    # Remove duplicates
    obs = env.step(DataCleanAction(command="remove_duplicates"))
    print(f"  Remove dups. Quality={obs.quality_score:.4f}")

    # Standardize regions
    obs = env.step(DataCleanAction(
        command="standardize",
        column="region",
        params={"mapping": {
            "north": "North", "NORTH": "North", "N.": "North",
            "south": "South", "S.": "South",
            "east": "East",
            "west": "West", "W.": "West",
        }}
    ))
    print(f"  Standardize regions. Quality={obs.quality_score:.4f}")

    # Fill missing unit_price
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="unit_price",
        params={"strategy": "median"}
    ))
    print(f"  Fill unit_price. Quality={obs.quality_score:.4f}")

    # Filter outliers in quantity (use zscore to be more targeted)
    obs = env.step(DataCleanAction(
        command="filter_outliers",
        column="quantity",
        params={"method": "zscore", "threshold": 3.0}
    ))
    print(f"  Filter outliers. Quality={obs.quality_score:.4f}")

    # Submit
    obs = env.step(DataCleanAction(command="submit"))
    assert obs.done
    assert 0.0 <= obs.quality_score <= 1.0
    print(f"  Submit OK. Final quality={obs.quality_score:.4f}")
    return obs.quality_score


def test_hard():
    """Scripted policy for the hard task."""
    print("\n[TEST] Hard task")
    env = DataCleanEnvironment(task_id="hard", seed=42)
    obs = env.reset()
    print(f"  Reset OK. Quality={obs.quality_score:.4f}, Issues={len(obs.issues_found)}")

    # Remove duplicates
    obs = env.step(DataCleanAction(command="remove_duplicates"))
    print(f"  Remove dups. Quality={obs.quality_score:.4f}")

    # Standardize departments
    obs = env.step(DataCleanAction(
        command="standardize",
        column="department",
        params={"mapping": {
            "engineering": "Engineering", "Eng": "Engineering",
            "ENGINEERING": "Engineering", "Engg": "Engineering",
            "marketing": "Marketing", "Mktg": "Marketing", "MARKETING": "Marketing",
            "sales": "Sales", "SALES": "Sales",
            "hr": "HR", "Human Resources": "HR", "H.R.": "HR",
            "finance": "Finance", "FIN": "Finance", "FINANCE": "Finance",
        }}
    ))
    print(f"  Standardize depts. Quality={obs.quality_score:.4f}")

    # Filter salary outliers
    obs = env.step(DataCleanAction(
        command="filter_outliers",
        column="salary",
        params={"method": "iqr", "threshold": 2.0}
    ))
    print(f"  Filter salary outliers. Quality={obs.quality_score:.4f}")

    # Fill missing salary
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="salary",
        params={"strategy": "median"}
    ))
    print(f"  Fill salary. Quality={obs.quality_score:.4f}")

    # Fill missing performance
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="performance_score",
        params={"strategy": "median"}
    ))
    print(f"  Fill perf. Quality={obs.quality_score:.4f}")

    # Fill missing department
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="department",
        params={"strategy": "mode"}
    ))
    print(f"  Fill dept. Quality={obs.quality_score:.4f}")

    # Fill missing phone
    obs = env.step(DataCleanAction(
        command="fill_missing",
        column="phone",
        params={"strategy": "value", "value": "+1-555-0000"}
    ))
    print(f"  Fill phone. Quality={obs.quality_score:.4f}")

    # Submit
    obs = env.step(DataCleanAction(command="submit"))
    assert obs.done
    assert 0.0 <= obs.quality_score <= 1.0
    print(f"  Submit OK. Final quality={obs.quality_score:.4f}")
    return obs.quality_score


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n[TEST] Edge cases")

    env = DataCleanEnvironment(task_id="easy", seed=42)
    obs = env.reset()

    # Unknown command
    obs = env.step(DataCleanAction(command="unknown_cmd"))
    assert obs.reward < 0, "Unknown command should have negative reward"
    print("  Unknown command → negative reward OK")

    # Column not found
    obs = env.step(DataCleanAction(command="fill_missing", column="nonexistent"))
    assert obs.reward < 0
    print("  Column not found → negative reward OK")

    # Fill on column with no nulls
    obs = env.step(DataCleanAction(command="fill_missing", column="name", params={"strategy": "mean"}))
    assert obs.reward <= 0
    print("  Fill on non-null column → non-positive reward OK")

    # Step after done (submit first)
    obs = env.step(DataCleanAction(command="submit"))
    assert obs.done
    obs = env.step(DataCleanAction(command="inspect"))
    assert obs.done, "Should stay done after episode ends"
    print("  Step after done → stays done OK")

    # Test max steps boundary
    env2 = DataCleanEnvironment(task_id="easy", seed=42)
    obs = env2.reset()
    env2._max_steps = 3
    for i in range(4):
        obs = env2.step(DataCleanAction(command="inspect"))
    assert obs.done, "Should be done after exceeding max steps"
    print("  Max steps boundary OK")

    # Forbidden pattern in drop_rows
    env3 = DataCleanEnvironment(task_id="easy", seed=42)
    env3.reset()
    obs = env3.step(DataCleanAction(
        command="drop_rows",
        params={"condition": "import os"}
    ))
    assert obs.reward < 0
    print("  Security: forbidden pattern blocked OK")

    print("\n  All edge cases passed!")


def main():
    print("=" * 60)
    print("  Data Cleaning Environment — Local Validation")
    print("=" * 60)

    scores = {}
    scores["easy"] = test_easy()
    scores["medium"] = test_medium()
    scores["hard"] = test_hard()
    test_edge_cases()

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for task_id, score in scores.items():
        status = "PASS" if 0.0 <= score <= 1.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {task_id:8s}: {score:.4f}  [{status}]")

    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':8s}: {avg:.4f}")

    if all_pass:
        print("\n  ALL TESTS PASSED!")
    else:
        print("\n  SOME TESTS FAILED!")
        sys.exit(1)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
