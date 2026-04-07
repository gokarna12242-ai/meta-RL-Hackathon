"""
Data Cleaning Environment - Pydantic Models.

Action, Observation, and State types for the data cleaning environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Action, Observation, State


class DataCleanAction(Action):
    """Action for the data cleaning environment.

    The agent issues cleaning commands to fix data quality issues.
    """

    command: str = ""
    column: Optional[str] = None
    params: Dict[str, Any] = {}


class DataCleanObservation(Observation):
    """Observation returned by the data cleaning environment."""

    dataset_preview: str = ""
    column_info: str = ""
    message: str = ""
    quality_score: float = 0.0
    issues_found: List[str] = []
    task_id: str = ""
    task_description: str = ""
    step_count: int = 0
    max_steps: int = 30
    available_commands: str = ""
    suggested_actions: List[str] = []


class DataCleanState(State):
    """Extended state for the data cleaning environment."""

    task_id: str = ""
    quality_score: float = 0.0
