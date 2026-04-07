"""
DataCleanEnv — client-side wrapper.

Maintains a persistent WebSocket connection to the Data Cleaning
environment server.
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import DataCleanAction, DataCleanObservation, DataCleanState


class DataCleanEnv(EnvClient[DataCleanAction, DataCleanObservation, DataCleanState]):
    """Client for the Data Cleaning environment."""

    def _step_payload(self, action: DataCleanAction) -> dict:
        return {
            "command": action.command,
            "column": action.column,
            "params": action.params,
        }

    def _parse_result(self, payload: dict) -> StepResult[DataCleanObservation]:
        obs = DataCleanObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> DataCleanState:
        return DataCleanState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            quality_score=payload.get("quality_score", 0.0),
        )
