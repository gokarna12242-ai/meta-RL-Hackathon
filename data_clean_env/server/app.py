"""
FastAPI application for the Data Cleaning Environment.

Usage:
    uvicorn data_clean_env.server.app:app --reload --host 0.0.0.0 --port 8000
"""

from data_clean_env.models import DataCleanAction, DataCleanObservation
from data_clean_env.server.environment import DataCleanEnvironment
from openenv.core.env_server import create_app

# create_app expects a *class* (factory) so each WebSocket session gets its own
# environment instance.
app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="data_clean_env",
)


def main():
    """Entry point for `uv run server` or `python -m data_clean_env.server.app`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
