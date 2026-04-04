# Root Dockerfile — Hugging Face Spaces compatible (single-stage)
# Runs the Data Cleaning Environment server on port 8000
# Constraints: 2 vCPU / 8 GB RAM

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no heavy ML libs — stays within 8 GB)
COPY data_clean_env/pyproject.toml /app/data_clean_env/pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openenv-core[core]>=0.2.2" \
        "fastapi>=0.115.0" \
        "pydantic>=2.0.0" \
        "uvicorn[standard]>=0.24.0" \
        "pandas>=2.0.0" \
        "openai>=1.0.0"

# Copy application code
COPY data_clean_env/ /app/data_clean_env/
COPY inference.py /app/inference.py
COPY test_local.py /app/test_local.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "data_clean_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
