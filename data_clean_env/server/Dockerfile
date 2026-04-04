# Dockerfile for Data Cleaning Environment
# Build: docker build -t data-clean-env:latest -f server/Dockerfile .
# Run:   docker run -d -p 8000:8000 data-clean-env:latest

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the data_clean_env package
COPY . ./data_clean_env/

# Install openenv-core + this environment
RUN pip install --no-cache-dir "openenv-core[core]>=0.2.2" && \
    pip install --no-cache-dir ./data_clean_env/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "data_clean_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
