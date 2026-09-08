# Dockerfile — Fraud Detection Data Pipeline (Module 3)
# Containerizes the ETL pipeline so it runs identically in any environment,
# supporting the reproducibility requirement in Section 5.

FROM python:3.11-slim

WORKDIR /app

# System dependencies for pandas/numpy compiled extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching: this layer only
# rebuilds when requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY src/ ./src/
COPY dags/ ./dags/
COPY data/raw/ ./data/raw/

# Create output directory for processed data and audit logs
RUN mkdir -p ./data/processed ./docs

# Default command: run the full orchestrated pipeline
CMD ["python", "dags/fraud_pipeline_flow.py"]
