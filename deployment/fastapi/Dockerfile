# Dockerfile for FastAPI XGBoost deployment
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (must exist before building)
COPY ../../notebook/models/model.pkl ./models/model.pkl
COPY ../../notebook/models/features.pkl ./models/features.pkl

# Copy FastAPI app
COPY main.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
