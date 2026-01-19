# Simple Dockerfile for HAR app
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install
COPY deployment/streamlit/requirements.txt .
RUN pip install -r requirements.txt

# Copy app files
COPY deployment/streamlit/app.py .
COPY notebook/models/ ./notebook/models/

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
