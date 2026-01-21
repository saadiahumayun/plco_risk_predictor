# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create directories for MLflow and logs
RUN mkdir -p /app/mlruns /app/logs /app/artifacts

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Expose port (Railway uses dynamic port)
EXPOSE 8000

# Run the application (uses $PORT for Railway compatibility)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}