# Base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for OpenCV and scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure models and artifacts directories exist (can be mounted in production)
RUN mkdir -p /app/models /app/artifacts/processed_data

# Expose API port
EXPOSE 8000

# Start FastAPI using the provided script
# Note: script accepts args for host/port/models/artifacts paths
CMD ["python", "scripts/04_load_and_predict.py", "--host", "0.0.0.0", "--port", "8000", "--models_path", "/app/models", "--artifacts_path", "/app/artifacts"]