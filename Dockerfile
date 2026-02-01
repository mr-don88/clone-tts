FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TTS with extra dependencies
RUN pip install --no-cache-dir TTS

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p outputs temp static/css static/js templates voices models

# Expose port
EXPOSE 8000

# Pre-download models (optional)
ENV PRELOAD_MODELS=false

# Run the application
CMD ["python", "app.py"]
