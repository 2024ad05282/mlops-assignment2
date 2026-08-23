# ============================================================
# Dockerfile - Cat vs Dog CNN Inference Service
# MLOps Assignment 2 - Model Packaging & Containerization
# ============================================================

# ── Stage 1: Base Image ─────────────────────────────────────
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    MODEL_PATH=models/cat_dog_cnn.pt

# ── Stage 2: Install Dependencies ───────────────────────────
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-inference.txt

# ── Stage 3: Copy Application ───────────────────────────────
# Copy the trained model
COPY models/cat_dog_cnn.pt models/cat_dog_cnn.pt

# Copy the application code
COPY app.py .

# ── Stage 4: Runtime ────────────────────────────────────────
# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the inference server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
