# Multi-stage build for Attrahere ML Code Analysis Platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && chmod +x /app/generate_report.py

USER appuser

# Create volume mount point for code analysis
VOLUME ["/workspace"]

# Set working directory for analysis
WORKDIR /workspace

# Default command runs the CLI analyzer
# Usage: docker run -v $(pwd):/workspace attrahere-image --target your_file.py
ENTRYPOINT ["python", "/app/generate_report.py"]

# Health check for CLI tool (checks if Python dependencies are working)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import analysis_core.ml_analyzer.detectors; print('healthy')" || exit 1