# Multi-stage Docker build for formal-circuits-gpt

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Development stage
FROM base as development

# Copy requirements and install dependencies
COPY --chown=app:app pyproject.toml ./
RUN pip install --user -e .[dev]

# Copy source code
COPY --chown=app:app . .

# Install in development mode
RUN pip install --user -e .

# Test stage
FROM development as test

# Run tests
CMD ["pytest", "--cov=formal_circuits_gpt", "--cov-report=term-missing"]

# Production stage
FROM base as production

# Copy requirements and install production dependencies only
COPY --chown=app:app pyproject.toml ./
RUN pip install --user .

# Copy source code
COPY --chown=app:app src/ ./src/

# Set PATH to include user pip packages
ENV PATH="/home/app/.local/bin:${PATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD formal-circuits-gpt --version || exit 1

# Default command
CMD ["formal-circuits-gpt", "--help"]