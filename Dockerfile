# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for LightGBM (libgomp1 is a C++ library required by LGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, and API modules
COPY src/ ./src/
COPY api/ ./api/
COPY main.py .

# Expose FastAPI default port
EXPOSE 8000

# Command to run uvicorn server serving the FastAPI endpoints
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
