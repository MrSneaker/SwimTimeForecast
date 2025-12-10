FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && rm -rf /root/.cache/pip

# Copy backend code
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
