# LCF — always-on scheduler container.
# Runs `python lcf.py schedule`, which drives all 5 flows on cadence.
# Secrets are provided at runtime via environment variables (see README/Azure
# Container App settings) — NOT baked into the image.

FROM python:3.11-slim

# Stream logs immediately (so `az containerapp logs` shows output live).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (see .dockerignore for what's excluded).
COPY . .

# The always-on process. Container Apps keeps this running (min replicas = 1).
CMD ["python", "lcf.py", "schedule"]
