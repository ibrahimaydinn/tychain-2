FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (curl for healthchecks; libsql wheels need libgcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Setup permissions for Hugging Face Spaces
# Hugging Face Spaces runs containers with user ID 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /app/cache && \
    chown -R user:user /app && \
    chmod -R 777 /app/cache

USER user

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Production env defaults — actual secrets come from HF Spaces "Variables and secrets"
# (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN, SMTP_HOST/PORT/USER/PASS, MAIL_FROM, APP_URL).
ENV PORT=7860 \
    PYTHONPATH=/app \
    APP_URL=https://huggingface.co/spaces

# Start the Flask app using Gunicorn.
# Single worker on free HF CPU: avoids cross-worker session-cookie mismatch
# (SECRET_KEY MUST be set as a Space secret regardless), and the workload
# is dominated by long-running subprocess analyses, not concurrent requests.
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "360", "--workers", "1", "--threads", "4", "app:app"]
