# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/user/Dudzian" \
      org.opencontainers.image.description="bot_core trading bot with monitoring and compliance tooling" \
      org.opencontainers.image.licenses="Proprietary"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOT_CORE_CONFIG=/data/config.yaml \
    BOT_CORE_LOG_FORMAT=json \
    BOT_CORE_LOG_LEVEL=INFO \
    BOT_CORE_PROMETHEUS_PORT=8001

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        libffi-dev \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e . \
    && useradd -ms /bin/bash kryptobot

USER kryptobot

VOLUME ["/data", "/app/logs"]

EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python scripts/check_runtime_health.py || exit 1

CMD ["python", "-m", "bot_core.auto_trader.paper_app"]
