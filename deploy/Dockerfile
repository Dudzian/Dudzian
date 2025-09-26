# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/user/Dudzian" \
      org.opencontainers.image.description="KryptoLowca trading bot with monitoring and compliance tooling" \
      org.opencontainers.image.licenses="Proprietary"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KRYPT_LOWCA_CONFIG=/data/config.yaml \
    KRYPT_LOWCA_LOG_FORMAT=json \
    KRYPT_LOWCA_LOG_LEVEL=INFO \
    KRYPT_LOWCA_PROMETHEUS_PORT=8001

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        libffi-dev \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY KryptoLowca /app/KryptoLowca
COPY auto_trader.py ai_manager.py config_manager.py database_manager.py exchange_manager.py risk_management.py trading_gui.py trading_strategies.py event_emitter_adapter.py data_preprocessor.py core /app/

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e ./KryptoLowca \
    && pip install --no-cache-dir "uvicorn[standard]" prometheus-client==0.20.0 python-json-logger==2.0.7 hvac==2.1.0 boto3==1.34.162 \
    && useradd -ms /bin/bash kryptobot

USER kryptobot

VOLUME ["/data", "/app/logs"]

EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -m KryptoLowca.scripts.healthcheck || exit 1

CMD ["python", "-m", "KryptoLowca.bot.run_autotrade_paper"]
