# Przewodnik wdrożeniowy (Docker + monitoring)

## Wymagania

- Docker / Docker Compose v2
- Dostęp do konta demo giełdy (np. Binance Testnet)
- Konto Slack/email dla alertów (opcjonalnie)

## Budowanie obrazu

```bash
cd deploy
docker compose build
```

Obraz bazuje na `python:3.11-slim`, instalując zależności runtime oraz narzędzia monitoringu (`prometheus-client`, `uvicorn`).

## Uruchomienie środowiska

```bash
docker compose up -d
```

Uruchomione zostaną następujące usługi:

| Serwis | Port | Opis |
| --- | --- | --- |
| `bot` | 8000 / 8001 | AutoTrader + endpoint `/metrics` |
| `prometheus` | 9090 | Zbieranie metryk |
| `grafana` | 3000 | Dashboard (login/pwd: `admin`/`admin`) |
| `loki` | 3100 | Przechowywanie logów |
| `vector` | 8686 | Ingest logów z wolumenu `./logs` |

Wolumeny:
- `./data` – konfiguracja, bazy danych, klucze
- `./logs` – logi JSON, streamowane do Vector/Loki

## Konfiguracja środowiska

1. Skopiuj przykładową konfigurację: `cp KryptoLowca/config.yaml data/config.yaml` i dostosuj klucze API (zawsze konto demo!).
2. Ustaw zmienne środowiskowe w `deploy/docker-compose.yml` (np. `BOT_ENV=demo`, `KRYPT_LOWCA_PROMETHEUS_PORT=8001`).
3. Opcjonalnie ustaw `KRYPT_LOWCA_LOG_SHIP_VECTOR=http://vector:8686` jeśli chcesz wysyłać logi także przez HTTP.

## Monitoring

- Metryki Prometheus dostępne są pod `http://localhost:8001/metrics`.
- Grafana: panel `KryptoLowca – Overview` pokazuje liczbę zleceń, stan guardrails i logi centralne.
- Logi można przeglądać w Grafanie (datasource Loki) lub przez API Vectora (`http://localhost:8686/playground`).

## Rotacja kluczy i sekrety

- W kontenerze skrypt `python -m KryptoLowca.scripts.healthcheck` sprawdza podstawową gotowość.
- Do rotacji kluczy użyj `bot_core.security.RotationRegistry` wraz z `SecretManager`em (np. zadanie cron raz na 30 dni aktualizuje wpis poprzez `mark_rotated`).
- Sekrety przechowuj w wspieranym magazynie (`bot_core.security.SecretStorage`): plik wolumenu (`.secrets.json`), HashiCorp Vault, AWS Secrets Manager itp.

## Testy bezpieczeństwa

Po wdrożeniu wykonaj:

1. `pytest KryptoLowca/tests/test_security_rotation.py`
2. Chaos test (restart kontenera `bot`): `docker compose restart bot`
3. Symulację obciążenia (skrypt `scripts/load_test_signals.py` – w przygotowaniu)

Pamiętaj, by wszelkie zmiany testować na koncie demo zanim przełączysz tryb LIVE.
