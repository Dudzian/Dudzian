# Monitorowanie offline runtime

Nowy runtime może uruchomić lokalny eksporter Prometheusa i udostępnić panel monitoringu w aplikacji desktopowej. Poniżej opisano konfigurację oraz sposób korzystania z metryk.

## Konfiguracja `config/runtime.yaml`

Sekcja `observability` w pliku `config/runtime.yaml` kontroluje wszystkie elementy monitoringu:

```yaml
observability:
  enable_log_metrics: true        # włącza integrację logów z metrykami
  prometheus:
    enabled: true                 # uruchamia lokalny serwer metryk
    host: 127.0.0.1               # adres nasłuchu serwera
    port: 9177                    # port HTTP (0 = losowy port systemowy)
    path: /metrics                # ścieżka endpointu Prometheusa
  alerts:
    min_severity: warning         # minimalny poziom alertów logowanych lokalnie
```

Po zmianach można zaktualizować gotowy plik poleceniem:

```bash
python scripts/migrate_runtime_config.py --target config/runtime.yaml
```

## Eksporter Prometheusa

Podczas startu `scripts/run_local_bot.py` konfiguracja `observability.prometheus` uruchamia niewielki serwer HTTP.

* Adres eksportera pojawia się w pliku `--ready-file` oraz w zakładce **Monitoring** w UI.
* Jeśli `port` ustawiono na `0`, system przydzieli wolny port, który zostanie zwrócony w polu `metrics_url` komunikatu `ready`.
* Gromadzone są metryki m.in.:
  * `bot_exchange_requests_total`, `bot_exchange_errors_total`, `bot_exchange_rate_limited_total`, `bot_exchange_health_status`.
  * `bot_strategy_decisions_total`, `bot_strategy_alerts_total`.
  * `bot_security_events_total`, `bot_security_failures_total`.

Eksporter można zatrzymać/zrestartować wraz z całym kontekstem runtime – nie wymaga dodatkowych procesów.

## Integracja logów z metrykami

Handler `bot_core.logging.config.install_metrics_logging_handler()` (automatycznie wywoływany przez `run_local_bot.py`) analizuje rekordy logowania i aktualizuje licznik metryk. Aby dodać własne konteksty, loguj komunikaty z nazwami modułów:

```python
LOGGER.error("API timeout", extra={"latency_ms": 250})
LOGGER.warning("Decision rejected", extra={"decision_latency_seconds": 0.12})
```

Atrybuty `latency_ms`, `latency_seconds`, `decision_latency_seconds`, `security_event`, `exchange` czy `strategy` będą uwzględnione w odpowiednich metrykach.

## Alerty offline

Wszystkie alerty generowane przez moduły bezpieczeństwa oraz serie błędów adapterów (> 4 pod rząd) trafiają do globalnego dispatcher’a i są zapisywane w lokalnych logach (`bot_core.alerts.offline`). Minimalny poziom alertów kontroluje `observability.alerts.min_severity`.

## Zakładka „Monitoring” w UI

W aplikacji desktopowej pojawiła się nowa zakładka **Monitoring**. Panel:

* automatycznie pobiera metryki z eksportera (`/metrics`),
* prezentuje statystyki giełd, strategii i modułów bezpieczeństwa,
* umożliwia ręczne odświeżenie danych oraz wyłączenie auto-odświeżania.

Dzięki temu cała obserwowalność działa w pełni lokalnie – bez zależności od usług chmurowych.
