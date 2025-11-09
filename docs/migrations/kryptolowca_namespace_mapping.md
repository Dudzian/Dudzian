# Mapowanie modułów KryptoLowca → bot_core

Aby ułatwić migrację istniejących integracji oraz skryptów narzędziowych,
zebraliśmy mapowanie najczęściej używanych modułów z legacy pakietu
`KryptoLowca` na nowe lokalizacje w przestrzeni `bot_core`.  Wszystkie
nowe uruchomienia powinny importować z `bot_core` – tabela ma pomóc
utrzymać kompatybilność w zespołach, które nadal posiadają lokalne
skrypty lub środowiska testowe zależne od dawnych ścieżek.

| KryptoLowca | bot_core | Notatki |
| --- | --- | --- |
| `KryptoLowca.logging_utils.setup_logging` | `bot_core.logging.app.setup_app_logging` | Nowa wersja wspiera zmienne środowiskowe `BOT_CORE_*` z fallbackiem dla prefiksów legacy. |
| `KryptoLowca.logging_utils.get_logger` | `bot_core.logging.app.get_logger` | Loggery są konfigurowane kolejkującą warstwą, gotową na Vectora/Prometheusa. |
| `KryptoLowca.runtime.bootstrap.bootstrap_frontend_services` | `bot_core.runtime.frontend.bootstrap_frontend_services` | Funkcja zwraca `FrontendBootstrap` z managerem giełd, routerem egzekucji i modułem market intel. |
| `KryptoLowca.ui.trading.risk_helpers` | `bot_core.ui.trading.risk_helpers` | Zestaw helperów GUI wykorzystuje `RiskSnapshot` i obsługuje brak modułu GUI w instalacjach headless. |
| `KryptoLowca.auto_trader.paper_app` | `bot_core.auto_trader.paper_app` | Launcher papierowy ma wbudowaną degradację, gdy pakiet GUI jest niedostępny. |
| `KryptoLowca.services.alerting` | `bot_core.services.alerting` | Warstwa alertów obsługuje `AlertSink` dla email/webhooków i integruje się z `AlertRouter`. |
| `KryptoLowca.services.risk_dashboard` | `bot_core.services.risk_dashboard` | Serwis FastAPI wystawia health-check i endpoints `/risk/audits`. |

Jeśli brakuje tu modułu, który był używany w starszym środowisku,
zgłoś to w wątku `#migration-support` – sukcesywnie będziemy uzupełniać
tabelę, aby uniknąć powrotu do usuniętego pakietu `KryptoLowca`.

> **Status archiwum:** Katalog `archive/legacy_bot` został trwale usunięty.
> Repozytorium nie zawiera już shimów przekierowujących do `bot_core`, dlatego
> wszelkie integracje muszą importować nowe moduły bezpośrednio z przestrzeni
> `bot_core.*`.
