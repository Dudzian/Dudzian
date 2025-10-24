# Runbook: OEM Installer Acceptance

## Cel
Zapewnienie, że bundel „Core OEM” jest gotowy do dystrybucji offline: artefakty
są podpisane, telemetria i licencjonowanie działają end-to-end, a operator
otrzymuje komplet instrukcji serwisowych.

## Lista kontrolna
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 0. Uruchom `python scripts/run_oem_acceptance.py --artifact-root var/audit/acceptance` lub workflow CI | Release Engineer | `var/audit/acceptance/<TS>/metadata.json`, log CLI | Wszystkie kroki `ok`, brak statusów `failed` |
| 1. Zweryfikuj podpis HMAC `manifest.json` i integralność katalogów `daemon/`, `ui/`, `config/`, `bootstrap/` | Security | `manifest.json`, `manifest.sig`, klucz `OEM_BUNDLE_HMAC_KEY` | Wszystkie sumy SHA-384 zgodne z manifestem i podpis poprawny【F:deploy/packaging/README.md†L1-L44】 |
| 2. Potwierdź podpisy konfiguracji (`config/*.sig`) i fingerprint (`fingerprint.expected.json`) | Security | `config/*.sig`, `fingerprint.expected.json`, klucz HMAC | `payload` i `signature` zgodne; fingerprint odpowiada urządzeniu docelowemu |
| 3. Zweryfikuj pakiet mTLS oraz pinning SHA-256 | Security | `secrets/mtls/*`, log z weryfikacji | Certyfikaty ważne, ścieżki TLS zgodne z dokumentacją UI【F:ui/src/grpc/TradingClient.cpp†L88-L175】 |
| 4. Instalacja próbna bundla (Linux/macOS/Windows) | QA + DevOps | Raport instalacyjny, logi systemowe | Daemon i UI uruchamiają się poprawnie; CLI instalatora raportuje sukces |
| 5. Aktywuj licencję OEM z bundla (`var/licenses/inbox`) | QA | `var/licenses/active/license.json`, log UI | Licencja zapisana, fingerprint zgodny, UI przechodzi w stan aktywny【F:ui/src/license/LicenseActivationController.cpp†L66-L360】 |
| 6. Sprawdź synchronizację risk/AI i eksport CSV | QA | Raport z UI (`var/reports/*.csv`) | Harmonogram risk działa wg konfiguracji, eksport tworzy plik z limitami【F:ui/src/app/Application.cpp†L238-L256】【F:ui/src/app/Application.cpp†L827-L1314】 |
| 7. Zweryfikuj telemetrię oraz RBAC | Observability | Zrzut z MetricsService, log UI | `UiTelemetryReporter` wysyła próbki, brak błędów RBAC/TLS w logach【F:ui/src/telemetry/UiTelemetryReporter.cpp†L200-L272】【F:ui/src/grpc/MetricsClient.cpp†L100-L138】 |
| 8. Wykonaj test scenariusza tradingowego (Qt Quick Test / Playwright) | QA Automation | Raport `ctest`, log e2e | Testy aktywacji licencji i market data przechodzą (`ctest --tests-regex LicenseActivation`) |
| 9. Zweryfikuj panel administratora (RBAC, logi, raporty) | Security + Support | `logs/security_admin.log`, eksport logu wsparcia | Operacje bridge zapisują logi i emitują zdarzenia audytowe【F:ui/src/security/SecurityAdminController.cpp†L31-L175】 |
| 10. Potwierdź możliwość eksportu pakietu wsparcia | Support | Archiwum z logami/raportami | Zawiera `logs/`, eksporty CSV i raport telemetrii, gotowe do wysłania do L2 |
| 11. Archiwizuj artefakty w `var/audit/acceptance/<TS>` oraz decision log | Release Manager | `decision_log/entry.json`, raport PDF | Wpis podpisany, wskazuje lokalizację bundla i logów |

> 💡 **Walidacja przed pakowaniem:** przed uruchomieniem właściwego builda wywołaj `python deploy/packaging/build_core_bundle.py --dry-run` (z tym samym zestawem artefaktów), aby sprawdzić klucze HMAC, ścieżki oraz potencjalne konflikty nazw bez tworzenia archiwów. Flaga `--dry-run` ustawia domyślną wersję `0.0.0-dry-run`, dzięki czemu nie trzeba rezerwować numeru wydania na etapie wstępnej walidacji.

## Artefakty końcowe
- `core-oem-<wersja>-<platforma>.{tar.gz|zip}` wraz z `manifest.json(.sig)` i
  raportem sum SHA-384.
- `var/licenses/active/license.json` oraz podpisane dokumenty provisioning z
  procesu aktywacji.
- Raport z testów (`ctest`, e2e) i logi telemetrii dokumentujące poprawną
  komunikację TLS/RBAC.
- Wpis w decision logu (`var/audit/acceptance/<TS>/decision_log/entry.json`) z
  podpisanym podsumowaniem.
