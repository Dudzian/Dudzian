# Phase 2 – OEM: Plan wdrożenia

## Cel iteracji
- Dostarczyć w pełni offline'owy pakiet instalacyjny (daemon + UI Qt/QML) z podpisami kryptograficznymi.
- Uruchomić pipeline paper→live z rozszerzonym silnikiem ryzyka (profile + stres testy) oraz checklistami compliance.
- Zintegrować warstwę live execution z UI i alertami przy zachowaniu wymogów bezpieczeństwa (mTLS, RBAC, audyty).

## Zakres prac
### 1. Packaging OEM
- [x] Opracować format bundla "Core OEM" zawierający:
  - binaria demona (`bot_core`) oraz UI (`ui/` Qt/QML) w wersjach dla Linux/macOS/Windows,
  - prekonfigurowane pliki `.env` i `config/core.yaml` z podpisami HMAC (SHA-384 + klucze rotacyjne),
  - bootstrap skrypt instalacyjny z weryfikacją fingerprintu urządzenia (`deploy/packaging/bootstrap/*`).
- [x] Przygotować pipeline buildów:
  - GitHub Actions self-hosted (air-gapped) z jobami dla platform docelowych (`deploy/ci/github_actions_oem_packaging.yml`),
  - artefakty MSI (Windows), pkg/notarized dmg (macOS), AppImage + .deb (Linux) – publikowane jako artefakty CI (placeholder na binaria produkcyjne).
- [x] Wprowadzić proces podpisywania binariów:
  - Windows: SignTool + certyfikat EV/organizacyjny (konfiguracja w pipeline – krok podpisu do uzupełnienia kluczem produkcyjnym),
  - macOS: `codesign` + plan notarization (placeholder w jobie macOS, integracja z narzędziem Apple Notary w następnym kroku),
  - Linux: GPG + manifest SHA-384 (manifest generowany automatycznie, publikacja podpisu w `manifest.sig`).
- [x] Udokumentować procedurę "Installer Acceptance" (lista kontrolna artefakty/akceptacja) w `docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md`.
- [x] Zautomatyzować dry-run akceptacyjny (`scripts/run_oem_acceptance.py`) generujący raport JSON z wynikami bundla/licencji/Paper Labs/mTLS.

### 2. Fingerprint i licencjonowanie offline
- [x] Zaimplementować moduł `bot_core/security/fingerprint.py` generujący fingerprint (CPU, TPM, MAC, opcjonalny dongle) z podpisem HMAC.
- [x] Zaprojektować rejestr licencji (`var/licenses/registry.jsonl`) z podpisami oraz narzędziem `scripts/oem_provision_license.py`.
- [x] Stworzyć runbook provisioning (checklista + artefakty: request, approval, signed license).
- [x] Uzupełnić UI o ekran aktywacji licencji z obsługą QR/USB.

### 3. Rozszerzenie silnika ryzyka
- [x] Dodać moduł symulacji scenariuszy (`bot_core/risk/simulation.py`) z obsługą profili: konserwatywny, zbalansowany, agresywny, manualny.
- [x] Przygotować stres testy (flash crash, dry liquidity, latency spike) bazujące na danych Parquet.
- [x] Zintegrować raporty PDF/JSON z pipeline'em gatingowym (`deploy/ci/github_actions_paper_smoke.yml`).
- [x] Przygotować runbook "Paper Labs" (lista kontrolna: dane wejściowe, wyniki symulacji, akceptacja compliance).

### 4. Live execution + UI + Alerting
- [x] Uzupełnić `bot_core/execution/live_router.py` o realne routowanie i fallbacki giełdowe.
- [x] Dodać monitoring latencji/fill rate (metryki Prometheus, alerty w `config/core.yaml`).
- [x] Rozszerzyć gRPC o mTLS + RBAC tokeny, generatory certyfikatów (`scripts/generate_mtls_bundle.py`).
- [x] Zaktualizować UI (gRPC client) o obsługę TLS, telemetry overlay, checklistę przed startem live.

### 5. Dokumentacja i runbooki
- [x] Checklisty demo→paper→live w formacie: Krok, Odpowiedzialny, Artefakty, Akceptacja.
- [x] Plan reinstalacji w środowisku offline (backup licencji, recovery bundli, walidacja hashy).
- [x] Decision log JSONL z podpisami (narzędzie `scripts/verify_decision_log.py`).

## Wymagania bezpieczeństwa
- mTLS (TLS 1.3, certyfikaty rotowane co 90 dni, CRL offline).
- RBAC oparty na tokenach z podpisem HMAC (audyt JSONL).
- Brak WebSocketów – wyłącznie gRPC/HTTP2 lub IPC.
- Guard UI: FPS monitor, reduce motion, p95 <150 ms, jank <1%.

## Harmonogram (propozycja)
| Sprint | Zakres | Milestone |
| --- | --- | --- |
| 1 | Packaging OEM + podpisy | Artefakty instalatora + checklisty |
| 2 | Fingerprint/licencje + provisioning | Pierwszy klient OEM offline |
| 3 | Symulacje/stres testy + gating | Raport "Paper Labs" |
| 4 | Live execution + UI TLS | Tryb live pilotowy |

## Ryzyka i mitigacje
| Ryzyko | Mitigacja |
| --- | --- |
| Brak certyfikatów EV/Apple | Zamówienie certyfikatów równolegle, fallback self-signed + whitelist lokalna |
| Niespójne fingerprinty HW | Warstwa abstrakcji + fallback dongle/OTP |
| Opóźnienia w integracji UI | Równoległy prototyp CLI + testy gRPC contract |
| Brak danych do stres testów | Generatory syntetyczne + import publicznych datasetów |

## Artefakty wyjściowe
- Instalatory OEM (MSI, pkg/dmg, AppImage/.deb) z podpisami.
- Repozytorium licencji offline + narzędzia provisioning.
- Raporty symulacji/stres testów (PDF/JSON) + checklisty compliance.
- Zaktualizowana dokumentacja runbooków oraz UI z TLS/RBAC.

