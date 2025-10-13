# Runbook: OEM Installer Acceptance

## Cel
Zweryfikowanie, że bundel "Core OEM" spełnia wymagania dystrybucji offline (podpisy HMAC, struktura katalogów, checklisty bezpieczeństwa) przed przekazaniem klientowi.

## Lista kontrolna
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Zweryfikuj podpis HMAC `manifest.json` | Security | `manifest.json`, `manifest.sig`, klucz HMAC | `OEM_BUNDLE_HMAC_KEY` potwierdzony, podpis poprawny |
| 2. Sprawdź sumy SHA-384 wszystkich plików | Release Engineer | `manifest.json` | Porównanie sum zakończone bez różnic |
| 3. Potwierdź podpisy plików konfiguracyjnych | Release Engineer | `config/*.sig`, `OEM_BUNDLE_HMAC_KEY` | Dokumenty `.sig` zawierają prawidłowy `payload` oraz podpis `HMAC-SHA384` |
| 4. Uruchom `bootstrap/verify_fingerprint.py` na stacji testowej | QA | Log z konsoli, fingerprint urządzenia | Skrypt kończy się sukcesem, fingerprint zgodny |
| 5. Waliduj strukturę bundla (daemon/ui/config/bootstrap) | QA | Zawartość archiwum | Katalogi kompletne, brak plików tymczasowych |
| 6. Wykonaj instalację próbna (Linux/macOS/Windows) | QA + DevOps | Raport instalacyjny, logi systemowe | Daemon i UI uruchamiają się poprawnie |
| 7. Zarchiwizuj logi i artefakty kontroli | Release Manager | Repozytorium `var/audit`, raport PDF | Wpis w decision log podpisany |

## Artefakty końcowe
- Raport PDF z testów instalatora (załącznik do decision logu).
- Paczka bundla (`core-oem-<wersja>-<platforma>.{tar.gz|zip}`) oraz hash SHA-384.
- Wpis w `var/audit/decision_log.jsonl` potwierdzający akceptację.
