# Runbook: Weryfikacja decision logów

## Cel
Zdefiniowanie kroków audytowych dla podpisanych decision logów (demo, paper, live, reinstalacja) z wykorzystaniem narzędzia `scripts/verify_decision_log.py` i rotacji kluczy HMAC.

## Lista kontrolna
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Przygotuj klucz HMAC (`export DECISION_LOG_HMAC_KEY=...`) lub wskaż plik | Security | Klucz w `secrets/decision_log.key` | Klucz zgodny z `risk.decision_log.signing_key_id` |
| 2. Uruchom `python scripts/verify_decision_log.py <plik.jsonl> --hmac-key-file <plik>` | Compliance | Raport na stdout, log audytu | Walidacja podpisów zakończona sukcesem |
| 3. Sprawdź limity liczby zdarzeń (`--require-event-count incident=1`) | Compliance | Raport CLI | Liczba zdarzeń w oczekiwanym przedziale |
| 4. Zweryfikuj snapshot konfiguracji (`--require-tls-materials --require-auth-scope trading`) | Observability | Raport CLI, log TLS | Wszystkie materiały TLS obecne, scope zgodny |
| 5. Wygeneruj raport JSON (`--summary-json audit/decision_log_summary.json`) | Compliance | `audit/decision_log_summary.json` | Plik zawiera metadane, sumy FPS, listę ekranów UI |
| 6. Zarchiwizuj raport i log CLI w `var/audit/` | Operator Runtime | `var/audit/decision_log/` | Raport umieszczony, podpisany HMAC |
| 7. Dodaj wpis `decision_log_verified` do odpowiedniego decision logu | Compliance | Decision log docelowy | Wpis podpisany, zawiera `summary_sha256` |

## Przykładowe scenariusze CLI
- Weryfikacja decision logu live z wymaganiem TLS/mTLS i limitów zdarzeń:
  ```bash
  PYTHONPATH=. python scripts/verify_decision_log.py \
      audit/decision_logs/live_execution.jsonl \
      --hmac-key-file secrets/decision_log.key \
      --require-tls-materials root_cert client_cert client_key server_name \
      --require-auth-scope trading --require-risk-service-tls \
      --summary-json audit/decision_log_live_summary.json \
      --report-output audit/decision_log_live_report.md
  ```
- Podsumowanie logu paper wraz z kontrolą profili ryzyka:
  ```bash
  PYTHONPATH=. python scripts/verify_decision_log.py audit/decision_logs/paper.jsonl \
      --hmac-key-file secrets/decision_log.key \
      --risk-profile balanced --require-risk-service-auth-token \
      --require-risk-service-tls-materials root_cert client_cert
  ```

## Artefakty/Akceptacja
- Raport JSON (`audit/decision_log_*_summary.json`) i opcjonalnie Markdown z CLI.
- Potwierdzone sumy SHA-256 decision logów i raportów, zapisane w `var/audit/hash_registry.json`.
- Wpisy `decision_log_verified` podpisane w decision logach dla demo/paper/live/reinstall.
