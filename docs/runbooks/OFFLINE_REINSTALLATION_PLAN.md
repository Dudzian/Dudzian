# Runbook: Reinstalacja i odzyskiwanie w środowisku offline

## Cel
Zapewnienie bezpiecznej reinstalacji pakietu OEM (daemon + UI) na stacjach air-gapped przy zachowaniu ważności licencji, integralności bundla oraz audytu decyzji.

## Lista kontrolna reinstalacji
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Zweryfikuj kopie zapasowe licencji (`var/licenses/backup/*.jsonl`) | Operator OEM | Hash SHA-384, podpis HMAC | Backup aktualny, podpis poprawny |
| 2. Potwierdź ważność fingerprintu urządzenia (`python scripts/oem_provision_license.py --verify <fingerprint.json>`) | Security | Fingerprint, log weryfikacji | Fingerprint zgodny z rejestrem |
| 3. Sprawdź integralność bundla (`shasum -a 384 core-oem-*.tar.gz` + `manifest.sig`) | Release | `manifest.json`, `manifest.sig` | Zgodność sum SHA-384, podpis manifestu | 
| 4. Utwórz punkt przywracania (`rsync -av var/audit/ var/audit_backup/`) | Operator Runtime | Snapshot katalogu audit | Snapshot ukończony, log w decision logu |
| 5. Przeprowadź reinstalację wg `deploy/packaging/README.md` | Operator OEM | Log z instalatora, `install_report.json` | Instalacja zakończona, log bez błędów | 
| 6. Uruchom bootstrap `bootstrap/verify_fingerprint.py --license <licencja.jsonl>` | QA | Log z weryfikacji | Bootstrap PASS, licencja aktywna |
| 7. Zweryfikuj serwisy (`systemctl status bot-core.service`, `systemctl status bot-ui.service`) | DevOps | Logi systemowe, status usług | Status `active (running)` na obu usługach |
| 8. Aktualizuj decision log (`python scripts/verify_decision_log.py summary --append audit/decision_logs/reinstall.jsonl --stage reinstall --status reinstall_complete --artefact bundle_sha384=<SHA-384> --artefact license_id=<ID> --artefact-from-file summary_sha256=var/audit/reinstall/install_report.json --tag reinstall`) | Compliance | `audit/decision_logs/reinstall.jsonl`, `var/audit/reinstall/install_report.json` | Wpis podpisany HMAC, zawiera `bundle_sha384`, `license_id`, skrót raportu i tag `reinstall` |

## Procedura odzyskiwania awaryjnego
1. **Brak ważnej licencji:**
   - Uruchom `python scripts/oem_provision_license.py regenerate --device <fingerprint.json> --registry var/licenses/registry.jsonl --output var/licenses/recovery.jsonl`.
   - Zaktualizuj backup (`cp var/licenses/recovery.jsonl var/licenses/backup/`).
   - Dodaj wpis w decision logu (`status=license_recovered`).
2. **Uszkodzony bundel instalacyjny:**
   - Pobierz podpisany bundel z repozytorium offline (`/mnt/oem/releases`), zweryfikuj SHA-384.
   - Jeżeli podpis niezgodny, zgłoś incydent zgodnie z `docs/runbooks/operations/strategy_incident_playbook.md`.
3. **Niespójność konfiguracji po reinstalacji:**
   - Uruchom `python scripts/validate_config.py --profile live` oraz `python scripts/run_metrics_service.py --shutdown-after 0 --jsonl logs/metrics_post_reinstall.jsonl`.
   - Wykonaj checklisty `docs/runbooks/paper_trading.md` i `docs/runbooks/LIVE_EXECUTION_CHECKLIST.md` przed powrotem do handlu.

## Artefakty/Akceptacja
- `audit/decision_logs/reinstall.jsonl` z wpisami `status=reinstall_complete` oraz `status=license_recovered` (jeśli dotyczy).
- Hash bundla OEM (`core-oem-<wersja>-<platforma>.tar.gz.sha384`) podpisany kluczem HMAC.
- Backup licencji w `var/licenses/backup/` z logiem potwierdzającym datę i operatora.
- Log reinstalacji i raport bootstrapu (`logs/reinstall/bootstrap_verify.log`).
