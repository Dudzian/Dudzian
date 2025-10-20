# Stage6 – Checklista przygotowawcza (draft)

## 1. Zakres
Lista kontrolna służy do potwierdzenia gotowości środowiska i zespołu przed rozpoczęciem implementacji Etapu 6. Wersja draft
będzie aktualizowana po zakończeniu discovery i publikacji szczegółowych modułów.

## 2. Kontrola danych
- [ ] Zidentyfikowano wymagane źródła depth-of-book / sentiment i potwierdzono możliwość pozyskania offline.
- [ ] Przygotowano plan rozszerzeń manifestów Parquet/SQLite dla danych Stage6.
- [ ] Zweryfikowano zapotrzebowanie na dodatkowe zasoby storage / compute.

## 3. PortfolioGovernor & Decision Engine
- [ ] Zdefiniowano KPI adaptacyjnego zarządzania kapitałem (np. target allocation drift, max drawdown).
- [ ] Określono interfejs integracji PortfolioGovernora z DecisionOrchestrator i schedulerem multi-strategy.
- [ ] Ustalono procedury override operatora oraz wymagane wpisy decision logu.

## 4. Stress Labs & Resilience
- [ ] Zebrano przypadki testów DR/stres (blackout infrastrukturalny, degradacja giełdy, awaria adaptera).
- [ ] Zaplanowano harmonogram ćwiczeń `failover_drill` i wymagane artefakty audytowe.
- [ ] Przygotowano wymagania SLO2 i progi alertów dla obserwowalności Stage6.
- [ ] Udokumentowano procedurę Stress Lab (`STAGE6_STRESS_LAB_CHECKLIST.md`).
- [ ] Udokumentowano procedurę resilience (`STAGE6_RESILIENCE_DRILL_CHECKLIST.md`).
- [ ] Opracowano proces bundlowania i weryfikacji paczek resilience (`export_resilience_bundle.py`, `verify_resilience_bundle.py`).

## 5. Operacje i compliance
- [ ] Ustalono skład zespołu warsztatowego Stage6 (trading, risk, compliance, operations L1/L2).
- [ ] Zarejestrowano warsztaty discovery w decision logu (CLI `log_stage5_training.py`).
- [ ] Przygotowano repozytorium artefaktów `var/audit/stage6_discovery/` (podpisy HMAC, struktura katalogów).

## 6. Akceptacja wstępna
- [ ] Discovery Stage6 zatwierdzone przez product/compliance.
- [ ] Szkic specyfikacji Stage6 podpisany HMAC i zarchiwizowany.
- [ ] Runbook demo → paper → live zaktualizowany o sekcję Stage6 (migracja presetów i sekretów `preset_editor_cli --core-diff --core-backup --summary-json` z sumami kontrolnymi SHA-256 w podsumowaniu obejmującymi core, backup, magazyn oraz źródła sekretów i opcjonalną sól, wraz z identyfikacją źródeł haseł inline/plik/env oraz kontrolą, że pole `warnings` w `migration_summary.json` jest puste lub odnotowane w decision logu; sekcja `cli_invocation` przechowuje zanonimizowane argumenty CLI z hasłami zastąpionymi `***REDACTED***`, a sekcja `tool` rejestruje interpreter, wersję pakietu i rewizję git migratora i jest archiwizowana w decision logu).
