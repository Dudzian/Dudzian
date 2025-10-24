# Changelog runbooków operacyjnych

## 2025-10-23 – Aktualizacja komendy Paper Labs
- **Zakres**: `docs/runbooks/PAPER_LABS_CHECKLIST.md`
- **Zmiana**: doprecyzowano wywołanie `python scripts/run_risk_simulation_lab.py` wraz z obowiązkowymi flagami `--config config/core.yaml` i `--output-dir reports/paper_labs`.
- **Działanie dla zespołu Ryzyka**: od kolejnego cyklu Paper Labs używajcie nowej komendy, aby uniknąć uruchomień bez jawnie wskazanych ścieżek konfiguracyjnych i katalogu artefaktów.
## 2025-10-24 – Ujednolicenie komend CLI w runbookach
- **Zakres**: runbooki Stage4–Stage6, backfill, Paper Trading, OEM provisioning, decision log verification oraz checklisty hypercare/portfolio/resilience.
- **Zmiana**: doprecyzowano komendy `python scripts/...` (oraz warianty `PYTHONPATH=. python ...`) w instrukcjach operacyjnych, aby jednoznacznie wskazywały interpreter i wymagane flagi.
- **Działanie dla operatorów**: korzystajcie z ujednoliconych komend przy kolejnych procedurach (paper/demo/live), żeby uniknąć niejasności co do sposobu uruchamiania narzędzi.
- **Notatka**: rozszerzono dokumentację architektoniczną, warsztatową i operacyjną o identyczne prefiksy `python`, dzięki czemu opisy narzędzi w całym repozytorium wskazują pełne polecenia.
