# Case studies Stress Lab – komunikacja z klientami OEM

## Case study 1 – Desk prop-tradingowy (EMEA)
- **Cel klienta:** skrócenie czasu reakcji na blackout płynności na parze SOL/USDT.
- **Działania Stage6:** wdrożenie rekomendacji z warsztatu 7 czerwca 2024 r. (progi blackout 40 min, override latency 180 ms).
- **Rezultat:** czas eskalacji incydentu skrócono o 35%, a desk otrzymał automatyczne rekomendacje z modułu override.
- **Dowody:** raport warsztatowy i checklisty audytowe przechowywane w `var/audit/stage6/` wraz z podpisami HMAC.[^audit-workshop]

## Case study 2 – Klient OEM (APAC) z ograniczeniami chmurowymi
- **Cel klienta:** pełna automatyzacja testów ryzyka bez wysyłania danych poza środowisko on-prem.
- **Działania Stage6:** uruchomienie `run_stress_lab.py` w trybie offline, podpis raportów oraz dystrybucja przez marketplace offline.
- **Rezultat:** klient utrzymał zgodność regulacyjną (podpisy HMAC, manifesty) i zintegrował raporty z dashboardem Stage6.
- **Dowody:** runbook paper trading (sekcja Stress Lab) i marketingowy, które opisują proces publikacji oraz weryfikacji podpisów.[^paper-trading-runbook][^marketing-runbook]

## Case study 3 – Release marketplace (globalny rollout)
- **Cel klienta:** pokazanie wartości Stress Lab w katalogu presetów Stage6.
- **Działania Stage6:** powiązanie wyników Stress Lab z katalogiem marketplace (release notes, tagi ryzyka) oraz aktualizacja benchmarku vs CryptoHopper.
- **Rezultat:** wszystkie kampanie marketingowe zawierają odniesienie do Stress Lab, a katalog presetów ma oznaczenie kompatybilności z raportami.
- **Dowody:** benchmark CryptoHopper oraz runbook marketingowy (tabela checklisty) używane jako źródło prawdy.[^benchmark][^marketing-runbook]

## Wskazówki operacyjne
1. Dołączaj do materiałów marketingowych raport JSON + podpis `.sig` oraz manifest `.manifest.json`.
2. W runbooku marketingowym oznacz datę publikacji case study i powiązany release.
3. Synchronizuj benchmark z nowymi wynikami Stress Lab, aby utrzymać narrację przewagi konkurencyjnej.[^benchmark]

[^audit-workshop]: `var/audit/stage6/2024-06-07-stage6-operator-workshop.md` – decyzje dotyczące progów Stress Lab.
[^paper-trading-runbook]: `docs/runbooks/paper_trading.md` – sekcja podpisów HMAC.
[^marketing-runbook]: `docs/runbooks/marketplace_marketing.md` – checklisty marketingowe.
[^benchmark]: `docs/benchmark/cryptohopper_comparison.md` – sekcja przewag Stress Lab.
