# Benchmark Dudzian vs CryptoHopper

## Źródła referencyjne
- [docs/architecture/stage6_spec.md](../architecture/stage6_spec.md) – zakres autonomicznego portfela, hypercare i wymagania niefunkcjonalne, które definiują docelowy poziom automatyzacji i odporności.
- [README.md](../../README.md) – skrót głównych funkcji produktowych, w tym wsparcie wielu giełd i pipeline AI.
- [docs/runtime/status_review.md](../runtime/status_review.md) – aktualny status warstw runtime/UI oraz identyfikacja luk integracyjnych i automatyzacyjnych.

## Obszary porównawcze
- **Strategia** – adaptacyjne zarządzanie portfelem, marketplace presetów, symulacje stresowe i pipeline TCO/AI.
- **Automatyzacja** – orkiestracja hypercare, cykle resilience/observability oraz poziomy autonomii decyzji tradingowych.
- **UI** – integracja runtime z dashboardem, wizualizacja decyzji AI i wymagane integracje gRPC dla widoczności online.
- **Compliance** – podpisy HMAC, dzienniki decyzji i workflow audytowy w trakcie cykli hypercare.

## Skrót statusu obszarów
- **Strategia:** funkcje core pokrywają scenariusze CryptoHopper, wymaga dopracowania publicznego marketplace’u presetów i komunikacji Stress Labs.
- **Automatyzacja:** Stage6 utrzymuje przewagę dzięki Hypercare Orchestratorowi (podpisy HMAC, resilience/offline) i nowym fallbackom CCXT dla KuCoin/Huobi/Gemini.
- **UI:** feed gRPC spełnia SLO (p95 ≤3 s), telemetria decyzji jest kompletna, a testy PySide6 w CI pilnują regresji; kolejnym krokiem jest utrzymanie monitoringu SLA i alertów HyperCare.
- **Compliance:** przewaga dzięki offline-first journalingowi i podpisom HMAC. Konieczne regularne audyty bundli i aktualizacja materiałów produktowych.

### Tablica wyników Stage6

| Obszar | Status | Kluczowe metryki (cel) | Odpowiedzialny | Najbliższe działania |
| --- | --- | --- | --- | --- |
| Strategia | 🟡 Parzystość z luką marketplace | liczba presetów publicznych (≥15), SLA publikacji (≤48 h), pokrycie Stress Labs w marketingu (100% kampanii) | Zespół Strategii i AI | Udostępnić katalog presetów beta, zsynchronizować komunikację Stress Labs z marketingiem productowym |
| Automatyzacja | 🟢 Przewaga Stage6 | liczba pełnych cykli hypercare/miesiąc (≥4), odsetek podpisanych raportów (100%), średni czas self-healingu (≤5 min) | Zespół Hypercare | Włączyć regresję adapterów do nightly, raportować czasy self-healingu w status_review |
| UI | 🟢 SLO spełnione | opóźnienie feedu gRPC p95 (≤3 s), pokrycie telemetrii decyzji (100%), testy UI gRPC w CI (zielone) | Zespół UI Runtime | Utrzymać monitoring SLA (artefakt `decision-feed-metrics`), zautomatyzować alerty HyperCare i raportować compliance p95 |
| Compliance | 🟢 Przewaga | pokrycie podpisów HMAC (100%), audyty kwartalne bez zastrzeżeń (100%), kompletność TradingDecisionJournal (≥99%) | Zespół Compliance & Audyt | Zestawić wyniki audytu Q2, odświeżyć materiały produktowe i checklisty |

## Stress Lab i materiały marketingowe
- **Whitepaper:** [`docs/marketing/stage6_stress_lab_whitepaper.md`](../marketing/stage6_stress_lab_whitepaper.md) – kompendium przewag technologicznych, integracji z Portfolio Governor oraz wymogów operacyjnych Stress Lab.
- **Case studies:** [`docs/marketing/stage6_stress_lab_case_studies.md`](../marketing/stage6_stress_lab_case_studies.md) – scenariusze wykorzystania Stress Lab u klientów OEM (desk prop, OEM on-prem, rollout marketplace).
- **Artefakty audytowe:** `var/audit/stage6/` – podpisane raporty i manifesty (JSON/CSV/HMAC) stanowią źródło danych dla materiałów marketingowych.
- **Workflow eksportu:** patrz sekcja „Automatyzacja eksportu Stress Lab” w niniejszym dokumencie (poniżej) – pipeline CI publikuje artefakt `stress-lab-report` dla każdego releasu.

## Tabela funkcji i różnic

| Funkcja | Dudzian (Stage6) | CryptoHopper (publiczny plan) | Status różnicy |
| --- | --- | --- | --- |
| Portfolio adaptacyjne / rebalancing | PortfolioGovernor z integracją Stress Lab, override SLO i logiem HMAC. | Automatyczne rebalancingi strategii Pro, oparte o sygnały i copy trading. | Parzystość – kontrolować poziom konfiguracji limitów ryzyka. |
| Poziomy automatyzacji | Stage6 Hypercare Orchestrator łączy cykle portfela, resilience i observability w jednym przebiegu podpisanym HMAC. | Tryby automatyczny/półautomatyczny (strategie, trailing stop, copy bots). | Przewaga Dudzian – utrzymać autonomiczny hypercare offline. |
| Obsługa wielu giełd | Integracje Binance, Coinbase, Kraken, OKX, Bitget, Bybit, KuCoin, Huobi, Gemini **oraz Deribit/BitMEX futures** (paper/testnet/live z podpisanymi checklistami). | Wsparcie >16 giełd, w tym Binance, Coinbase, Kraken, KuCoin, Huobi. | Luka domknięta w segmencie futures – utrzymać regresje adapterów i monitoring HyperCare. |
| Marketplace strategii | Lokalny marketplace presetów i pipeline AI walk-forward. | Globalny marketplace z copy tradingiem, algorytmami społeczności. | Luka – przygotować publiczne listingi presetów i recenzje. |
| Stress Lab i symulacje | Scenariusze multi-market, blackout infrastrukturalny i bundling raportów podpisanych HMAC. | Backtesting i paper trading, brak publicznych stres testów multi-market. | Przewaga Dudzian – komunikować stress labs w marketingu. |
| Resilience / DR | ResilienceHypercareCycle, self-healing runtime, failover drill i bundler artefaktów podpisanych HMAC. | Failover podstawowy (API failover, monitoring uptime). | Przewaga Dudzian – utrzymać przewagę w audycie DR. |
| UI decyzji | Dashboard QML z kartą „Decyzje AI”, wymagająca integracji gRPC dla pełnego feedu runtime. | Webowy UI z dostępem do sygnałów i alertów w czasie rzeczywistym. | Luka – zakończyć integrację gRPC, zapewnić widok live. |
| Compliance i audyt | TradingDecisionJournal, podpisy HMAC dla raportów hypercare oraz logowanie decyzji AI. | Raporty działania bota, brak potwierdzonych podpisów HMAC offline. | Przewaga Dudzian – utrzymać offline-first compliance. |

## Priorytety uzupełniania luk
1. **Utrzymanie pokrycia giełdowego**
   - Cel: utrzymać ≥12 giełd (w tym Deribit/BitMEX futures) z podpisanymi checklistami HyperCare oraz raportem `scripts/list_exchange_adapters.py` w pakiecie benchmarkowym.
   - Metryki: liczba aktywnych adapterów, czas failover (p95), potwierdzenie `live_readiness_signed` w raporcie CSV.
   - Wymagane działania: regresje adapterów nightly, publikacja raportu `reports/exchanges/<data>.csv`, aktualizacja checklist HyperCare.
2. **Marketplace presetów i społeczności**
   - Cel: publiczny katalog presetów z recenzjami i kontrolą wersji offline.
   - Metryki: liczba presetów, liczba aktywnych użytkowników marketplace, czas publikacji nowego presetu.
   - Wymagane działania: rozszerzenie pipeline AI i packaging presetów do dystrybucji OEM.
3. **Alerty SLA feedu UI**
   - Cel: utrzymać zielony status SLO (p95 ≤3 s) i eskalacje degradacji do HyperCare.
   - Metryki: opóźnienie aktualizacji widoku (p95), liczba alertów SLA, czas reakcji operatora.
   - Wymagane działania: zautomatyzować alerty SLA na podstawie `decision-feed-metrics`, rozszerzyć dashboard o p50/p95 cyklu oraz logować eskalacje HyperCare.
4. **Komunikacja przewag compliance**
   - Cel: zachowanie przewagi offline-first (HMAC, journale) w materiałach produktowych.
   - Metryki: pokrycie podpisów HMAC w raportach, liczba audytów zaliczonych bez zastrzeżeń.
   - Wymagane działania: publikacja bundli audytowych w planie release’owym i checklistach wsparcia.

## Harmonogram działań korygujących

| Kwartał | Fokus | Kluczowe kroki | Artefakty kontroli |
| --- | --- | --- | --- |
| Q3 2024 | Integracja UI ↔ runtime | Implementacja kanału gRPC, testy PySide6 w CI, runbook demo dla operatorów L2 | Raport z `docs/runtime/status_review.md`, zaktualizowany benchmark (sekcja UI) |
| Q4 2024 | Marketplace presetów | Publikacja katalogu presetów z recenzjami, rollout procesu wersjonowania offline | Release notes, listing w README, bundler presetów w `var/audit/marketplace/`, workflow `marketplace-catalog` + runbook marketingowy |
| Q1 2025 | Rozszerzenie giełd | Dodanie adapterów do poziomu 15+, testy failover, aktualizacja konfiguracji paper/live (Deribit, BitMEX) | Raport resilience, log failover w `var/audit/` |

## Cadence utrzymania benchmarku
- **Miesięcznie:** aktualizacja tablicy wyników Stage6 na podstawie raportu `docs/runtime/status_review.md` i logów hypercare.
- **Kwartalnie:** porównanie metryk automatyzacji i compliance z danymi audytu; archiwizacja pakietu w `var/audit/benchmark/`.
- **Przed releasem Stage6:** rewizja priorytetów luk i zatwierdzenie checklisty wsparcia.

## Metryki benchmarkowe
- **Poziomy automatyzacji:** ręczny → hypercare półautomatyczny → pełna autonomizacja Stage6 (orchestrator) z podpisami HMAC.
- **Obsługa wielu giełd:** liczba giełd skonfigurowanych out-of-the-box, czasy failover oraz pokrycie trybu paper/live.
- **Ciągłość operacyjna:** dostępność bundli resilience, liczba zamkniętych cykli hypercare z kompletem artefaktów.
- **Compliance:** procent raportów z podpisem HMAC, kompletność TradingDecisionJournal, integracja dzienników z UI i eksportami offline.

## Cykl utrzymania benchmarku
1. **Przegląd release’owy:** przed podpisaniem releasu hypercare wykonaj checklistę z `docs/support/plan.md`, aktualizując statusy funkcji i priorytetów.
2. **Synchronizacja produktowa:** streszcz wyniki benchmarku podczas przeglądu planów produktowych i zaktualizuj linki w `README.md` oraz roadmapach.
3. **Audyt kwartalny:** zestaw wyniki metryk automatyzacji, pokrycia giełdowego i compliance z raportem `docs/runtime/status_review.md`, archiwizując pakiet w `var/audit/benchmark/`.

## Historia aktualizacji benchmarku
| Data | Obszar(y) | Najważniejsze zmiany | Artefakty referencyjne |
| --- | --- | --- | --- |
| 2024-06-30 | UI, Marketplace | Dodano status 🔴 dla feedu gRPC, otwarto działania korygujące Q3. | `docs/runtime/status_review.md`, `reports/ui/grpc_gap_demo.mp4` |
| 2024-07-31 | Automatyzacja | Potwierdzono status 🟢 po regresjach nightly; rozszerzono checklistę wsparcia. | `var/audit/hypercare/2024-07/summary.json`, `docs/support/plan.md` |
| 2024-08-31 | Strategia, Compliance | Publikacja roadmapy presetów i wyników audytu Q2. | `reports/strategy/presets_beta.md`, `var/audit/compliance/2024-Q2.pdf` |
| 2024-09-30 | Integracje giełdowe | Dodano adaptery KuCoin/Huobi/Gemini (paper/testnet/failover) oraz zaktualizowano marketplace i benchmark. | `docs/roadmap/exchange_adapter_rollout.md`, `config/marketplace/presets/exchanges/` |

> Utrzymuj tabelę w formacie kroniki – każdy wpis powinien mieć link do źródłowych artefaktów oraz krótkie streszczenie wpływu na plan domykania luk.

## Procedura zbierania metryk
1. **Zrzut metryk automatyzacji:** uruchom `python scripts/run_stage6_hypercare_cycle.py --export var/audit/hypercare/<data>/summary.json` i zweryfikuj podpis HMAC (`python scripts/verify_stage6_hypercare_summary.py`).
2. **Pokrycie giełd:** wywołaj `python scripts/list_exchange_adapters.py --output reports/exchanges/<data>.csv` i oznacz adaptery w trybie live/paper.
3. **Marketplace presetów:** wygeneruj raport `python scripts/export_preset_catalog.py --format markdown --output reports/strategy/presets_<data>.md` zawierający liczbę presetów publicznych i status recenzji.
4. **Telemetria UI:** z CI pobierz artefakt `decision-feed-metrics` (plik `reports/ci/decision_feed_metrics.json`) generowany przez job „gRPC Decision Feed Integration” i oblicz p50/p95 opóźnienia feedu (`python scripts/calc_ui_feed_latency.py` lub szybka analiza w arkuszu).
5. **Compliance:** zaktualizuj `var/audit/compliance/` o wyniki audytów (`python scripts/export_compliance_report.py`) i zweryfikuj kompletność wpisów `TradingDecisionJournal` (`python scripts/validate_decision_journal.py`).
6. **Aktualizacja benchmarku:** nanieś wartości metryk w tabeli wyników, odśwież priorytety oraz dopisz wpis w sekcji „Historia aktualizacji benchmarku”.

### Automatyzacja eksportu Stress Lab
1. Workflow CI `stress-lab-report.yml` uruchamia `python scripts/run_stress_lab.py run --config config/core.yaml --output reports/stress_lab/ci_report.json --signing-key-env STRESS_LAB_HMAC --fail-on-breach`, a następnie zapisuje podpis `.sig` oraz manifest `.manifest.json` do katalogu artefaktów `stress-lab-report`.
2. Raport i podpis są kopiowane do `var/audit/stage6/ci/` podczas przygotowania release notes.
3. Marketing dodaje link do artefaktu w whitepaperze oraz case studies i aktualizuje sekcję Stress Lab w niniejszym benchmarku.
4. Test `tests/docs/test_marketing_links.py` pilnuje spójności linków do materiałów marketingowych i raportów Stress Lab.

### Walidacja konsystencji
- Zestaw metryki z raportem `docs/runtime/status_review.md` – różnice >5% wymagają otwarcia zadania w Jira/Linear.
- Porównaj liczbę presetów w raporcie marketingowym oraz w katalogu – rozbieżności dokumentuj w kronice.
- Dla statusu 🔴 wymagane jest przypisanie właściciela, terminu i linku do planu działań w harmonogramie korygującym.
