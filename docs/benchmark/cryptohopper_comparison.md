# Benchmark Dudzian vs CryptoHopper & Gunbot

## Źródła referencyjne
- [docs/architecture/stage6_spec.md](../architecture/stage6_spec.md) – zakres autonomicznego portfela, hypercare i wymagania niefunkcjonalne, które definiują docelowy poziom automatyzacji i odporności.
- [README.md](../../README.md) – skrót głównych funkcji produktowych, w tym wsparcie wielu giełd i pipeline AI.
- [docs/runtime/status_review.md](../runtime/status_review.md) – aktualny status warstw runtime/UI oraz identyfikacja luk integracyjnych i automatyzacyjnych.
- [config/marketplace/catalog.md](../../config/marketplace/catalog.md) – podpisany katalog (JSON/Markdown) z ≥15 strategiami i personami wykorzystywany w porównaniach rynkowych.

## Obszary porównawcze
- **Strategia** – adaptacyjne zarządzanie portfelem, marketplace presetów, symulacje stresowe i pipeline TCO/AI (kontrastowane z CryptoHopperem i Gunbotem).
- **Automatyzacja** – orkiestracja hypercare, cykle resilience/observability oraz poziomy autonomii decyzji tradingowych.
- **UI** – integracja runtime z dashboardem, wizualizacja decyzji AI i wymagane integracje gRPC dla widoczności online.
- **Compliance** – podpisy HMAC, dzienniki decyzji i workflow audytowy w trakcie cykli hypercare.

## Skrót statusu obszarów
- **Strategia:** funkcje core pokrywają scenariusze CryptoHoppera i Gunbota, a `PresetPublicationWorkflow` + testy `tests/test_marketplace_workflow.py` zapewniają podpisane marketplace’y i payload do kreatora PySide6; kontynuować komunikację Stress Labs oraz publikację `config/marketplace/catalog.md`.
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

## Stress Lab

| Giełda | Data snapshotu (UTC) | Liczba zleceń | Fill ratio | Średni slippage (bps) | Failures | Alerty watchdog | Notatki |
| --- | --- | --- | --- | --- | --- | --- | --- |
| testex | 2025-11-10 17:41:43 | 2 | 1.00 | 0.0 | 0 | 0 | Tryb margin, opóźnienia <0,2 ms; komplet podpisanych rekordów. |

- Dane bazowe pochodzą z raportów `reports/exchanges/signal_quality/*.json`; snapshot `reports/exchanges/signal_quality/testex.json` (`total=2`, `fill_ratio=1.0`, `slippage_bps=0.0`, `watchdog.alerts=[]`) potwierdza brak slippage i brak alertów watchdog, co spełnia wewnętrzne SLO Stress Lab.
- **Porównanie konkurencyjne:**
  - *CryptoHopper:* brak publikowanych, podpisanych raportów Stress Lab oraz brak jawnych metryk `fill_ratio`/`slippage_bps` w release notes – utrzymujemy przewagę transparentności i audytu.
  - *Gunbot:* oferuje głównie backtesty per para bez multi-market watchdogów; brak raportów HMAC/manifestów = brak dowodu na niezawodność w scenariuszach DR. Nasze snapshoty Stress Lab pozostają wyróżnikiem do komunikacji marketingowej.

## Stress Lab i materiały marketingowe
- **Whitepaper:** [`docs/marketing/stage6_stress_lab_whitepaper.md`](../marketing/stage6_stress_lab_whitepaper.md) – kompendium przewag technologicznych, integracji z Portfolio Governor oraz wymogów operacyjnych Stress Lab.
- **Case studies:** [`docs/marketing/stage6_stress_lab_case_studies.md`](../marketing/stage6_stress_lab_case_studies.md) – scenariusze wykorzystania Stress Lab u klientów OEM (desk prop, OEM on-prem, rollout marketplace).
- **Artefakty audytowe:** `var/audit/stage6/` – podpisane raporty i manifesty (JSON/CSV/HMAC) stanowią źródło danych dla materiałów marketingowych.
- **Workflow eksportu:** patrz sekcja „Automatyzacja eksportu Stress Lab” w niniejszym dokumencie (poniżej) – pipeline CI publikuje artefakt `stress-lab-report` dla każdego releasu.
- **Katalog presetów:** [`reports/strategy/presets_2025-01-15.md`](../../reports/strategy/presets_2025-01-15.md) – lista podpisanych presetów (≥15) wraz z metadanymi review i linkami do artefaktów QA.
- **Katalog Marketplace (Markdown):** [`config/marketplace/catalog.md`](../../config/marketplace/catalog.md) – podpisany listing z personami wykorzystywany w komunikacji marketingowej oraz porównaniach z CryptoHopperem i Gunbotem.
- **Rollout presetów:** proces publikacji podpisanych presetów i katalogu opisany w [`docs/marketplace/README.md`](../marketplace/README.md) (sekcja „Rollout presetów”) – wymagany przed aktualizacją benchmarku.

### Publikacja materiałów marketingowych
- **Pakietowanie Stress Lab:** najnowsze raporty z `reports/stress_lab/` (JSON + `.sig` + `.manifest.json`) są kopiowane do `var/marketing/benchmark/stress_lab/` oraz linkowane w manifeście release (artefakt CI `stress-lab-report`).
- **Checklisty adapterów:** dzienny snapshot checklist HyperCare i jakości sygnałów z `reports/exchanges/signal_quality/` jest dołączany do pakietu marketingowego (CSV/JSON) jako dowód pokrycia giełdowego i gotowości failoveru.
- **Artefakty release:** manifest bundla marketingowego zawiera linki do artefaktów CI (`stress-lab-report`, `benchmark-marketing-bundle`) i katalogu `var/marketing/benchmark/`, aby ułatwić aktualizację whitepapera/case studies.

### Agregat stres-testów i checklist adapterów
- **Cel:** jedna paczka „signal-quality + checklisty” dla marketingu i release notes, oparta o świeże stres-testy z `reports/exchanges/signal_quality/` i CSV z `reports/exchanges/<data>.csv` (generowane przez `scripts/list_exchange_adapters.py`).
- **Zakres danych:**
  - ostatnie 7 dni snapshotów JSON dla każdej giełdy (`reports/exchanges/signal_quality/*.json`) z polami `fill_ratio`, `slippage_bps`, `failures`, `watchdog.alerts`,
  - najnowszy CSV checklisty adapterów (kolumny `hypercare_checklist_signed`, `signal_quality_snapshot_status`, `futures_checklist_ready`, `hypercare_cost_status`).
- **Przebieg aktualizacji:**
  1. Wygeneruj checklistę: `python scripts/list_exchange_adapters.py --report-date $(date +%Y-%m-%d) --report-dir reports/exchanges --push-dashboard --dashboard-dir reports/exchanges/signal_quality --hypercare-config config/stage6/hypercare.yaml`.
  2. Zweryfikuj świeżość stres-testów: `find reports/exchanges/signal_quality -name "*.json" -mtime -2 -print` powinien zwrócić pliki dla kluczowych giełd (`binance`, `coinbase`, `deribit_futures`, `bitmex_futures`). Brak wyników = blokada publikacji.
  3. Zbuduj paczkę marketingową z agregatem: `python scripts/export_marketing_bundle.py --report-range $(date +%Y-%m-%d) --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC --include-signal-quality`.
  4. Sprawdź manifest bundla (`var/marketing/benchmark/manifest.json`) – musi zawierać sekcje `stress_lab`, `signal_quality` (JSON + CSV) oraz podpis HMAC (`benchmark_marketing_bundle.sig`).
  5. Upewnij się, że agregat `var/marketing/benchmark/signal_quality/index.csv` zawiera wiersze dla wszystkich giełd z `reports/exchanges/<data>.csv` i że timestampy w kolumnie `snapshot_created_at` są nie starsze niż 48 h; w przypadku braków ponów eksport po odświeżeniu snapshotów.
  6. Zsynchronizuj pakiet z repozytorium marketingowym (jeśli istnieje lustrzany bucket S3/Git) i oznacz wersję w release notes w formacie `benchmark_marketing_bundle-<data>.sig`.
- **Artefakty porównawcze:**
  - `var/marketing/benchmark/signal_quality/index.csv` – agregat ostatnich stres-testów i checklist adapterów wykorzystywany w whitepaperze,
  - `reports/exchanges/signal_quality/` – źródło prawdy dla historycznych snapshotów (dashboard Hypercare),
  - `reports/exchanges/<data>.csv` – CSV checklisty adapterów włączane do release notes jako dowód pokrycia giełdowego.
- **Walidacja spójności bundla:**
  - porównaj liczbę wierszy w `signal_quality/index.csv` z liczbą snapshotów `.json` w katalogu dashboardu (`find reports/exchanges/signal_quality -name "*.json" | wc -l`) – różnice oznaczają brakujące rekordy w agregacie,
  - sprawdź, że `manifest.json` zawiera identyczne ścieżki jak release notes (sekcja „Release artifacts”) oraz że hash pliku `benchmark_marketing_bundle.sig` jest przeklejony do zgłoszenia marketingowego,
  - jeśli marketing korzysta z lustrzanego bucketa S3/Git, potwierdź, że commit/tag bundla zawiera ten sam `index.csv` (porównanie sumy SHA256) co wersja archiwalna w `var/audit/benchmark/<data>/` – w razie rozbieżności powtórz eksport i walidację.
  - jeżeli bundel w lustrze wymaga rekordu w rejestrze marketingowym, dopisz identyfikator releasu (`benchmark_marketing_bundle-<data>.sig`) oraz status walidacji hashy; brak wpisu blokuje dystrybucję do kanałów zewnętrznych,
  - gdy pipeline CI wykryje rozjazd hashy między lustrami, na czas naprawy włącz tryb „freeze” (oznaczenie releasu jako oczekującego w release notes, blokada publikacji materiałów) i wyłącz dopływ nowych snapshotów do bundla, aby zachować odtwarzalność pakietu,
  - po usunięciu rozbieżności opublikuj notatkę audytową z zestawieniem hashy „przed/po” oraz wersją bundla, która została uznana za źródło prawdy.
  - przed każdym podpisaniem bundla uruchom szybki diff na `manifest.json` względem poprzedniego releasu (`python scripts/export_marketing_bundle.py --destination var/marketing/benchmark --diff-only`), aby potwierdzić, że ścieżki w sekcji `signal_quality` pokrywają się z listą snapshotów w `reports/exchanges/signal_quality/`;
  - po publikacji bundla porównaj listę giełd w `signal_quality/index.csv` z checklistą HyperCare (`reports/exchanges/<data>.csv`) i zapisz wynik w `var/audit/benchmark/<data>/parity_report.json` (różnice = blokada dystrybucji do kanałów zewnętrznych),
  - gdy marketing zgłasza użycie bundla w materiałach zewnętrznych, sporządź zapis „proof-of-source” (hash `manifest.json`, timestamp walidacji HMAC, ścieżka do archiwum) w dzienniku marketingowym – brak wpisu uniemożliwia referencję w whitepaper/case studies,
  - jeśli nowy bundel zmienia liczbę wierszy `index.csv` o >10% vs. poprzedni release, oznacz release notes notatką o zakresie zmian i dołącz krótką tabelę różnic (`added_exchanges`, `dropped_exchanges`) do `var/audit/benchmark/<data>/delta.csv`.
- **Kontrola jakości:**
  - jeżeli `signal_quality_snapshot_status != "fresh"` lub `hypercare_cost_status != "ready"` dla `deribit`/`bitmex`, otwórz zadanie w HyperCare i wstrzymaj publikację benchmarku,
  - marketing otrzymuje tylko paczki z ważnym podpisem HMAC; w razie rozbieżności uruchom `--validate-only` w bundlerze i powtórz eksport po korektach.
  - dla materiałów zewnętrznych (whitepaper/newsletter) wygeneruj notatkę `marketing_bundle_proof.md` z listą giełd, hashami `index.csv`/`manifest.json`, timestampem walidacji HMAC i adresem lustra S3/Git; dołącz ją do zgłoszenia marketingowego i release notes jako „proof-of-source”,
  - przy zmianie pokrycia (<-10% lub >+10% liczby giełd vs. poprzedni release) dołącz tabelę różnic do `var/audit/benchmark/<data>/delta.csv`, oznacz release notes statusem „ważna zmiana pokrycia” i wyślij alert do właścicieli Strategia/HyperCare.

#### Monitoring po publikacji i retrospekcje
- **Alerty świeżości:** skonfiguruj job w CI (`cron` lub `workflow_dispatch`) sprawdzający co 12 h, czy `reports/exchanges/signal_quality/*.json` mają `mtime < 48 h`; w przypadku naruszenia wyślij alert na kanał operacyjny i dodaj notatkę do `docs/audit/paper_trading_log.md`.
- **Parzystość z checklistą:** po publikacji uruchom szybki diff CSV → `python - <<'PY'\nimport csv, pathlib, sys\nidx = pathlib.Path('var/marketing/benchmark/signal_quality/index.csv')\nchecklist = sorted(row['exchange'] for row in csv.DictReader(open('reports/exchanges/$(date +%Y-%m-%d).csv')))\nindex = sorted(row['exchange'] for row in csv.DictReader(idx.open()))\nmissing = sorted(set(checklist) - set(index))\nextra = sorted(set(index) - set(checklist))\nprint({'missing_in_index': missing, 'extra_in_index': extra})\nif missing or extra:\n    sys.exit('signal_quality/index.csv wymaga regeneracji')\nPY` – wynik z brakami oznacza konieczność regeneracji bundla marketingowego.
- **Retrospekcja releasu:** w ciągu 48 h od releasu porównaj `var/audit/benchmark/<data>/manifest.json` z `var/marketing/benchmark/manifest.json` i release notes; brakujące lub dodatkowe artefakty dopisz do dziennika audytowego wraz z hashami SHA256.
- **Parzystość międzylustrzana:** jeżeli marketing przechowuje bundel w lustrze S3/Git, uruchom diff hashy `index.csv` oraz `manifest.json` (`sha256sum <plik>`) między lokalnym `var/marketing/benchmark/` a lustrem. Rozbieżności oznaczają blokadę publikacji lub konieczność wymiany linków w release notes.

#### Reakcja na brakujące lub niespójne snapshoty
- **Brakujące rekordy w agregacie:** jeśli `index.csv` zawiera mniej wierszy niż raport CSV checklisty, wygeneruj brakujące stres-testy (`scripts/run_stress_lab.py run --exchanges <lista>`) i powtórz kroki 1–5, aż liczba wierszy i plików `.json` będzie zgodna.
- **Stare snapshoty (>48 h):** ponownie uruchom `scripts/list_exchange_adapters.py` z `--report-date $(date +%Y-%m-%d)` i dopisz wynik do `reports/exchanges/signal_quality/`, a następnie zaktualizuj `index.csv` bundlerem marketingowym.
- **Niespójny manifest:** jeśli `manifest.json` nie zawiera sekcji `signal_quality` lub brakuje ścieżek podlinkowanych w release notes, uruchom bundler z `--include-signal-quality --force-rebuild` i zweryfikuj sumę SHA256 `benchmark_marketing_bundle.sig` względem archiwum `var/audit/benchmark/<data>/`.
- **Fallback na ostatni poprawny zestaw:** gdy bieżący cykl nie przechodzi walidacji HMAC lub liczby wierszy, użyj ostatniej paczki z `var/audit/benchmark/<poprzednia_data>/` (kopiuj cały katalog do `var/marketing/benchmark/`), a w release notes zaznacz, że użyto snapshotu archiwalnego.

#### Procedura awaryjna bundla marketingowego
- **Szybkie przywrócenie:** jeśli walidacja HMAC lub hashów lustra zawodzi, skopiuj w 1:1 ostatni zweryfikowany katalog z `var/audit/benchmark/<poprzednia_data>/` do `var/marketing/benchmark/`, podmień `benchmark_marketing_bundle.sig` na wersję archiwalną i oznacz wydanie jako rollback w `docs/audit/paper_trading_log.md`.
- **Przebudowa z wymuszeniem:** ponów eksport `python scripts/export_marketing_bundle.py --report-range $(date +%Y-%m-%d) --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC --include-signal-quality --force-rebuild --regenerate-index` i porównaj sumy SHA256 z plikami z lustrzanej kopii; dopiero po zgodności hashów i walidacji HMAC opublikuj nową wersję.
- **Deklaracja statusu:** w release notes i zgłoszeniu marketingowym umieść sekcję „Bundel marketingowy – status awaryjny” z informacją, czy użyto fallbacku czy nowej regeneracji, wraz z timestampem walidacji HMAC.

## Tabela funkcji i różnic

| Funkcja | Dudzian (Stage6) | CryptoHopper (publiczny plan) | Gunbot Ultimate | Status różnicy |
| --- | --- | --- | --- | --- |
| Portfolio adaptacyjne / rebalancing | PortfolioGovernor z integracją Stress Lab, override SLO i logiem HMAC dla każdej zmiany portfela. | Automatyczne rebalancingi strategii Pro, oparte o sygnały i copy trading. | Regułowe profile per para (DCA, grid, step gain), brak centralnego rebalancera ani guardraili. | Przewaga Dudzian – jedyny produkt z audytowalnym rebalancingiem i override’ami SLO. |
| Poziomy automatyzacji | Stage6 Hypercare Orchestrator łączy cykle portfela, resilience i observability w jednym przebiegu podpisanym HMAC. | Tryby automatyczny/półautomatyczny (strategie, trailing stop, copy bots). | Automation przez scheduler + TradingView alerts, brak orkiestratora HyperCare lub podpisanych cykli. | Przewaga Dudzian – utrzymać autonomiczny hypercare offline. |
| Obsługa wielu giełd | Integracje Binance, Coinbase, Kraken, OKX, Bitget, Bybit, KuCoin, Huobi, Gemini **oraz Deribit/BitMEX futures** z checklistami opisanymi w `reports/exchanges/2025-01-15.csv`. | Wsparcie >16 giełd, w tym Binance, Coinbase, Kraken, KuCoin, Huobi. | >100 integracji, lecz konfiguracja futures/HyperCare odbywa się ręcznie bez checklist ani podpisów. | Luka futures domknięta – utrzymać regresje adapterów i publikować raporty CSV jako dowód przewagi. |
| Marketplace strategii | Lokalny marketplace presetów (`PresetPublicationWorkflow`, wizard PySide6) z podpisami HMAC, recenzjami QA i katalogiem Markdown (`config/marketplace/catalog.md`). | Globalny marketplace z copy tradingiem, algorytmami społeczności. | Gunbot Marketplace/Gunthy Marketplace – brak person, brak podpisów HMAC i wymogu ≥15 strategii. | Przewaga Dudzian – audytowalny pipeline i persony. |
| Tryby AI Governor | AutoTrader AI Governor (scalping/hedge/grid) z telemetrią `riskMetrics`/`cycleMetrics`, test `tests/e2e/test_autotrader_autonomy.py::test_autotrader_ai_governor_snapshot_reports_mode`. | Tryby automatyczne/półautomatyczne wymagające ręcznej konfiguracji kosztów. | Brak natywnego AI – strategie oparte na wskaźnikach technicznych lub sygnałach TradingView. | Przewaga Dudzian – eksponować adaptacyjne tryby w marketingu. |
| Stress Lab i symulacje | Scenariusze multi-market, blackout infrastrukturalny i bundling raportów podpisanych HMAC. | Backtesting i paper trading, brak publicznych stres testów multi-market. | Backtesty i symulatory per para, brak orkiestracji DR ani podpisanych raportów Stress Lab. | Przewaga Dudzian – komunikować Stress Lab jako element wyróżniający. |
| Resilience / DR | ResilienceHypercareCycle, self-healing runtime, failover drill i bundler artefaktów podpisanych HMAC. | Failover podstawowy (API failover, monitoring uptime). | Self-hosted wdrożenia, odtwarzanie po awarii zależne od operatora (brak audytu). | Przewaga Dudzian – utrzymać przewagę w audycie DR. |
| UI decyzji | Dashboard QML z kartą „Decyzje AI” korzystającą z feedu gRPC `AutoTraderAIGovernor` (timeline, confidence, rekomendowane tryby i telemetry z blurami PySide6). | Webowy UI z dostępem do sygnałów i alertów w czasie rzeczywistym. | Konsola web/desktop z wykresami TradingView i panelami konfiguracji strategii – brak timeline decyzji AI. | Przewaga Dudzian – live timeline + SLA HyperCare w PySide6. |
| Compliance i audyt | TradingDecisionJournal, podpisy HMAC dla raportów hypercare oraz logowanie decyzji AI. | Raporty działania bota, brak potwierdzonych podpisów HMAC offline. | Brak podpisów HMAC; audyt ograniczony do logów runtime i powiadomień Telegram. | Przewaga Dudzian – utrzymać offline-first compliance. |

## Priorytety uzupełniania luk
1. **Utrzymanie pokrycia giełdowego**
   - Cel: utrzymać ≥12 giełd (w tym Deribit/BitMEX futures) z podpisanymi checklistami HyperCare oraz raportem `scripts/list_exchange_adapters.py` w pakiecie benchmarkowym, aby zachować przewagę nad CryptoHopperem i Gunbotem.
   - Metryki: liczba aktywnych adapterów, czas failover (p95), potwierdzenie `live_readiness_signed` w raporcie CSV.
   - Wymagane działania: regresje adapterów nightly, publikacja raportu `reports/exchanges/<data>.csv`, aktualizacja checklist HyperCare.
2. **Marketplace presetów i społeczności**
   - Cel: publiczny katalog presetów z recenzjami i kontrolą wersji offline – prezentowany w `config/marketplace/catalog.md` jako przewaga nad listingami CryptoHoppera/Gunbota.
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
| 2025-01-15 | Strategia, UI | Udokumentowano AI Governor, payload importu presetów do kreatora PySide6 oraz przewagę vs CryptoHopper (testy e2e). | `tests/e2e/test_autotrader_autonomy.py`, `tests/test_marketplace_workflow.py`, `reports/strategy/presets_2025-01-15.md` |

> Utrzymuj tabelę w formacie kroniki – każdy wpis powinien mieć link do źródłowych artefaktów oraz krótkie streszczenie wpływu na plan domykania luk.

## Procedura zbierania metryk
1. **Zrzut metryk automatyzacji:** uruchom `python scripts/run_stage6_hypercare_cycle.py --export var/audit/hypercare/<data>/summary.json` i zweryfikuj podpis HMAC (`python scripts/verify_stage6_hypercare_summary.py`).
2. **Pokrycie giełd:** uruchom `python scripts/list_exchange_adapters.py --report-date <data> --push-dashboard --dashboard-dir reports/exchanges/signal_quality --generate-hypercare-assets` – powstanie `reports/exchanges/<data>.csv`, checklisty HyperCare oraz dzienne snapshoty CSV w `reports/exchanges/signal_quality/`. Zweryfikuj kolumny `futures_margin_mode`, `liquidation_feed`, `hypercare_checklist_signed` oraz nowe `simulator_cost_status`/`simulator_cost_flags` (koszty z walidacji PaperFuturesSimulator muszą być zielone dla Deribit/BitMEX).
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
