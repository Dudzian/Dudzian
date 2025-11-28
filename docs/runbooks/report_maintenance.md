# Runbook: Utrzymanie raportów operacyjnych

Ten runbook opisuje procedury obsługi raportów generowanych przez `bot_core.reporting` i mostek `ui_bridge`. Skupiamy się na czterech komendach CLI (`overview`, `delete`, `purge`, `archive`), filtrach umożliwiających selekcję raportów oraz trybie podglądu `--dry-run`. Zawiera także scenariusze krok-po-kroku i wskazówki bezpieczeństwa pomagające chronić dane audytowe.

> **Środowisko referencyjne**: katalog bazowy raportów `var/reports`, archiwa wynikowe w `audit/smoke_archives/` lub katalogach operatorskich, interpreter Pythona zainstalowany w środowisku uruchomieniowym aplikacji desktopowej.

## 1. Przegląd raportów (`overview`)

Polecenie `python -m bot_core.reporting.ui_bridge overview` zwraca listę raportów w formacie JSON. Najważniejsze parametry:

| Parametr | Opis |
| --- | --- |
| `--base-dir` | Wskazuje katalog raportów (domyślnie `var/reports`). |
| `--since` / `--until` | Filtrują raporty po dacie utworzenia/aktualizacji (ISO-8601). |
| `--category` | Pozwala ograniczyć wynik do wybranych kategorii (można podać wielokrotnie). |
| `--summary-status` | `any`, `valid`, `missing`, `invalid` – filtruje status podsumowań. |
| `--limit` / `--offset` | Paginuje wynik (np. `--limit 25 --offset 25`). |
| `--sort` / `--sort-direction` | Sortuje po `updated_at`, `created_at`, `name` lub `size` rosnąco/malejąco. |
| `--query` | Tekstowe wyszukiwanie po nazwie, kategorii i metadanych. |
| `--has-exports` | `any`, `yes`, `no` – filtruje raporty posiadające eksporty. |

Wynik zawiera sekcje `reports`, `summary`, `categories` oraz `pagination`, dzięki czemu łatwo ocenić wolumen danych, najnowsze aktualizacje oraz brakujące podsumowania.

## 2. Usuwanie pojedynczych raportów (`delete`)

`python -m bot_core.reporting.ui_bridge delete <ścieżka>` usuwa wskazany raport lub katalog eksportów. Kluczowe opcje:

* `path` – ścieżka względna względem `--base-dir` (np. `daily/2024-04-01/paper_binance.zip`). Ścieżki bezwzględne są dopuszczalne, ale zostają zweryfikowane pod kątem przynależności do katalogu bazowego.
* `--base-dir` – katalog z raportami (domyślnie `var/reports`).
* `--dry-run` – drukuje statystyki (`removed_files`, `removed_size`) bez usuwania plików.

Komenda zawsze raportuje status (`preview`, `deleted`, `not_found`, `error`) oraz pełną ścieżkę, co ułatwia logowanie audytowe.

## 3. Masowe czyszczenie (`purge`)

`python -m bot_core.reporting.ui_bridge purge` stosuje te same filtry co `overview`, ale usuwa wszystkie dopasowane raporty. Dodatkowe zasady:

* Raporty zagnieżdżone wewnątrz innego dopasowania są pomijane, by uniknąć podwójnego liczenia.
* `--dry-run` tworzy podgląd (`targets.status == "preview"`) z informacją o liczbie plików i rozmiarze do usunięcia.
* Wynik zawiera `matched_count`, `planned_count`, `deleted_count` i listę `targets` z pełnymi ścieżkami.

## 4. Archiwizacja (`archive`)

Komenda `python -m bot_core.reporting.ui_bridge archive` kopiuje raporty do katalogu archiwum. Oprócz filtrów znanych z `overview/purge` obsługuje parametry:

| Parametr | Opis |
| --- | --- |
| `--destination` | Docelowy katalog lub plik archiwum. Domyślnie `<base>_archives`. |
| `--format` | `directory`, `zip`, `tar` – decyduje o formacie archiwum. |
| `--overwrite` | Pozwala zastąpić istniejące archiwa (inaczej raport zostanie pominięty). |
| `--dry-run` | Oblicza liczbę kopiowanych plików/rozmiar bez tworzenia archiwów. |

Ścieżki docelowe muszą znajdować się poza katalogiem bazowym raportów. W trybie `zip` i `tar` pliki są pakowane do pojedynczego archiwum o nazwie pochodzącej od identyfikatora raportu.

## 5. Tryb `--dry-run`

W każdej komendzie `--dry-run` pozwala ocenić skutki operacji, co jest wymagane przed wykonaniem trwałych zmian. Runbook zaleca, by:

1. Wykonać `overview` z tymi samymi filtrami, by potwierdzić, które raporty zostaną objęte operacją.
2. Włączyć `--dry-run`, zapisać wynik (JSON) w katalogu audytowym i poprosić przełożonego o akceptację.
3. Włączyć właściwą komendę bez `--dry-run` dopiero po otrzymaniu zgody oraz wykonaniu kopii zapasowej.

## 6. Scenariusze operacyjne

### 6.1. Czyszczenie starych raportów smoke-testowych

1. Uruchom podgląd: `python -m bot_core.reporting.ui_bridge purge --category smoke --until 2024-04-01 --dry-run`.
2. Zweryfikuj wynik (`targets`, `removed_size`) i sprawdź, czy lista nie zawiera raportów wymaganych przez audyt.
3. Utwórz backup katalogu (`rsync -av var/reports/smoke/ audit/backups/smoke_$(date +%Y%m%d)/`).
4. Ponownie wykonaj komendę bez `--dry-run`. Zachowaj plik JSON z wynikiem w `audit/maintenance_logs/`.

### 6.2. Archiwizacja raportów dziennych do ZIP

1. Przygotuj katalog docelowy, np. `mkdir -p audit/daily_archives`.
2. Wykonaj podgląd: `python -m bot_core.reporting.ui_bridge archive --category daily --since 2024-05-01 --format zip --destination audit/daily_archives --dry-run`.
3. Po zatwierdzeniu wyniku uruchom komendę bez `--dry-run`.
4. Zweryfikuj, że archiwa ZIP zawierają pliki `summary.json` i `ledger.csv`, a wynik komendy ma `status=completed`.

### 6.3. Przywrócenie miejsca na dysku po raporcie awaryjnym

1. `python -m bot_core.reporting.ui_bridge overview --query incident` – identyfikuj katalog raportu awaryjnego.
2. `python -m bot_core.reporting.ui_bridge delete emergency/2024-06-12 --dry-run` – sprawdź liczbę plików.
3. Upewnij się, że z raportu istnieje kopia w archiwum (`audit/emergency_archives/`).
4. Usuń katalog (`delete` bez `--dry-run`) i wpisz notatkę w `docs/audit/paper_trading_log.md`.

### 6.4. Ręczna promocja challengera do championa

1. Ustal wersję challengera w panelu Strategy Management lub poprzez `python -m bot_core.reporting.ui_bridge champion`.
2. Wykonaj podgląd raportu jakości (`overview`) i upewnij się, że posiadasz artefakty audytowe dla wskazanej wersji.
3. Uruchom komendę: `python -m bot_core.reporting.ui_bridge promote --model decision_engine --version v2 --reason "Manual override"`.
4. Narzędzie utworzy wpis audytowy w `audit/champion_promotions/<model>/<timestamp>_<version>/summary.json` oraz zaktualizuje `champion.json` i `challengers.json`.
5. Zweryfikuj w UI (panel Strategy Management) pojawienie się nowego championa oraz log `ReportCenter` (ostatnia notyfikacja powinna zawierać uzasadnienie promocji).

## 7. Wskazówki bezpieczeństwa

* **Kopie zapasowe** – przed `purge` lub `delete` wykonaj kopię katalogu docelowego (np. `tar -czf backups/reports_$(date).tar.gz var/reports`).
* **Uprawnienia** – uruchamiaj komendy z konta technicznego o ograniczonych prawach (bez dostępu do innych udziałów sieciowych). Dla archiwów na S3 używaj dedykowanych kluczy z ograniczonym zakresem.
* **Walidacja ścieżek** – nigdy nie wskazuj `--destination` wewnątrz `var/reports`; narzędzie odrzuci taką ścieżkę, ale praktyka zmniejsza ryzyko utraty danych.
* **Integracja z UI** – dialogi „Usuń” i „Archiwizuj” w panelu administratora korzystają z tych samych komend. Potwierdzenie operacji w UI powinno być poprzedzone eksportem wyników `--dry-run` do wglądu przełożonego.
* **Monitorowanie** – po każdej operacji sprawdź logi (`logs/report_center.log`, `audit/paper_trading_log.md`) i metryki dyskowe. Duże operacje archiwizacyjne mogą wymagać ponownej indeksacji w monitoringu.

## 8. Raporty jakości sygnałów giełdowych

Moduł giełdowy generuje raporty jakości sygnałów w `reports/exchanges/signal_quality`. Każdy plik `<exchange>.json` zawiera:

* listę ostatnich realizacji (z backendem `native` lub `ccxt`),
* zagregowane fill ratio oraz poślizg (w punktach bazowych),
* statusy operacji wykorzystywane do audytu failoveru.

Do szybkiej inspekcji wykorzystaj `jq` lub `python -m json.tool`. Przykład: `jq '.records[-1]' reports/exchanges/signal_quality/binance.json`. Jeśli pole `failures` jest większe od zera, uruchom runbook `docs/runbooks/STAGE6_RESILIENCE_CHECKLIST.md`, aby potwierdzić stabilność limitów API.

Raporty podlegają tym samym zasadom retencji co inne katalogi w `var/reports`. Przed czyszczeniem upewnij się, że wpisy nie są wymagane przez audyt incydentów lub przeglądów miesięcznych.

Komenda `python -m bot_core.reporting.ui_bridge purge` automatycznie czyści katalog jakości sygnałów na podstawie progu retencji (domyślnie 30 dni). W razie potrzeby możesz wskazać alternatywną lokalizację (`--signal-quality-dir`) lub zmodyfikować okres przechowywania (`--signal-quality-retention-days`). Wynik polecenia zawiera sekcję `signal_quality_cleanup` z liczbą usuniętych plików oraz ewentualnymi błędami, co ułatwia logowanie operacji w runbooku.

### 8.1. Raport adapterów futures i eksport do dashboardu

Benchmark CryptoHoppera/Gunbota wymaga comiesięcznego CSV z listą adapterów giełdowych, w którym uwzględnione są kolumny `futures_margin_mode`, `liquidation_feed`, `hypercare_checklist_signed`, `missing_required_documents`, `futures_checklist_id`, `futures_checklist_ready` oraz statusy `hypercare_failover_status`/`hypercare_latency_status`/`hypercare_cost_status`. Raport generujemy poleceniem:

```
python scripts/list_exchange_adapters.py \
  --report-date $(date +%Y-%m-%d) \
  --report-dir reports/exchanges \
  --push-dashboard \
  --dashboard-dir reports/exchanges/signal_quality \
  --hypercare-config config/stage6/hypercare.yaml
```

Polecenie utworzy plik `reports/exchanges/<data>.csv` oraz skopiuje go do `reports/exchanges/signal_quality/`, tak aby dashboard Prometheusa/Grafany mógł pobierać najnowszy snapshot. Jeśli CI/HyperCare musi wypchnąć dane do zewnętrznego datasource, dodaj `--dashboard-endpoint https://grafana.example/api/ds/push` – w przypadku błędu publikacji skrypt zakończy się statusem !=0.

**Weryfikacja futures i HyperCare:** po wygenerowaniu CSV sprawdź, że wiersze `deribit,live` i `bitmex,live` mają `hypercare_checklist_signed == True`, `futures_checklist_ready == True` oraz pustą kolumnę `missing_required_documents`. Jeśli wartości są puste lub `False`, oznacza to brak podpisu checklisty HyperCare i należy otworzyć zadanie w HyperCare/Compliance. Kolumna `liquidation_feed` powinna wskazywać pełny URL kanału long-pollowego (np. `https://stream.hyperion.dudzian.ai/exchanges/deribit/futures/private`) – użyj jej w dashboardzie do szybkiego porównania konfiguracji feedów. Statusy `hypercare_failover_status`/`hypercare_latency_status`/`hypercare_cost_status` muszą raportować `ready` (źródłem prawdy jest `config/stage6/hypercare.yaml`); odchylenia blokują publikację benchmarku CryptoHopper/Gunbot.

**Nowe kolumny jakości sygnałów:** raport wymusza obecność dziennych snapshotów `reports/exchanges/signal_quality/*.json` dla `deribit_futures` i `bitmex_futures`. Kolumny `signal_quality_snapshot_status`/`signal_quality_snapshot_age_minutes` muszą raportować `fresh` oraz wiek < 48h – w przeciwnym razie skrypt zakończy się błędem. Wartości `signal_quality_records` pomagają szybko wychwycić puste raporty przed publikacją benchmarku.

Eksport checklisty HyperCare w adapterach Deribit/BitMEX korzysta z istniejącego `SignalQualityReporter` (lub ostatniego snapshotu z dysku), więc przed uruchomieniem eksportu w pipeline CI/HyperCare upewnij się, że reporter zebrał realne rekordy z runtime. W przeciwnym razie checklisty zostaną podpisane, ale wskażą zerowe metryki jakości, co zablokuje publikację dashboardu.

### 8.2. Kopiowanie snapshotów do pakietu marketingowego
1. Wygeneruj najnowsze raporty: `python scripts/run_stress_lab.py run --config config/core.yaml --output reports/stress_lab/latest.json --signing-key-env STRESS_LAB_HMAC` oraz `python scripts/list_exchange_adapters.py --report-date $(date +%Y-%m-%d) --report-dir reports/exchanges --dashboard-dir reports/exchanges/signal_quality`.
2. Uruchom bundler marketingowy: `python scripts/export_marketing_bundle.py --report-range $(date +%Y-%m-%d) --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC`. Skrypt skopiuje najświeższe pliki z `reports/stress_lab/` oraz checklisty z `reports/exchanges/signal_quality/` do `var/marketing/benchmark/` i utworzy manifest linków.
3. Zweryfikuj podpisy HMAC: `python scripts/export_marketing_bundle.py --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC --validate-only` – manifest zostanie porównany z kluczem release’owym. W przypadku rozbieżności zatrzymaj publikację i ponów eksport po zweryfikowaniu źródeł.
4. Zanim opublikujesz materiały marketingowe, upewnij się, że w katalogu `var/marketing/benchmark/` znajdują się co najmniej: `stress_lab/*.json` + `.sig` + `.manifest.json`, `signal_quality/*.json` + nowszy CSV adapterów oraz podpisany manifest bundla marketingowego.

## 9. Publikacja snapshotów i stres-testów do marketingu oraz release notes

### 9.1. Przygotowanie materiałów źródłowych
1. Zweryfikuj świeżość stres-testów: `find reports/stress_lab -name "*.json" -mtime -3` i `find reports/exchanges/signal_quality -name "*.json" -mtime -2`. Brak wyników dla kluczowych giełd (`binance`, `coinbase`, `deribit_futures`, `bitmex_futures`) oznacza blokadę publikacji.
2. Wygeneruj checklistę adapterów: `python scripts/list_exchange_adapters.py --report-date $(date +%Y-%m-%d) --report-dir reports/exchanges --push-dashboard --dashboard-dir reports/exchanges/signal_quality --hypercare-config config/stage6/hypercare.yaml`.
3. Zweryfikuj kolumny `hypercare_checklist_signed`, `signal_quality_snapshot_status`, `futures_checklist_ready` oraz `hypercare_cost_status` w wygenerowanym CSV (`reports/exchanges/<data>.csv`). Wszelkie wartości różne od `True`/`ready` dla Deribit/BitMEX blokują publikację.

### 9.2. Budowanie i publikacja pakietu
1. Zbuduj bundel marketingowy ze stres-testami i checklistą: `python scripts/export_marketing_bundle.py --report-range $(date +%Y-%m-%d) --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC --include-signal-quality`.
2. Uruchom walidację podpisu: `python scripts/export_marketing_bundle.py --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC --validate-only` – bez poprawnej walidacji publikacja jest niedozwolona.
3. Dołącz bundel do release notes: w sekcji „Release artifacts” dodaj link do `var/marketing/benchmark/manifest.json` oraz CSV checklisty (`reports/exchanges/<data>.csv`).
4. Przekaż marketingowi ścieżki `var/marketing/benchmark/stress_lab/`, `var/marketing/benchmark/signal_quality/` oraz podpis `benchmark_marketing_bundle.sig`; upewnij się, że linki trafią do whitepaper/case studies.
5. Zweryfikuj, że `signal_quality/index.csv` zawiera komplet giełd z CSV checklisty oraz że kolumna `snapshot_created_at` ma wartości <48 h; w razie braków ponów eksport po uzupełnieniu snapshotów.
6. Potwierdź spójność bundla: liczba rekordów `signal_quality/index.csv` powinna odpowiadać liczbie plików `.json` w `reports/exchanges/signal_quality/`, a ścieżki w `manifest.json` muszą zgadzać się z artefaktami wypisanymi w release notes; w przypadku lustrzanego bucketa S3/Git porównaj sumę SHA256 `index.csv` z wersją zarchiwizowaną w `var/audit/benchmark/<data>/`.
7. Zarchiwizuj paczkę w `var/audit/benchmark/<data>/` (kopiuj cały katalog bundla wraz z podpisem) i odnotuj w dzienniku `docs/audit/paper_trading_log.md` datę publikacji oraz listę artefaktów.
8. Przeklej skrót (źródła, ścieżki artefaktów, status walidacji HMAC) do szablonu release notes i zgłoszenia marketingowego; brak wpisu blokuje zamknięcie releasu.
9. Zsynchronizuj lustrzane repozytorium (S3/Git): `aws s3 sync var/marketing/benchmark/ s3://<bucket>/benchmark/ --delete` lub `git add var/marketing/benchmark && git commit -m "Update benchmark bundle"`; po synchronizacji wykonaj `sha256sum` na `index.csv` i `manifest.json` w obu lokalizacjach i zapisz wynik w audycie. Rozbieżności oznaczają blokadę publikacji do czasu wyrównania hashy.

### 9.3. Obsługa odchyleń i publikacja cross-channel
1. Jeśli bundler zwróci niespójny manifest (`signal_quality` lub `stress_lab` nie zawiera pełnych ścieżek), uruchom go ponownie z `--force-rebuild --include-signal-quality`, a poprzedni manifest zachowaj w `var/audit/benchmark/<data>/failed_manifest.json` jako dowód audytowy.
2. W przypadku braków snapshotów w `index.csv` (< liczby plików `.json`), dobuduj stres-testy dla brakujących giełd (`scripts/run_stress_lab.py run --exchanges <lista>`) i powtórz eksport; dopiero po zgodności liczby wierszy i plików opublikuj bundel.
3. Po zatwierdzeniu bundla opublikuj linki w trzech miejscach: release notes (sekcja „Release artifacts”), zgłoszenie marketingowe (biała księga/case studies) oraz wewnętrzny kanał operacyjny; wszędzie wklej sumę SHA256 `benchmark_marketing_bundle.sig` oraz timestamp walidacji HMAC.
4. Jeśli marketing korzysta z lustrzanego repozytorium S3/Git, porównaj sumę SHA256 `signal_quality/index.csv` z wersją w lustrze; w razie rozbieżności wykonaj `aws s3 cp var/marketing/benchmark/signal_quality/index.csv s3://<bucket>/benchmark/ --metadata sha256=<hash>` lub odpowiedni `git commit` w repo marketingowym, dokumentując zmianę w `docs/audit/paper_trading_log.md`.
5. W przypadku opóźnionego potwierdzenia lustrzanego (np. region S3 z opóźnioną spójnością) oznacz release notes statusem „oczekuje na parzystość lustra” i w ciągu 24 h powtórz walidację hashy; brak zgodności po 24 h wymaga rollbacku bundla do ostatniej zweryfikowanej wersji.

### 9.4. Monitoring powydawniczy i rollback
1. Utwórz job kontrolny (CI lub cron) uruchamiany co 12 h: `find reports/exchanges/signal_quality -name "*.json" -mtime -2 | wc -l` oraz `python scripts/export_marketing_bundle.py --destination var/marketing/benchmark --signing-key-env MARKETING_BUNDLE_HMAC --validate-only`. Jeżeli liczba plików jest mniejsza niż liczba wierszy w `reports/exchanges/<data>.csv` lub walidacja HMAC zwraca błąd, eskaluj do HyperCare i marketingu.
2. W razie wykrycia niespójności po publikacji (różne sumy SHA między lustrami lub brakujące snapshoty) wykonaj regenerację bundla z `--force-rebuild --include-signal-quality`, a poprzednią wersję przenieś do `var/audit/benchmark/<data>/superseded/` wraz z notatką audytową opisującą różnicę.
3. Jeżeli release notes zostały już podpisane, dołącz erratę z nową sumą SHA256 i timestampem walidacji HMAC; w marketingowym repo/buckecie dodaj plik `ERRATA_<data>.md` z listą zmienionych artefaktów i linkiem do nowego manifestu.
4. Po rollbacku zaktualizuj `docs/audit/paper_trading_log.md` oraz wewnętrzny kanał operacyjny, wskazując przyczynę, zakres poprawki i potwierdzenie ponownej walidacji bundla.
5. Gdy rollback nastąpił z powodu niespójności lustra, odnotuj w audycie, który bucket/branch posiadał odchylenie oraz dołącz wynik `sha256sum` przed i po korekcie; pozwoli to na późniejszą retrospekcję SLA replikacji.
6. Jeśli w trakcie walidacji bundla pojawi się nowy snapshot lub checklista (zmieniający liczbę rekordów vs. release notes), zamroź bundel w wersji z artefaktami użytymi do podpisu (`freeze`), wykonaj rebuild na kopii roboczej i dopiero po zatwierdzeniu audytowym zamień wersję w marketingu/release notes.
7. Przy każdym rebuildzie, który dotyka release notes lub zgłoszenia marketingowego, zaktualizuj dziennik z odniesieniem do wersji manifestu (`manifest.json`/`benchmark_marketing_bundle.sig`) oraz dołącz tabelę „przed/po” (liczba snapshotów, hash `index.csv`, hash `manifest.json`).

### 9.5. Checklist publikacji wielokanałowej
1. Przed publikacją na kanałach zewnętrznych (whitepaper, case studies, newsletter) wygeneruj diff bundla: `python scripts/export_marketing_bundle.py --destination var/marketing/benchmark --diff-only` i dołącz wynik do zgłoszenia marketingowego; brak diffu = blokada publikacji.
2. Zweryfikuj parytet liczby wpisów w `signal_quality/index.csv` z liczbą plików `.json` w `reports/exchanges/signal_quality/` oraz liczbą wierszy w `reports/exchanges/<data>.csv`; odchylenie >0 = wymagany rebuild i adnotacja w audycie.
3. Zanim opublikujesz linki, dodaj zapis „proof-of-source” w `docs/audit/paper_trading_log.md` zawierający hash `manifest.json`, sumę SHA256 `benchmark_marketing_bundle.sig` oraz ścieżkę archiwum w `var/audit/benchmark/<data>/`.
4. Jeśli liczba wierszy `index.csv` zmienia się o >10% względem poprzedniego releasu, dołącz tabelę różnic (`added_exchanges`, `dropped_exchanges`) do release notes oraz oznacz release statusem „ważna zmiana pokrycia” w komunikacji marketingowej.
5. Po dystrybucji do luster S3/Git wykonaj `sha256sum` na `index.csv` i `manifest.json` w obu lokalizacjach; wynik zapisz w `var/audit/benchmark/<data>/parity_report.json` i podlinkuj w release notes. Rozbieżność hashy = blokada i rollback bundla zgodnie z pkt 9.4.

### 9.6. Proof-of-source i rotacja bundli marketingowych
1. Po pozytywnej walidacji bundla utwórz plik `var/audit/benchmark/<data>/marketing_bundle_proof.md` zawierający: listę giełd z `signal_quality/index.csv`, hash `index.csv`, hash `manifest.json`, sumę SHA256 `benchmark_marketing_bundle.sig`, timestamp walidacji HMAC oraz adres lustra S3/Git.
2. Do release notes i zgłoszenia marketingowego wklej link do `marketing_bundle_proof.md`; brak linku blokuje publikację materiałów zewnętrznych.
3. Jeśli bundel został opublikowany w trybie „freeze” (np. oczekiwanie na parytet luster), dopisz sekcję „status freeze” w `marketing_bundle_proof.md` z datą rozpoczęcia, listą brakujących hashy i planem odblokowania.
4. Gdy parytet hashy zostanie osiągnięty, zaktualizuj `marketing_bundle_proof.md` o tabelę „przed/po” (hash `index.csv`, hash `manifest.json`, liczba snapshotów) i oznacz datę odblokowania; w release notes dodaj informację o usunięciu statusu freeze.
5. Przy każdym rebuildzie bundla z nowym zakresem danych pozostaw poprzedni `marketing_bundle_proof.md` w `var/audit/benchmark/<data>/superseded/` wraz z linkiem do erraty i hashami wersji bieżącej – umożliwia to śledzenie rotacji w audycie marketingu.

## 10. Eksport champion/challenger i reakcja na degradację modeli

### 9.1. Generowanie raportu porównawczego championów

1. Uruchom narzędzie audytowe: `python -m scripts.audit.champion_diff --lhs deploy/packaging/samples/var/models/quality --rhs var/models/quality`.
2. W katalogu `audit/champion_diffs/` pojawi się plik `champion_diff_<data>.json` zawierający listę modeli, metadane championa oraz różnice w metrykach i parametrach.
3. Jeżeli chcesz ograniczyć analizę do wybranych modeli, dodaj `--model decision_engine --model risk_model`. Argument `--output-dir` pozwala wskazać alternatywne miejsce zapisu (np. katalog współdzielony z QA).
4. Raport można przekazać do pipeline CI lub dołączonego zgłoszenia serwisowego – format JSON zawiera sekcje `metrics_delta`, `parameter_changes` oraz status brakujących artefaktów.

### 9.2. Eskalacja i reagowanie na degradację championa

1. Jeśli w raporcie `champion_diff` sekcja `metrics_delta` sygnalizuje spadek kluczowych metryk (np. `directional_accuracy`, `expected_pnl`) poniżej ustalonych progów, uruchom dodatkową walidację na challengeru (`python -m bot_core.reporting.ui_bridge champion --model <nazwa>`).
2. Gdy degradacja zostanie potwierdzona, wykonaj „suchy” przegląd rekomendacji AI: `python -m bot_core.runtime.ui_bridge snapshot --model <nazwa> --dry-run`, a wynik zapisz w `audit/champion_promotions/preview_<data>.json`.
3. Zwołaj komitet operacyjny i przygotuj promocję najlepszego challengera: `python -m bot_core.reporting.ui_bridge promote --model <nazwa> --version <challenger_version> --reason "Degradacja championa"`.
4. Po promocji zweryfikuj w panelu Strategy Management, że nowy champion jest aktywny, a wpis audytowy pojawił się w `audit/champion_promotions/<model>/`.
5. Uzupełnij dziennik operacyjny (`docs/audit/paper_trading_log.md`) o decyzję, dołącz raport `champion_diff` i snapshot rekomendacji, aby zachować pełny kontekst eskalacji.

## 11. Materiały uzupełniające

* Dokument konfiguracji instalatora desktopowego: `docs/deployment/desktop_installer.md`.
* Walidacja hooka HWID i keyring: `docs/deployment/installer_build.md#walidacja-hooka-hwid-i-keyring`.
* Opis modułu UI i mostka raportów: `ui/README.md`.
* Dziennik audytowy raportów: `docs/audit/paper_trading_log.md`.
