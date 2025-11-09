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

### 8.1. Monitorowanie i historia failoveru CCXT

Reporter jakości sygnałów publikuje teraz zdarzenia degradacji na magistrali `exchange.signal_quality.*`. Menedżer giełdy reaguje na nie, automatycznie włączając failover CCXT i emituje zdarzenia operacyjne:

* `exchange.failover.engaged` – fallback został aktywowany (`payload.previous_backend == "native"`).
* `exchange.failover.recovered` – powrót do backendu natywnego po ustąpieniu degradacji.

W telemetryce Prometheusa pojawiły się nowe metryki:

| Metryka | Opis |
| --- | --- |
| `exchange_failover_state{exchange="binance"}` | 0 – aktywny backend natywny, 1 – wymuszony fallback CCXT. |
| `exchange_failover_switch_total{exchange="binance",backend="ccxt|native"}` | Licznik przełączeń backendu, pomocny przy audycie incydentów. |

Procedura kontroli failoveru:

1. Sprawdź aktualny stan: `curl localhost:9000/metrics | grep exchange_failover_state`.
2. W runbooku incydentu zapisz wynik licznika `exchange_failover_switch_total` przed i po działaniach.
3. Zweryfikuj zdarzenia w logach (`logs/exchange_manager.log`) – powinny zawierać wpisy `exchange.failover.engaged` oraz `exchange.failover.recovered` z kontekstem `previous_backend`.

## 9. Eksport champion/challenger i reakcja na degradację modeli

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

### 9.3. Korelacja retreningów z blokadami guardrail

Nowy dashboard Grafany **Stage7 – Retraining vs Guardrail Correlation** (`deploy/grafana/provisioning/dashboards/stage7_retraining_guardrails.json`) zestawia liczbę zmian championa z aktywnością guardrail w horyzoncie 15/60 minut. Panel statystyczny „Udział zmian z guardrailem” pokazuje, jak duża część auto-promocji przebiegła przy wymuszonej blokadzie ryzyka – wartości >30 % wymagają przeglądu w `reports/exchanges/signal_quality`. Tabela „Ostatnie zmiany modelu” korzysta z metryki `auto_trader_model_change_timestamp_seconds` i pozwala szybko skorelować wpisy decision journala (`event=model_change`) z alertami guardrail. Przed zatwierdzeniem promocji championa przejrzyj dashboard i upewnij się, że wskaźnik guardrail nie utrzymuje się w strefie czerwonej – w przeciwnym razie rozważ pauzę automatyzacji (`ui_bridge promote --dry-run`) i dodatkową walidację challengera.

Uzupełniający dashboard **Stage8 – Retraining Guardrail Impact** (`deploy/grafana/provisioning/dashboards/stage8_retraining_guardrail_impact.json`) zestawia liczbę zakończonych cykli retrainingu (`auto_trader_retraining_cycles_total`) z histogramem blokad guardrail (`auto_trader_retraining_guardrail_blocks_total`) oraz globalnym licznikiem `auto_trader_guardrail_blocks_total`. Panel statystyczny „Udział retrainingu z guardrail” na horyzoncie 2 godzin wskazuje procent cykli wykonanych w stanie awaryjnym – wartości powyżej 20 % powinny uruchomić przegląd decyzji w journali i panelu AutoMode (sekcja „Ostatnie zdarzenia model_changed”). Tabela z metryką `auto_trader_retraining_timestamp_seconds` pozwala szybciej zweryfikować, czy najnowszy retrening nastąpił podczas blokady ryzyka. Dashboard jest referencją przy analizie wpływu retreningu na automatyczną blokadę handlu w kolejnych sprintach.

## 10. Materiały uzupełniające

* Dokument konfiguracji instalatora desktopowego: `docs/deployment/desktop_installer.md`.
* Walidacja hooka HWID i keyring: `docs/deployment/installer_build.md#walidacja-hooka-hwid-i-keyring`.
* Opis modułu UI i mostka raportów: `ui/README.md`.
* Dziennik audytowy raportów: `docs/audit/paper_trading_log.md`.
