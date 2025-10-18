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

## 7. Wskazówki bezpieczeństwa

* **Kopie zapasowe** – przed `purge` lub `delete` wykonaj kopię katalogu docelowego (np. `tar -czf backups/reports_$(date).tar.gz var/reports`).
* **Uprawnienia** – uruchamiaj komendy z konta technicznego o ograniczonych prawach (bez dostępu do innych udziałów sieciowych). Dla archiwów na S3 używaj dedykowanych kluczy z ograniczonym zakresem.
* **Walidacja ścieżek** – nigdy nie wskazuj `--destination` wewnątrz `var/reports`; narzędzie odrzuci taką ścieżkę, ale praktyka zmniejsza ryzyko utraty danych.
* **Integracja z UI** – dialogi „Usuń” i „Archiwizuj” w panelu administratora korzystają z tych samych komend. Potwierdzenie operacji w UI powinno być poprzedzone eksportem wyników `--dry-run` do wglądu przełożonego.
* **Monitorowanie** – po każdej operacji sprawdź logi (`logs/report_center.log`, `audit/paper_trading_log.md`) i metryki dyskowe. Duże operacje archiwizacyjne mogą wymagać ponownej indeksacji w monitoringu.

## 8. Materiały uzupełniające

* Dokument konfiguracji instalatora desktopowego: `docs/deployment/desktop_installer.md`.
* Opis modułu UI i mostka raportów: `ui/README.md`.
* Dziennik audytowy raportów: `docs/audit/paper_trading_log.md`.
