# Raport audytu licencji

Dokument opisuje sposób generowania raportów licencyjnych oraz ich przeglądania
w aplikacji desktopowej.

## Zawartość raportu

Raport audytu składa się z następujących sekcji:

* **Podsumowanie** – liczba aktywacji, liczba unikalnych urządzeń, identyfikator
  licencji, edycja oraz znacznik czasu ostatniej aktywacji.
* **Bieżący status** – zawartość pliku `var/security/license_status.json`
  (jeśli dostępny), w tym aktywne moduły, informacje o utrzymaniu i okresie
  próbnym.
* **Historia aktywacji** – wpisy z dziennika `logs/security_admin.log`
  uporządkowane malejąco według czasu.
* **Ostrzeżenia** – komunikaty o brakujących plikach lub błędach parsowania.

Raport może być zapisany w formacie JSON (pełne dane) oraz Markdown (wersja
czytelna dla człowieka). Oba formaty zachowują informacje o ścieżkach plików,
dacie wygenerowania oraz ostrzeżeniach.

## Generowanie raportu z CLI

Do tworzenia raportów służy skrypt `scripts/manage_license.py`.

```bash
# podgląd raportu w formacie JSON
python -m scripts.manage_license audit \
    --status-path var/security/license_status.json \
    --audit-log logs/security_admin.log \
    --format json

# eksport do katalogu reports/security
python -m scripts.manage_license export \
    --output-dir reports/security \
    --basename license_audit_$(date +%Y%m%d)
```

Najważniejsze opcje:

| opcja            | opis                                                                 |
|------------------|----------------------------------------------------------------------|
| `--status-path`  | ścieżka do pliku statusu licencji (domyślnie `var/security/license_status.json`). |
| `--audit-log`    | ścieżka do dziennika audytu (`logs/security_admin.log`).              |
| `--limit`        | limit wpisów aktywacji w raporcie (domyślnie 50).                     |
| `--format`       | `json` lub `markdown` dla polecenia `audit`.                         |
| `--output-dir`   | katalog eksportu (polecenie `export`).                               |
| `--basename`     | nazwa bazowa plików eksportu.                                        |

Kod zakończy się statusem `2`, jeśli wystąpi błąd audytu (np. niepoprawny JSON), oraz
`3` przy błędach wejścia/wyjścia.

## Widok w UI

Komponent `LicenseAuditView.qml` udostępnia przegląd raportu bezpośrednio w
interfejsie użytkownika. Najważniejsze funkcje:

* przycisk **Odśwież** pobiera aktualne dane,
* sekcja **Historia aktywacji** wyświetla wpisy z dziennika wraz z informacją o
  powtórnych aktywacjach,
* sekcja **Eksport raportu** pozwala zapisać aktualny raport do wskazanego
  katalogu.

Widok korzysta z kontrolera `LicenseAuditController`, który można wstrzyknąć do
QML przez `contextProperty` albo zarejestrować jako singleton w aplikacji.

## Lokalizacja plików

Domyślne ścieżki wykorzystywane przez raport:

* `var/security/license_status.json` – status licencji i podpisane metadane,
* `logs/security_admin.log` – dziennik audytu generowany przez `LicenseService`.

Jeśli plik statusu lub dziennik nie istnieje, raport zostanie wygenerowany, a
informacja o brakującym zasobie pojawi się w sekcji ostrzeżeń.

