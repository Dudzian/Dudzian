# Diagnostyka i zgłoszenia serwisowe

Ten dokument opisuje proces przygotowania paczki diagnostycznej bota oraz sposoby jej wykorzystania w komunikacji ze wsparciem OEM.

## Generowanie paczki diagnostycznej z UI
1. Otwórz **Centrum pomocy** w aplikacji desktopowej i wybierz przycisk **Zgłoś problem**.
2. W oknie dialogowym *Zgłoszenie serwisowe* opisz krótko napotkany problem oraz potwierdź katalog, do którego zostanie zapisane archiwum ZIP.
3. Kliknij **Wygeneruj paczkę**. Po zakończeniu operacji otrzymasz informację o miejscu zapisu i stanie procesu.
4. Utworzone archiwum zawiera logi, bieżącą konfigurację oraz ostatnie raporty — przekaż je zespołowi wsparcia razem z opisem błędu.

> **Uwaga:** generowanie paczki może potrwać kilkanaście sekund w zależności od wielkości katalogu `logs/` oraz `reports/`.

## Generowanie paczki z linii poleceń
Do automatyzacji zgłoszeń można użyć narzędzia `scripts/generate_diagnostics.py`:

```bash
python scripts/generate_diagnostics.py \
    --description "Problem z synchronizacją giełdy" \
    --output-dir /tmp/diagnostics \
    --metadata '{"kontakt": "support@example.com"}'
```

Parametry skryptu:
- `--base-path` – katalog bazowy projektu (domyślnie bieżący katalog roboczy),
- `--output-dir` – katalog, w którym zostanie zapisane archiwum ZIP,
- `--description` – opis zgłoszenia widoczny w manifeście,
- `--extra` – dodatkowe pliki lub katalogi dołączane do paczki,
- `--metadata` – dodatkowe dane w formacie JSON.

Wyjście skryptu zawiera listę plików umieszczonych w paczce oraz ścieżkę do utworzonego archiwum. W przypadku błędów (np. braku logów) zwracany jest kod wyjścia różny od zera.

## Zawartość paczki
Wewnątrz archiwum znajduje się katalog `manifest/` z plikiem `diagnostics.json`, który przechowuje metadane zgłoszenia (opis, datę utworzenia oraz listę plików). Pozostałe katalogi odzwierciedlają strukturę `logs/`, `config/` oraz `reports/` z maszyny użytkownika.

Dzięki temu zespół wsparcia otrzymuje kompletny materiał do analizy i może szybciej zaproponować rozwiązanie problemu.
