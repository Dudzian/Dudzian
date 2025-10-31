# Scenariusz E2E: przejście demo → paper

Ten scenariusz ilustruje minimalny przebieg przygotowujący użytkownika do startu w trybie paper po zakończeniu demonstracji na lokalnej instalacji.

## Wymagania wstępne
- lokalne środowisko z zainstalowanymi zależnościami projektu;
- dostęp do konfiguracji `config/e2e/demo_paper.yml` oraz domyślnego `config/core.yaml`;
- brak aktywnych procesów runtime korzystających z tego samego katalogu raportów (`logs/e2e`).

## Krok po kroku
1. **Uruchom tryb demo**
   ```bash
   python scripts/run_local_bot.py \
       --config config/e2e/demo_paper.yml \
       --mode demo \
       --state-dir var/runtime \
       --report-dir logs/e2e \
       --report-markdown-dir reports/e2e
   ```
   Po zamknięciu procesu w katalogu `var/runtime` zapisany zostanie checkpoint (`state.json`), a w `logs/e2e` pojawi się raport z walidacji.

2. **Przejdź do trybu paper** – wymaga zakończonej fazy demo.
   ```bash
   python scripts/run_local_bot.py \
       --config config/e2e/demo_paper.yml \
       --mode paper \
       --state-dir var/runtime \
       --report-dir logs/e2e \
       --report-markdown-dir reports/e2e
   ```
   Skrypt zweryfikuje checkpoint i zapisze nowy raport. Brak checkpointu spowoduje przerwanie uruchamiania.

3. **Opcjonalnie przejdź do live** – dostępne tylko po przygotowaniu sekcji `execution.live` w runtime.
   ```bash
   python scripts/run_local_bot.py --mode live [...]
   ```
   Jeśli konfiguracja nie zawiera aktywnego trybu live, narzędzie zwróci błąd walidacji.

## Raportowanie i walidacja
- Każde uruchomienie generuje plik `logs/e2e/run_local_bot_<mode>_<timestamp>.json` zawierający:
  - użyty punkt wejścia i środowisko,
  - wyniki walidacji poświadczeń API oraz symboli,
  - identyfikator checkpointu (jeżeli został zapisany lub odczytany),
  - listę ostrzeżeń/błędów.
- Dodatkowo powstaje raport Markdown `reports/e2e/demo_paper_<mode>_<timestamp>.md` z KPI (czas wykonania, liczba zdarzeń tradingowych, ostrzeżenia/błędy) oraz statusem poszczególnych kroków.
- Checkpoint demo (`var/runtime/state.json`) przechowuje nazwę entrypointu, tryb oraz metadane (np. endpoint metryk) i jest wymagany przed przejściem do kolejnych faz.

## Sprzątanie
Aby ponowić scenariusz od początku, usuń checkpoint i raporty:
```bash
rm -f var/runtime/state.json
rm -f logs/e2e/run_local_bot_*.json
```
