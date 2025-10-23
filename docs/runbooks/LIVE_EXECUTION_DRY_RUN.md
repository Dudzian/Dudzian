# Runbook: Live Execution Dry-Run

Ten dokument opisuje procedurę walidacji konfiguracji egzekucji live bez
wysyłania zleceń na giełdy. Celem dry-runa jest upewnienie się, że
konfiguracja `config/core.yaml`, rejestr adapterów giełdowych oraz
instrumenty używane przez środowiska live są spójne i gotowe do startu.

## Krok po kroku

1. **Przygotuj środowisko**
   - Zaktualizuj repozytorium do wersji zawierającej `scripts/live_execution_dry_run.py`.
   - Upewnij się, że plik `config/core.yaml` zawiera aktualne środowiska typu
     `live` oraz poprawnie skonfigurowane instrumenty i routingi.
   - (Opcjonalnie) przygotuj katalog docelowy na raporty, np.
     `mkdir -p logs/live_dry_run`.

2. **Uruchom skrypt dry-run**
   ```bash
   python scripts/live_execution_dry_run.py \
       --config config/core.yaml \
       --environment binance_live \
       --environment kraken_live \
       --report logs/live_dry_run/latest.json \
       --decision-log audit/decision_logs/live_execution.jsonl \
       --decision-log-hmac-key-file secrets/live_decision_log.key \
       --decision-log-key-id live-router-dry-run
   ```
   - `--environment` można powtarzać dla wielu środowisk; brak argumentu
     oznacza walidację wszystkich środowisk live z konfiguracji.
   - Raport JSON zawiera status każdego środowiska, wynik inicjalizacji
     adapterów oraz podsumowanie symulowanych zleceń paper tradingu.

3. **Zweryfikuj raport**
   - Sprawdź sekcję `status` w pliku raportu – wartość `PASS` oznacza brak
     błędów.
   - Upewnij się, że każda giełda (`adapters`) ma status `PASS` i poprawny
     identyfikator klasy.
   - W sekcji `simulation` zweryfikuj liczbę wykonanych zleceń oraz liczbę
     wpisów w ledgerze symulatora.

4. **Zatwierdź decision log**
   - Skrypt automatycznie dopisuje wpis JSONL z kategorią
     `live_execution_dry_run`. Wpis zawiera status, ścieżkę raportu i liczbę
     zleceń wykonanych w symulacji.
   - Jeżeli podano klucz HMAC (inline/plik/env), podpis zostanie dodany do
     wpisu. Zweryfikuj podpis poleceniem:
     ```bash
     python scripts/verify_decision_log.py summary \
         --log audit/decision_logs/live_execution.jsonl
     ```

5. **Udokumentuj wynik**
   - Dodaj link do raportu i numer joba/commitu w runbooku promocji live.
   - W razie problemów dołącz logi ze standardowego wyjścia skryptu oraz
     fragment raportu JSON opisujący błędy adapterów.

## Interpretacja błędów

| Sekcja           | Co oznacza błąd                                              | Działania naprawcze                               |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `adapters`       | Brak fabryki lub błąd inicjalizacji adaptera live            | Zweryfikuj `exchange_adapters`, zależności modułu |
| `simulation`     | Liczba wykonanych zleceń równa 0                             | Sprawdź uniwersum instrumentów i symbole          |
| `decision_log`   | Brak wpisu/klucza HMAC                                       | Upewnij się, że wskazano poprawny plik lub zmienną|

Dry-run należy wykonać ponownie po każdej zmianie kluczy API, konfiguracji
`runtime.live_routing` lub zestawu instrumentów środowiska live.

