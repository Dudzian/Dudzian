# OfflineRuntimeService – scenariusze użycia

## Przygotowanie snapshotu strumieniowego
1. Uruchom narzędzie `scripts/capture_stream_snapshot.py` wskazując adres lokalnego mostka long-pollowego.
2. Narzędzie zapisze plik JSON zawierający zdarzenia uporządkowane według znacznika czasu oraz wyposażone w pole `sequence`.
3. Snapshot można zaktualizować ręcznie, ale należy zachować rosnącą kolejność `timestamp_ms`; w przeciwnym razie runtime zignoruje plik i powróci do datasetu CSV.

```bash
poetry run python scripts/capture_stream_snapshot.py \
    --base-url http://127.0.0.1:9000 \
    --path /stream \
    --adapter binance \
    --scope public \
    --environment paper \
    --channels ohlcv \
    --output data/snapshots/binance_ohlcv.json
```

## Konfiguracja aplikacji desktopowej
1. W panelu ustawień wskaż ścieżkę do snapshotu (`Ustawienia → Dane → Snapshot strumieniowy`).
2. Włącz przełącznik „Strumieniowanie offline”. Aplikacja zaloguje w `bot.shell.offline.service`, że snapshot został załadowany wraz z liczbą świec.
3. W razie błędu (np. brak pól OHLC lub niesortowana sekwencja) log zawiera komunikat diagnostyczny, a system wraca do lokalnego datasetu CSV.

## Tryb mieszany (historia → live)
* Snapshot stanowi rozgrzewkę dla strategii. Po uruchomieniu strumienia runtime oczekuje, że kolejne dane pojawią się już w kolejności rosnącej.
* Jeżeli aplikacja przełączy się z datasetu CSV na snapshot, otrzymasz wpis: `Załadowano snapshot strumieniowy <ścieżka> (N świec)`.
* Przełączenie strumieniowania on/off również generuje wpis informacyjny, co pomaga diagnozować konfigurację w środowisku testowym.

## Obsługa błędów
* **Niesortowane dane:** log `zawiera dane poza kolejnością`. Snapshot nie zostanie użyty.
* **Niepoprawne wartości liczbowej świecy:** wpis ostrzegawczy, a rekord zostanie pominięty.
* **Brak pliku:** log ostrzegawczy i powrót do datasetu CSV.

Stany te można monitorować w panelu „Monitoring → Runtime offline”, który subskrybuje sygnały `connectionStateChanged` oraz `historyReady`.
