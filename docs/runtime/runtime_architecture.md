# Architektura runtime

## Asynchroniczny dispatcher operacji I/O

W warstwie runtime wprowadzono komponent `AsyncIOTaskQueue`, który zarządza równoległym
wykonywaniem zapytań sieciowych dla adapterów giełdowych. Dispatcher działa jako kolejka
per giełda, egzekwując dwa typy limitów:

* **`max_concurrency`** – maksymalna liczba równolegle wykonywanych operacji dla danego klucza.
* **`burst`** – maksymalna liczba jednocześnie oczekujących zadań, co zapobiega lawinowemu
  odkładaniu requestów w przypadku spowolnienia API.

Kolejka jest tworzona podczas budowania `MultiStrategyScheduler` i przekazywana do runtime.
Scheduler wykorzystuje dispatcher przy pobieraniu historii i najnowszych danych ze źródeł
strategii (np. REST-owych feedów OHLCV), dzięki czemu limity są respektowane niezależnie od
liczby aktywnych harmonogramów.

Konfiguracja znajduje się w pliku `config/runtime.yaml` w sekcji `io_queue`:

```yaml
io_queue:
  max_concurrency: 6
  burst: 16
  exchanges:
    binance_spot:
      max_concurrency: 4
      burst: 8
    nowa_gielda_spot:
      max_concurrency: 2
      burst: 6
```

Wartości domyślne (`max_concurrency=8`, `burst=16`) można nadpisać globalnie, a następnie
precyzyjnie dopasować dla poszczególnych giełd. Dodanie nowych adapterów wymaga jedynie
uzupełnienia wpisu w `exchanges` lub skorzystania z limitów globalnych.

## Weryfikacja podpisów dokumentów LIVE

Procedura `bootstrap_environment` blokuje uruchomienie środowiska `Environment.LIVE`, dopóki
nie zweryfikuje podpisów HMAC dla wszystkich wymaganych dokumentów:

- **Compliance/KYC** – każdy dokument podpisany przez zespół compliance (np. pakiet KYC)
  musi posiadać ważny podpis HMAC; klucz oczekiwany jest w `secrets/hmac/compliance*.key`.
- **Ryzyko** – raporty limitów lub alignmentu profili ryzyka podpisane przez zespół risk
  muszą być zweryfikowane tym samym mechanizmem (`secrets/hmac/risk*.key`).
- **Penetration test** – raport z testów penetracyjnych (dokument z nazwą zawierającą
  `penetration` lub `pentest`) wymaga podpisu HMAC i klucza w `secrets/hmac/`.

Podczas bootstrapa moduł oblicza sumy SHA-256 plików, wczytuje metadane podpisu, a następnie
weryfikuje podpis za pomocą `bot_core.security.signing.verify_hmac_signature`. Brak dokumentu,
klucza lub niepoprawny podpis powodują `RuntimeError`, dzięki czemu tryb live nie zostanie
uruchomiony w stanie niespełniającym wymogów compliance.

Informacje o rezultacie weryfikacji (ścieżki, identyfikatory kluczy, kategorie) są przechowywane
w `BootstrapContext.live_signature_verification` i mogą być wyświetlane w UI lub raportach
operacyjnych.
