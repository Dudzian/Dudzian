# Telemetria anonimowa

Dokument opisuje sposób działania modułu `core.telemetry.anonymous_collector` oraz zasady
zbierania zanonimizowanych danych użycia. Funkcja telemetrii jest **domyślnie wyłączona** i wymaga
świadomej zgody użytkownika.

## Zakres danych

System gromadzi wyłącznie metadane niepozwalające na identyfikację konkretnej osoby:

- znaczniki czasu zdarzeń (UTC),
- identyfikator instalacji generowany lokalnie (`installation_id`),
- pseudonim obliczony z wykorzystaniem soli i opcjonalnego fingerprintu sprzętowego,
- typ zdarzenia (np. `runtime.start`, `strategy.switch`),
- niewielkie payloady konfiguracyjne (wersje aplikacji, nazwy strategii, klasy błędów),
- informacje diagnostyczne dotyczące działania aplikacji (np. liczba błędów guardrail'i).

Żadne klucze API, nazwy kont czy inne dane wrażliwe nie są dodawane do kolejki telemetrii.
Wszystkie wartości są sanitizowane do typów prostych (napisy, liczby, wartości logiczne).

## Opt-in i opt-out

1. Użytkownik włącza telemetrię w zakładce **Ustawienia → Prywatność**.
2. Po aktywacji aplikacja generuje pseudonim (`sha256(fingerprint|salt)`), który może być
   dodatkowo powiązany z fingerprintem sprzętowym (pole opcjonalne w UI).
3. Użytkownik może wyłączyć telemetrię w dowolnym momencie – kolejne zdarzenia nie są wówczas
   rejestrowane.
4. Przycisk „Wyczyść kolejkę” usuwa wszystkie zebrane dane z dysku.

## Buforowanie i eksport

- Zdarzenia są zapisywane lokalnie w pliku `~/.kryptolowca/telemetry/queue.jsonl`.
- Kolejka jest ograniczona do 1000 wpisów; najstarsze elementy są usuwane po przekroczeniu limitu.
- Eksport danych tworzy plik `telemetry_YYYYmmddTHHMMSS.json` w katalogu `exports/`, zawierający
  zestawienie zdarzeń wraz z pseudonimem i identyfikatorem instalacji.
- Eksport jest wykonywany wyłącznie na żądanie użytkownika – brak połączeń sieciowych.

## Procedury bezpieczeństwa

- Pliki są przechowywane w katalogu domowym użytkownika i mogą zostać ręcznie usunięte.
- Pseudonimizacja wykorzystuje losową sól zapisaną lokalnie; zmiana soli spowoduje wygenerowanie
  nowego pseudonimu.
- Logi operacji telemetrii trafiają do ogólnego systemu logowania UI (jeżeli jest skonfigurowany).
- Wersjonowanie ustawień w `settings.json` umożliwia migrację konfiguracji bez naruszania zgody.

## Integracja z UI

Widok `PrivacySettings.qml` prezentuje:

- aktualny status zgody i pseudonim,
- licznik zdarzeń w kolejce i ścieżkę do pliku `queue.jsonl`,
- podgląd pierwszych zdarzeń w formacie JSON,
- akcje: eksport, czyszczenie kolejki, odświeżenie pseudonimu.

Testy `tests/ui/test_privacy_settings.py` zapewniają podstawową regresję ładowania komponentu oraz
integracji z kontrolerem `PrivacySettingsController`.
