# Granica nasłuchu plaintext gRPC

`LocalRuntimeServer`, `MarketDataServer`, `MetricsServer` i `RiskServer`
akceptują nieszyfrowany nasłuch wyłącznie na jawnych adresach loopback.
Wspólny helper `bot_core.runtime._grpc_support.validate_plaintext_grpc_host`
sprawdza host przed utworzeniem serwera i jego zasobów.

Dozwolone są kanoniczne adresy IPv4 z zakresu loopback oraz IPv6 `::1`.
Adresy są normalizowane, a IPv6 otrzymuje nawiasy wymagane w adresie gRPC.
Nazwa `localhost` jest zamieniana na stały adres `127.0.0.1`, bez rozwiązywania
DNS. Pozostałe nazwy DNS, adresy spoza loopback, wildcardy, listy adresów,
prefiksy resolverów, hosty z portem, scope IPv6 i IPv4-mapped IPv6 są odrzucane
przez `ValueError`. Nieprawidłowy typ lub pusty host również oznacza odmowę.

Zdalny nasłuch `MarketDataServer`, `MetricsServer` i `RiskServer` wymaga
skonfigurowanych `server_credentials`; dotychczasowa gałąź `add_secure_port`
pozostaje oddzielna. `LocalRuntimeServer` udostępnia wyłącznie lokalny plaintext.
Zmiana zawęża obsługiwane konfiguracje: zdalny plaintext oraz scoped IPv6 nie
są obsługiwane. Loopback nie zastępuje uwierzytelniania lokalnych klientów.

CLI `run_local_bot.py` stosuje ten sam guard przed utworzeniem stanu i kontekstu
runtime. Nieprawidłowy host kończy się kodem 2, również w trybie planu. Regresje
sprawdzają odmowę bez alokacji; rzeczywista próba loopback pozostaje oddzielna
od kwalifikacji TLS i środowiska produkcyjnego.

Regresje w `tests/security/test_grpc_bind_boundary.py` sprawdzają odmowę przed
alokacją dla wszystkich czterech konstruktorów, rzeczywiste efemeryczne
nasłuchy IPv4/IPv6 oraz normalizację hosta. Testy gałęzi TLS rejestrują wybór
bezpiecznego bindingu; nie wykonują handshake TLS. Jawny zakres CI obejmuje
te regresje i branch coverage pięciu zmienionych modułów; helper wchodzi do
skonfigurowanego mypy Linux/Windows. Dotychczasowe testy zachowano, a zakres
rozszerzono o nowe regresje. Progi, manualne bramki Stage 9/10 i kwalifikacja
fizycznego środowiska pozostają bez zmian.
