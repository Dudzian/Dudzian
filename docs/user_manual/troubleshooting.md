# Troubleshooting

## Licencja
### Problem: "License validation failed"
- Sprawdź, czy plik `secrets/license/license_token.json` istnieje i nie został zmodyfikowany.
- Uruchom ponownie kreator i wykonaj synchronizację czasu systemowego.
- Jeżeli zmienił się sprzęt, skontaktuj się z pomocą techniczną w celu regeneracji licencji.

### Problem: "Fingerprint mismatch"
- Upewnij się, że aplikacja działa na sprzęcie, dla którego została wydana licencja.
- Usuń plik cache `var/security/fingerprint_cache.json` i uruchom ponownie aplikację, aby odświeżyć fingerprint.

## Połączenia sieciowe
### Problem: Brak połączenia z giełdą
- Sprawdź status sieci i firewall (porty 80/443).
- Zweryfikuj, czy w kreatorze wprowadzono poprawne API key / secret / passphrase.
- Uruchom `scripts/run_local_bot.py --diag exchange=<nazwa>` w celu wykonania diagnostyki.

### Problem: Błędy limitów (rate limit)
- Skorzystaj z panelu monitoringu i sprawdź metrykę `exchange_rate_limit_usage`.
- Zmniejsz liczbę równoległych strategii lub zwiększ interwał pobierania danych.
- W razie potrzeby aktywuj tryb fallback (polling) dostępny w konfiguracji runtime.

## Giełdy i zlecenia
### Problem: Zlecenia pozostają w stanie "pending"
- Upewnij się, że zegar systemowy jest zsynchronizowany (NTP).
- Sprawdź panel dashboardu, sekcję aktywnych zleceń – jeżeli widnieje alert, wykonaj restart adaptera.
- W trybie papierowym sprawdź, czy wybrana giełda wspiera symulację danego instrumentu.

### Problem: Odrzucone zlecenia
- Zweryfikuj limity notional / wielkość minimalną instrumentu.
- Ustaw parametry strategii zgodne z wymaganiami giełdy (np. krok ceny, krok ilości).
- Włącz logowanie debug w `config/runtime.yaml` dla modułu execution, aby pozyskać szczegóły.

## UI i alerty
### Problem: Brak wykresów w dashboardzie
- Sprawdź, czy działa lokalny eksporter Prometheus (`http://localhost:9464/metrics`).
- Upewnij się, że runtime został uruchomiony i generuje metryki.
- W razie potrzeby odśwież widok dashboardu (przycisk „Odśwież dane”).

### Problem: Brak powiadomień toast
- Zweryfikuj, czy w preferencjach użytkownika włączona jest obsługa alertów.
- Sprawdź, czy `bot_core/alerts/dispatcher.py` nie jest w trybie "silent" (konfiguracja runtime).

## Marketplace
### Problem: Nieudany import presetu
- Upewnij się, że plik nie jest uszkodzony i zawiera podpis w sekcji `signature`.
- Sprawdź, czy certyfikat wydawcy znajduje się w `secrets/marketplace/trusted_publishers/`.
- W logach aplikacji znajdziesz szczegóły błędu walidacji.

### Problem: Konflikt wersji presetu
- Usuń lub zdezaktywuj starszą wersję presetu z Marketplace.
- Zmień alias presetu podczas importu, aby uniknąć kolizji nazw.

## Optymalizacja
### Problem: Zadanie optymalizacji nie startuje
- Sprawdź, czy w konfiguracji runtime ustawiono harmonogram optymalizacji.
- Upewnij się, że `var/data/` zawiera wymagane dane historyczne.

### Problem: Brak raportu po zakończeniu optymalizacji
- Sprawdź logi w `logs/optimization.log`.
- Upewnij się, że katalog `var/reports/optimization/` ma prawa zapisu.

## Dalsza pomoc
Jeśli powyższe kroki nie rozwiążą problemu, zgłoś incydent zgodnie z procedurą w `docs/support/plan.md`.
