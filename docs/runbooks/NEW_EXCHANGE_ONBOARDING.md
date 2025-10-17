# Runbook: Onboarding nowej giełdy do bot-core

## Cel
Ten runbook prowadzi operatorów oraz deweloperów przez etapy dodawania nowej giełdy do platformy bot-core, od walidacji wymagań API po uruchomienie w środowisku produkcyjnym.

## 1. Analiza wstępna
1. Zweryfikuj dostępność dokumentacji REST/Websocket oraz wymagania dotyczące podpisów.
2. Sprawdź politykę rate limitów, dostępność kont testowych oraz wymagane IP allowlisty.
3. Ustal minimalny zestaw uprawnień API i dopasuj go do istniejących `permission_profiles` lub zaplanuj utworzenie nowego profilu.

## 2. Implementacja adaptera
1. Utwórz moduł w `bot_core/exchanges/<nazwa>` dziedziczący z `ExchangeAdapter`.
2. Zaimplementuj mapowanie symboli w osobnym module `symbols.py` wraz z testami round-trip.
3. Zdefiniuj reguły limitów (waga, okno, maksymalna liczba żądań) i udostępnij metody pomocnicze (`rate_limit_rule`, `request_weight`).
4. Zapewnij funkcję podpisującą żądania HMAC z deterministycznym kanonicznym ładunkiem.
5. Dodaj adapter do `bot_core/exchanges/__init__.py` oraz `_DEFAULT_ADAPTERS` w `bot_core/runtime/bootstrap.py`.

## 3. Konfiguracja i sekrety
1. Dodaj środowiska (`paper`, `live`) w `config/core.yaml`, wskazując odpowiedni `permission_profile`.
2. Upewnij się, że `SecretManager` posiada wpisy `key_id`, `secret` i opcjonalnie `passphrase` dla nowej giełdy.
3. Zaktualizuj dokumentację `.env`/`secrets` o nowe klucze przestrzeni nazw.

## 4. Testy i weryfikacja
1. Uzupełnij testy jednostkowe w `tests/exchanges/test_<nazwa>_adapter.py` weryfikujące podpisy i limity rate.
2. Rozszerz `tests/test_runtime_bootstrap.py` o przypadek uruchomienia środowiska z `permission_profile` nowej giełdy.
3. Uruchom `pytest tests/exchanges/test_<nazwa>_adapter.py tests/test_runtime_bootstrap.py`.

## 5. Observability i rollout
1. Skonfiguruj kanały alertowe oraz UI alert sink dla nowej giełdy.
2. Dodaj integrację do runbooków operacyjnych (health-checki, procedury awaryjne).
3. Przeprowadź smoke testy w środowisku paper i zaplanuj stopniowe wdrożenie produkcyjne.

## 6. Checklist przed produkcją
- [ ] Klucze API posiadają minimalne wymagane uprawnienia.
- [ ] Zdefiniowane i udokumentowane limity rate (automatyczne throttlowanie).
- [ ] Monitorowanie metryk (latencja, błędy podpisów) włączone w Grafanie.
- [ ] Runbook aktualny i przekazany do zespołu NOC.

> **Uwaga:** Utrzymuj runbook w repozytorium wraz z kodem, aktualizując go przy każdej zmianie API giełdy.
