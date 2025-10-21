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
1. Dodaj środowiska (`paper`, `testnet`, `live`) w `config/core.yaml`, wskazując odpowiedni `permission_profile`.
2. Upewnij się, że `SecretManager` posiada wpisy `key_id`, `secret` i opcjonalnie `passphrase` dla nowej giełdy.
3. Zaktualizuj dokumentację `.env`/`secrets` o nowe klucze przestrzeni nazw.
4. Dla Bybit/OKX/Coinbase ustaw zmienne środowiskowe `BYBIT_ENVIRONMENT`, `OKX_ENVIRONMENT`, `COINBASE_ENVIRONMENT`
   – umożliwia to wymuszenie `paper`/`testnet` w health-checkach oraz w `ExchangeManager`.

### Profile Stage5 – Bybit, OKX, Coinbase

| Giełda  | Tryby natywne | Namespace sekretów | Wymagane dodatki |
| --- | --- | --- | --- |
| Bybit | `margin`, `futures`, `paper` | `secrets/exchanges/bybit_{paper,testnet,live}.json` | `hedgeMode` konfiguracja CCXT, sandbox HMAC |
| OKX | `margin`, `futures`, `paper` | `secrets/exchanges/okx_{paper,testnet,live}.json` | `instType` w parametrach REST, opcjonalne IP allowlist |
| Coinbase Advanced | `margin`, `futures`, `paper` | `secrets/exchanges/coinbase_{paper,testnet,live}.json` | `product_type` (margin/futures) oraz sandbox API |

* Wszystkie powyższe adaptery wspierają watchdog-i Stage5 (`WatchdogCCXTAdapter`).
* W trybie paper/testnet utrzymuj oddzielne klucze API oraz konta sub-account.
* Health-checki muszą obejmować publiczny ticker (`BTC/USDT`) oraz prywatne saldo USDT >= 50 (ustaw `private_min_balance`).

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
- [ ] Health-check Stage5 (`python -m bot_core.cli health-check --environment <env>`) przechodzi dla trybów paper/testnet/live.
- [ ] Runbook aktualny i przekazany do zespołu NOC.

> **Uwaga:** Utrzymuj runbook w repozytorium wraz z kodem, aktualizując go przy każdej zmianie API giełdy.
