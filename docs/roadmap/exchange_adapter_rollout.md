# Roadmap: KuCoin, Huobi i Gemini – adaptery, testnety i symulatory paper

## Cel inicjatywy
Zapewnić pełny parytet funkcjonalny adapterów KuCoin, Huobi (HTX) i Gemini pomiędzy środowiskami live/testnet/paper oraz
udokumentować wymagania integracyjne (API, limity, testnet), aby zespół mógł prowadzić rollout w sposób kontrolowany.

## Oś czasu i kamienie milowe
| Tydzień | Deliverable | Kluczowe zależności |
| --- | --- | --- |
| 1 | Audyt API + podpisanie wymagań dostępowe (KYC, whitelisty IP) dla trzech giełd. | Dostęp do kont testnet/sandbox, Security Ops |
| 2 | Kontrakt testów failover + parametry symulatorów paper (margin vs spot). | QA, zespół resilience |
| 3–4 | Implementacja adapterów (CCXT fallback, retry policy, limity) + aktualizacja konfiguracji marketplace. | DevOps (sekrety), dokumentacja |
| 5 | Testy end-to-end: failover managera, smoke testy paper simulators, eksport raportów benchmarkowych. | QA automation, Observability |
| 6 | Rollout produkcyjny (feature flag), publikacja roadmapy i raportu benchmarkowego. | Marketing, Support |

---

## Analiza API i testnet – KuCoin
- **Endpointy główne:** `https://api.kucoin.com` (live), `https://openapi-sandbox.kucoin.com` (sandbox). REST + WebSocket;
  CCXT wymaga `options.defaultType="spot"` dla adaptera spot/margin.
- **Autoryzacja:** klucz API + secret + passphrase; podpis HMAC SHA256 z `KC-API-PASSPHRASE`. Sandbox wymaga osobnego
  zestawu kluczy.
- **Limity:** 30 req/3 s (REST private), 1800 req/min; dodatkowo rate limit WebSocket 10 subskrypcji / 2 s.
- **Testnet:** sandbox dla margin/spot; wymaga włączenia `set_sandbox_mode(true)` w CCXT i whitelisting IP.
- **Retry policy:** najlepiej 4 próby, `base_delay=0.2`, `max_delay=2.0`, jitter 100–300 ms.
- **Failover:** fallback CCXT w ExchangeManagerze po pierwszym błędzie sieciowym; cooldown 30 s.
- **Paper simulator:** wariant margin (leverage limit 3.0, maintenance margin 12%, funding 0.02%/4h) do zachowania
  parytetu z margin live.

## Analiza API i testnet – Huobi (HTX)
- **Endpointy:** `https://api.huobi.pro` (live), `https://api.testnet.huobi.pro` (testnet REST). Wymagany parametr `type=spot`
  dla OHLCV/cancel order.
- **Autoryzacja:** klucz/secret z podpisem HMAC SHA256, timestamp w UTC; testnet aktywowany osobno.
- **Limity:** 90 req/3 s dla endpointów REST, 900 req/min globalnie; WebSocket 50 subskrypcji kanałów.
- **Retry policy:** 4 próby, `base_delay=0.15`, `max_delay=1.0`, jitter 50–200 ms.
- **Failover:** przełączenie na CCXT po pierwszym błędzie sieciowym; logowanie throttlingu w metrics `exchange_ccxt_failures`.
- **Paper simulator:** wariant spot z możliwością konfiguracji fee + snapshotu margin (PaperMarginSimulator z domyślnym
  maintenance 15%).

## Analiza API i testnet – Gemini
- **Endpointy:** `https://api.gemini.com` (live), `https://api.sandbox.gemini.com` (sandbox). Konto `primary` dla CCXT.
- **Autoryzacja:** klucz/secret Base64 + payload JSON podpisany HMAC SHA384; sandbox tworzy osobne klucze.
- **Limity:** 15 req/s (burst) oraz 1200 req/min; WebSocket 20 kanałów na połączenie.
- **Retry policy:** 5 prób, `base_delay=0.25`, `max_delay=2.5`, jitter 100–400 ms.
- **Failover:** włączony CCXT fallback w ExchangeManagerze, log telemetry `failover_count` + status `active_backend`.
- **Paper simulator:** wariant spot z możliwością zmiany waluty bazowej (USD), fee rate 0.35% jako start.

---

## Matryca wymagań i QA
| Obszar | KuCoin | Huobi | Gemini |
| --- | --- | --- | --- |
| Sandbox/Testnet | Wymaga `set_sandbox_mode` + osobne klucze i IP whitelist. | `testnet=true` w konfiguracji + parametry `type=spot`. | Sandbox z kontem `primary`; brak whitelisty, ale rate-limit osobny. |
| Failover | Cooldown 30 s, próg 1 błąd sieciowy, fallback CCXT. | Cooldown 45 s, próg 2 błędy throttlingu. | Cooldown 60 s, próg 1 błąd sieci lub auth. |
| Paper | Margin simulator (3x leverage). | Spot simulator + margin snapshot. | Spot simulator, fee rate 0.35%. |
| Raporty | Marketplace preset + benchmark update. | Marketplace preset + benchmark update. | Marketplace preset + benchmark update. |

## Testy i automaty
1. **Unit:** `tests/exchanges/test_ccxt_adapters.py` – asercje na retry policy, sandbox i limity.
2. **Paper simulators:** `tests/exchanges/test_paper_simulators.py` – snapshot equity, funding events, konfiguracja.
3. **Failover:** `tests/integration/test_exchange_manager_failover.py` – fallback CCXT; dodatkowo planowane smoke`i w QA.
4. **Marketplace:** walidacja presetów (`python scripts/check_marketplace_presets.py --exchanges kucoin,huobi,gemini`).
5. **Benchmark:** aktualizacja `docs/benchmark/cryptohopper_comparison.md` + wpis w kronice.

## Ryzyka i mitigacje
- Brak dostępu do sandboxów → fallback PaperMarginSimulator + rejestr capture z CCXT stubów.
- Zmiany regulacyjne (KYC) → checklista z compliance + rotacja kluczy co kwartał.
- Zbyt agresywne limity → dynamiczny limiter i alerty w Grafanie (`exchange_ccxt_failures_total`).

## Deliverables
- Zaktualizowane moduły adapterów (`bot_core/exchanges/kucoin|huobi|gemini`).
- Testy jednostkowe + smoke failover.
- Marketplace presets (`config/marketplace/presets/exchanges/exchange_*.json`) oraz konfiguracje YAML.
- Raport benchmarkowy Stage6 z nową sekcją „Obsługa wielu giełd (KuCoin/Huobi/Gemini)”.
