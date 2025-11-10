# Stage6 – Portfolio Stress

Nowy moduł `portfolio_stress` rozszerza pipeline ryzyka Stage6 o scenariusze
makroekonomiczne oceniane na poziomie całego portfela. Raport zawiera kluczowe
metryki P&L, drawdown, wpływ na płynność i najgorsze pozycje, a wynik trafia do
cykli hypercare PortfolioGovernora.

## Dane wejściowe

* **Baseline portfela** – snapshot wartości pozycji i ekspozycji czynnikowych.
  * Domyślna lokalizacja: `stage6_samples/portfolio_stress_baseline.json`.
  * Pola wymagane: `portfolio_id`, `cash_usd`, lista `positions` z
    `value_usd` i `factor_betas`.
* **Konfiguracja scenariuszy** – sekcja `portfolio_stress.scenarios` w
  `config/core.yaml`.
  * Każdy scenariusz zawiera wstrząsy czynnikowe (`factors`) oraz opcjonalne
    szoki per-aktywo (`assets`).
  * Parametry: `horizon_days`, `probability`, `cash_return_pct`, tagi i
    metadane opisowe (np. źródło kalibracji).

## Aktualny zestaw scenariuszy (2024)

| Nazwa | Opis | Horyzont | Kluczowe czynniki |
| ----- | ---- | -------- | ----------------- |
| `usd_liquidity_crunch` | Kryzys płynności USD i ujemne fundingi | 5 dni | `usd_liquidity`, `funding_rates` |
| `alt_season_boom_and_bust` | Dwufazowa hossa/korekta altów | 10 dni | `alt_beta`, `alt_liquidity` |
| `rates_regime_shift` | Szok stóp procentowych i odpływ z ryzykownych aktywów | 7 dni | `global_rates`, `usd_liquidity`, `vol_targeting` |

Każdy scenariusz posiada dopasowaną sekcję `assets` dla instrumentów o najwyższej
ekspozycji (BTC, ETH, SOL, ADA, DOT, BNB) oraz parametry `liquidity_haircut_pct`
wyrażające dodatkowe wymogi płynności.

## Kalibracja

1. **Ekspozycje czynnikowe** – aktualizuj plik baseline na podstawie raportu
   risk/analytics Stage6. Zwracamy uwagę na sumę wag oraz zgodność z `portfolio_id`.
2. **Intensywność scenariuszy** – `return_pct` to procentowa zmiana wartości
   (np. -0.18 = -18%). `liquidity_haircut_pct` reprezentuje dodatkową utratę
   płynności (np. 0.35 = 35% haircut).
3. **Walidacja** – uruchom `scripts/run_portfolio_stress.py --scenario …` i
   zweryfikuj raport JSON/CSV. W razie potrzeby zaktualizuj pola `metadata`
   (np. źródło kalibracji, data warsztatów).

## Integracja operacyjna

* CLI: `python scripts/run_portfolio_stress.py --output-json var/audit/stage6/portfolio_stress/report.json --output-csv …`
* Raport automatycznie trafia do cyklu hypercare (`PortfolioHypercareCycle`)
  i jest uwzględniany w runbookach Stage6.
* Podpis HMAC wykorzystuje klucz z `portfolio_stress.signing_key_*`.
* Sekcja `summary` w raporcie zawiera skrót agregatów (liczba scenariuszy,
  maksymalny drawdown, najgorszy scenariusz, zagregowane P&L ważone
  prawdopodobieństwem) oraz kwantylowe metryki VaR95/CVaR95 w ujęciu
  procentowym i dolarowym. Hypercare korzysta z tych pól przy prezentacji
  ryzyka. Od wersji 1 raportu dostępna jest również lista `tag_aggregates`
  grupująca scenariusze po tagach tematycznych (np. `liquidity`, `macro`).
  Dla każdej kategorii raport prezentuje liczbę scenariuszy, łączną wagę
  prawdopodobieństwa, oczekiwany P&L oraz najgorszy scenariusz w danym
  koszyku, co ułatwia priorytetyzację działań mitigacyjnych.

## Harmonogram aktualizacji

* **Co tydzień** – aktualizacja baseline (migracja wag / nowe pozycje).
* **Po warsztatach ryzyka** – rewizja parametrów `return_pct` i `liquidity_haircut_pct`.
* **Po incydentach rynkowych** – dodanie dodatkowych scenariuszy (np. stress
  związany z giełdą lub stablecoinami).
