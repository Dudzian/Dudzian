# Limity ryzyka i polityka kontroli ekspozycji

Silnik `ThresholdRiskEngine` udostępnia spójny zestaw limitów zgodny z polityką
ryzyka obowiązującą w Stage6.  Poniżej opisano najważniejsze mechanizmy
konfiguracji oraz logikę egzekwowania ograniczeń.

## Limit ryzyka na pojedynczą transakcję

* Profil wymusza budżet ryzyka na poziomie 0,5–1% kapitału w zależności od
  profilu (`trade_risk_pct_range`).
* Limit wyliczany jest na podstawie odległości stop loss (`stop_price`) od ceny
  wejścia.  Ryzyko dla przyrostu pozycji (`stop_distance * delta_quantity`) nie
  może przekroczyć górnego progu, a wartości znacząco poniżej dolnego progu
  generują alert (`trade_risk_floor`).
* Naruszenie limitu kończy się odrzuceniem zlecenia z sugestią maksymalnej
  wielkości (`max_quantity`).

## Limity ekspozycji na instrument i portfel

* Dla pojedynczego instrumentu obowiązuje alert przy 25–27% oraz twardy limit
  30% kapitału (`instrument_alert_pct`, `instrument_limit_pct`).
* Portfel raportuje alert przy 50–60% oraz blokuje transakcje po przekroczeniu
  60–70% kapitału (`portfolio_alert_pct`, `portfolio_limit_pct`).
* Po każdym przekroczeniu progów alertowych do rejestru alertów (`RiskAlertLog`)
  trafia wpis umożliwiający audyt (limity `instrument_exposure_alert` oraz
  `portfolio_exposure_alert`).

## Kill-switch dzienny i tygodniowy

* Dzienny kill-switch aktywuje się po stracie równej `2R` lub −4% kapitału
  (domyślnie `daily_kill_switch_r_multiple=2`, `daily_kill_switch_loss_pct=0.04`).
* Tygodniowy kill-switch blokuje handel przy stracie między −6% a −8% w zależności
  od profilu (`weekly_kill_switch_loss_pct`).
* Aktywacja kill-switcha ustawia profil w trybie awaryjnym – dozwolone są tylko
  transakcje redukujące ekspozycję.  Każda aktywacja rejestrowana jest w
  `RiskAlertLog` z limitem `kill_switch`.

## Limit kosztów transakcyjnych (30 dni)

* W oknie kroczącym 30 dni koszty transakcyjne nie mogą przekroczyć 25% zysku
  (`max_cost_to_profit_ratio`).
* Przekroczenie progu powoduje natychmiastowe przejście w tryb awaryjny oraz
  alert `cost_to_profit_ratio`.
* Aktualne wartości `rolling_profit_30d` i `rolling_costs_30d` należy przekazywać
  przez `on_mark_to_market(..., metadata={...})` lub w strumieniu `ACCOUNT_MARK`
  (`rolling_profit_30d`/`rollingProfit30d`, `rolling_costs_30d`/`rollingCosts30d`).

## Konfiguracja profili

Parametry dostępne są w klasach profili (`bot_core/risk/profiles/*`).  Każdy
profil udostępnia dodatkowe metody:

* `trade_risk_pct_range()` – zakres ryzyka na transakcję,
* `instrument_alert_pct()` / `instrument_limit_pct()` – progi instrumentu,
* `portfolio_alert_pct()` / `portfolio_limit_pct()` – progi portfela,
* `daily_kill_switch_r_multiple()` / `daily_kill_switch_loss_pct()` – progi kill
  switcha dziennego,
* `weekly_kill_switch_loss_pct()` – tygodniowy kill-switch,
* `max_cost_to_profit_ratio()` – limit kosztów w relacji do zysku.

Zmiany wartości można wprowadzić w plikach profili lub poprzez profil `ManualProfile`
(własne parametry przekazywane w konstruktorze).

## Alerty i audyt

Alerty generowane przez silnik ryzyka są dostępne poprzez metodę
`ThresholdRiskEngine.recent_alerts()` oraz klasę `RiskAlertLog`.  Każdy wpis
zawiera nazwę profilu, rodzaj limitu, wartość ekspozycji oraz kontekst
(aktywny symbol, PnL, udział kosztów).  Umożliwia to szybki audyt naruszeń i
monitoring przestrzegania polityki ryzyka.
