# W pełni autonomiczny przepływ AutoTradera

Nowy menedżer cyklu życia pozwala uruchomić AutoTradera w trybie 24/7 bez
interwencji operatora. Składa się on z dwóch elementów:

* `AutoTraderDecisionScheduler` — lekki harmonogram uruchamiający cykle decyzyjne
  w wątku taustowym lub wewnątrz pętli `asyncio`.
* `AutoTraderLifecycleManager` — warstwa koordynująca bootstrap, przełączanie
  reżimów, integrację guardrails oraz ponowne uruchomienia po awarii.

## Bootstrap z wykorzystaniem dzienników decyzji

Menedżer cyklu życia podczas startu odczytuje ostatnie wpisy z dziennika decyzji
i przywraca profil ryzyka oraz metadane reżimu. Dzięki temu AutoTrader kontynuuje
pracę dokładnie od stanu zapisanego w poprzedniej sesji. Wszystkie operacje są
audytowane w `DecisionAuditLog`, a w dzienniku decyzji pojawia się zdarzenie
`scheduler_bootstrap` z metadanymi środowiska.

```python
from bot_core.auto_trader import (
    AutoTrader,
    AutoTraderDecisionScheduler,
    AutoTraderLifecycleManager,
    DecisionAuditLog,
)
from bot_core.runtime.journal import InMemoryTradingDecisionJournal

journal = InMemoryTradingDecisionJournal()
audit_log = DecisionAuditLog()
trader = AutoTrader(
    emitter,
    gui,
    lambda: "BTCUSDT",
    decision_journal=journal,
    decision_audit_log=audit_log,
    enable_auto_trade=True,
)
scheduler = AutoTraderDecisionScheduler(trader, interval_s=5.0)
lifecycle = AutoTraderLifecycleManager(trader, scheduler=scheduler)
lifecycle.start()
```

## Automatyczna aprobata cykli i guardrails

`AutoTraderLifecycleManager` zdejmuje konieczność ręcznego `activate()` –
samodzielnie potwierdza auto-trade podczas bootstrapa i monitoruje wyniki cykli.
Jeżeli w szczegółach decyzji pojawią się powody guardrail (`guardrail_reasons`),
menedżer generuje alert `auto_trader.guardrail` wraz z kontekstem i wpisem do
dziennika audytowego. Pozwala to szybko zareagować na degradację jakości danych
lub blokady ryzyka.

## Restart po awarii

Scheduler i menedżer obsługują eskalujący backoff (1s, 2s, 4s, …) pomiędzy
kolejnymi próbami po nieudanym cyklu. Każda awaria jest raportowana jako
`scheduler_failure` w dzienniku decyzji wraz z liczbą prób i czasem kolejnego
uruchomienia. Dzięki temu kontroler można uruchomić raz i zostawić bez opieki.

## Monitorowanie 24/7

W połączeniu z `InMemoryTradingDecisionJournal` (lub wariantem zapisującym na
dysk) oraz `DecisionAuditLog` uzyskujemy pełną ścieżkę audytu:

1. feed danych z rynku,
2. ocena reżimu i ewentualna zmiana profilu ryzyka,
3. decyzja handlowa (z guardrailami),
4. egzekucja przez `ExecutionService`,
5. automatyczny wpis audytowy i alert.

Wdrożenie wymaga jedynie skonfigurowania scheduler-a i menedżera cyklu życia —
pozostałe elementy AutoTradera (journal, audit, alerty) podłączają się
automatycznie poprzez bootstrap.

## AI Governor – entrypoint manualny i launch-and-forget

`AutoTraderAIGovernorRunner` kapsułkuje `DecisionOrchestrator`, dzięki czemu
możemy zasilać UI oraz raporty demo bez grzebania w polach prywatnych
orkiestratora. Runner udostępnia dwa scenariusze pracy:

1. `run_cycle()` – pojedynczy krok wykonywany ręcznie (np. w trakcie testów,
   warsztatów lub budowania materiałów marketingowych). Metoda sama pobiera
   `MarketRegimeAssessment`, telemetry ryzyka/cyklu i szacowany koszt decyzji,
   po czym aktualizuje historię AI Governora.
2. `run_until(mode=..., limit=...)` – tryb launch-and-forget, który pętli cykle
   aż do osiągnięcia wskazanego trybu (scalping/hedge/grid) lub spełnienia
   predykatu. Z tego trybu korzysta m.in. `scripts/run_local_bot.py`, dzięki
   czemu snapshot `ai_governor` trafia do komunikatu `ready` oraz raportu E2E.

`RuntimeService.reloadAiGovernorSnapshot` używa tego samego runnera w trybie
demo, więc fallback QML nie musi już ręcznie manipulować schedulerem ani
bezpośrednio dotykać instancji AutoTradera. `scripts/run_local_bot.py` po
zbudowaniu pipeline papierowego wykonuje sekwencję `run_until()` dla
scalping/hedge/grid na tym samym orchestratorze, a snapshot trafia do eventu
`ready`, dzięki czemu operator widzi w UI faktyczny stan rekomendacji jeszcze
przed uruchomieniem pierwszych cykli decyzyjnych.
