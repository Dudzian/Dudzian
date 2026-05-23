# Paper Labs – checklista symulacji profili ryzyka

## Cel
Zapewnić, że przed przejściem z trybu paper do live wszystkie profile ryzyka (konserwatywny, zbalansowany, agresywny, manualny) zostały
zweryfikowane w ramach scenariuszy baseline oraz stres testów flash crash / dry liquidity / latency spike.

## Wymagane wejścia
- [ ] Zatwierdzona wersja konfiguracji `config/core.yaml` (commit SHA / tag).
- [ ] Dostęp do bundla danych Parquet lub włączenie trybu syntetycznego (`--synthetic-fallback`).
- [ ] Klucz HMAC do podpisu raportów (jeśli raport trafia do rejestru compliance).
- [ ] Ścieżka wyjściowa na artefakty (`var/paper_labs/<data>` lub dedykowany katalog CI).

## Kroki
1. [ ] Uruchom komendę:

       ```bash
       python scripts/run_risk_simulation_lab.py \
         --config config/core.yaml \
         --output-dir reports/paper_labs \
         --environment <env> \
         --symbols <lista_symboli> \
         --fail-on-breach
       ```

       gdzie `--config` domyślnie wskazuje `config/core.yaml`, a `--output-dir` – katalog `reports/paper_labs`.
2. [ ] Zweryfikuj, że raport JSON zawiera wszystkie profile oraz pola `breach_count == 0`, `stress_failures == 0`.
3. [ ] Otwórz PDF i potwierdź brak naruszeń oraz poprawne podpisy sekcji stres testów.
4. [ ] Zabezpiecz artefakty: JSON, PDF, log CLI oraz (opcjonalnie) podpis HMAC → katalog audytowy.
5. [ ] Uaktualnij decision log (`audit/decisions`) wpisem z wynikiem Paper Labs (status PASS/FAIL, operator, timestamp).

## Artefakty / Akceptacja
| Artefakt | Lokalizacja | Odpowiedzialny | Akceptacja |
| --- | --- | --- | --- |
| `risk_simulation_report.json` | katalog wyjściowy CLI | Risk Lead | [ ]
| `risk_simulation_report.pdf` | katalog wyjściowy CLI | Risk Lead | [ ]
| Log z CLI (`run_risk_simulation_lab.log`) | katalog wyjściowy CLI | Ops | [ ]
| Wpis w decision log (`audit/decisions`) | repozytorium audytowe | Compliance | [ ]

## Uwagi operacyjne
- W trybie syntetycznym raport oznaczony jest flagą `synthetic_data: true`; akceptacja wymaga zgody Compliance i planu pozyskania
  pełnych danych historycznych.
- W przypadku wykrycia naruszeń (breach/stress failure) należy wygenerować task w systemie ticketowym i zablokować przejście do etapu live.
- Raporty powinny być przechowywane co najmniej 730 dni w repozytorium audytowym (zgodnie z runbookiem OEM Licensing).


## Safe local paper/demo startup recipe

Cel: uruchomić lokalny scenariusz paper/demo/offline bez realnych zleceń i bez realnych kluczy API.

1. **Przygotowanie środowiska**

   ```bash
   python -m pip install -U pip setuptools wheel
   python -m pip install -e ".[dev]"
   ```

2. **Walidacja bezpiecznego configu demo/paper**

   ```bash
   python -m pytest -q tests/docs/test_demo_paper_startup_contract.py -vv
   ```

3. **Preflight config safety + paper-only (bez uruchamiania live runtime)**

   ```bash
   python scripts/demo_paper_precheck.py --config config/e2e/demo_paper.yml --json
   python scripts/paper_adapter_readiness.py --config config/e2e/demo_paper.yml --json
   python scripts/run_local_bot.py --mode demo --config config/e2e/demo_paper.yml --preview-plan
   python scripts/paper_precheck.py --config config/core.yaml --environment binance --json
   ```

   > `config/e2e/demo_paper.yml` to bezpieczny profil e2e demo/paper (overlay), a nie pełny `CoreConfig`.
   > `paper_precheck.py` oczekuje `CoreConfig` i środowiska z `core.environments` (np. `binance`), więc `--config config/e2e/demo_paper.yml --environment binance_paper` nie jest poprawnym wejściem.

4. **Smoke i sanity**

   ```bash
   python -m pytest -q tests/test_trading_controller.py -k "paper or demo or dry or sandbox or mock or environment or runtime" -vv --maxfail=20
   python -m pytest -q tests/runtime -k "paper or demo or dry or sandbox or runtime or config" -vv --maxfail=20
   python -m pytest -q tests/ui -k "runtime or demo or feed or snapshot" -vv --maxfail=20
   python scripts/ci/run_autonomy_matrix.py
   python scripts/ci/run_risk_execution_matrix.py
   python scripts/ci/run_observability_matrix.py
   python scripts/ci/run_recovery_matrix.py
   ```

5. **Bezpieczne granice (zakazy)**

- Nie używaj realnych kluczy API i nie podawaj sekretów live.
- Nie uruchamiaj `--mode live`.
- Nie podawaj konfiguracji środowiska live ani produkcyjnych exchange credentials.
- Jeśli komenda może łączyć się z live exchange albo składać realne ordery, nie uruchamiaj jej bez jednoznacznego dry-run/paper guarda.

## Safe local runtime preview boundary

- Granica operatorska preview to **wyłącznie** komendy precheck/preview-plan:
  - `python scripts/demo_paper_precheck.py --config config/e2e/demo_paper.yml --json`
  - `python scripts/run_local_bot.py --mode demo --config config/e2e/demo_paper.yml --preview-plan`
- `--preview-plan` nie uruchamia runtime, nie łączy się z exchange i nie wykonuje zleceń.
- Preview boundary to dry-run/plan, **nie** live trading.
- Zakazy:
  - nie używać `--mode live`,
  - nie używać realnych API keys,
  - nie używać konfiguracji live exchange,
  - nie uruchamiać runtime bez wcześniejszego precheck + preview-plan.

6. **Cleanup po testach/audycie**

   ```bash
   rm -rf bot_core/logs
   git checkout -- reports/ci/decision_feed_metrics.json || true
   git checkout -- reports/exchanges/signal_quality || true
   rm -f trading.db
   git clean -fd -- reports/exchanges/signal_quality || true
   rm -f test-results/qml/timeline.ndjson
   ```


## Controlled mock/offline runtime preview

1. Config safety

   ```bash
   python scripts/demo_paper_precheck.py --config config/e2e/demo_paper.yml --json
   ```

2. Preview plan safety gate

   ```bash
   python scripts/run_local_bot.py --mode demo --config config/e2e/demo_paper.yml --preview-plan
   ```

3. Bounded mock/offline runtime preview

   ```bash
   python scripts/mock_runtime_preview.py --mode demo --config config/e2e/demo_paper.yml --duration-seconds 5 --json
   ```

Restrictions:
- do not use `--mode live`,
- do not use real API keys,
- do not use live exchange config,
- do not execute real orders.

This command is still not live trading and not real exchange paper trading.


## Controller-backed mock preview

```bash
python scripts/controller_mock_preview.py --mode demo --config config/e2e/demo_paper.yml --max-signals 1 --json
```

This command executes a bounded controller-backed mock preview path (`TradingController/process_signals`) on synthetic signals.
JSON output includes a controller/mock execution outcome summary and safety invariants (`real_orders_submitted=false`).

Restrictions:
- no live exchange/API I/O,
- no API keys required,
- no real order submission,
- no production runtime loop startup.

This is still not live trading and not real exchange paper trading.



## Sandbox/testnet static readiness preflight

```bash
python scripts/sandbox_testnet_readiness.py --config config/e2e/demo_paper.yml --environment binance_paper --json
```

Static/config-only gate: no exchange/API I/O, no API keys, no secrets read, no order submission, no runtime loop.
This is not sandbox/testnet trading; it is only a readiness gate before future sandbox/testnet stages.

## Credential reference static readiness preflight

```bash
python scripts/credential_reference_readiness.py --config config/e2e/demo_paper.yml --environment binance_paper --json
```

Static/config-only guard: no secret reads, no keychain reads, no env secret value reads, no exchange/API I/O,
no API keys required, no order submission, no runtime loop. This does not prove real credentials are valid; it
only checks config safety and absence of inline secret values before future sandbox/testnet work.

## One-command operator preview bundle

```bash
python scripts/operator_preview_bundle.py --mode demo --config config/e2e/demo_paper.yml --duration-seconds 5 --max-signals 1 --json
```

This command runs the full safe operator preview package chain (precheck -> paper adapter readiness -> sandbox/testnet static readiness -> credential reference static readiness -> preview-plan -> mock runtime preview -> controller-backed mock preview).
Bundle includes a paper adapter readiness preflight before preview-plan/mock/controller steps.
Bundle includes sandbox/testnet static readiness after paper adapter readiness and before preview-plan/mock/controller steps.
Bundle includes credential reference static readiness after sandbox/testnet static readiness and before preview-plan/mock/controller steps.
It is not live trading, not real exchange paper trading, does not use API keys, does not submit real orders, and blocks live mode.


Paper adapter readiness preflight to statyczna walidacja kontraktu (no exchange I/O, no orders, no API keys); to nie jest start real paper runtime ani live trading.
Mock preview -> paper adapter readiness -> dopiero później paper/sandbox runtime smoke.

## Bounded paper runtime dry-run preflight

```bash
python scripts/paper_runtime_dry_run.py --mode demo --config config/e2e/demo_paper.yml --duration-seconds 5 --max-signals 1 --json
```

Bounded wrapper runs preview-plan + mock runtime preview + controller mock preview.
No live mode, no real API keys, no secret/keychain/env secret value reads, no exchange/API I/O,
no real order submission, no production runtime loop.
This is not real paper trading and not sandbox/testnet trading.
`paper_runtime_dry_run.py` is a bounded standalone command. Do not nest it into `operator_preview_bundle.py` unless shared `preview-plan` / `mock_runtime_preview` / `controller_mock_preview` steps are split first; otherwise those steps would run twice.

## Controlled paper runtime validation

```bash
python scripts/controlled_paper_runtime_validation.py --mode demo --config config/e2e/demo_paper.yml --duration-seconds 5 --max-signals 1 --json
```

Bounded validation wrapper runs preview-plan + mock runtime preview + controller mock preview and adds shutdown/thread/journal/event summary.
No live mode, no real API keys, no secret/keychain/env secret reads, no exchange/API I/O, no real order submission, no production runtime loop.
This is not real paper trading and not sandbox/testnet trading; journal visibility may be limited when mock preview does not expose journal export.
`controlled_paper_runtime_validation.py` can optionally persist the full JSON session report via `--report-path`, which is intended for comparing bounded validation runs before mini-soak.
