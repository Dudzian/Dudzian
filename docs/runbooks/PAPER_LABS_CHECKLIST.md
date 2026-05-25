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
Current bounded mode is intentionally conservative and valid only within the script's configured duration guard (after CPR-35, explicitly allowing up to 259200 seconds / 72h for controlled validation).
A 5-minute / 300-second, 10-minute / 600-second, 30-minute / 1800-second, 60-minute / 3600-second, 24-hour / 86400-second and 72-hour / 259200-second controlled validation durations are allowed by guard eligibility.
After CPR-35, the child `mock_runtime_preview.py` duration guard is aligned to the same 259200-second upper bound used by `controlled_paper_runtime_validation.py`.
This guard update does not execute a 86400-second target run in-place; the 24h target run is handled in a separate stage.
CPR-35 raises guard eligibility to include 72h/259200, but does not execute a 72h target run in-place; the target run remains a separate stage.
Non-live boundary remains unchanged: no real API keys, no secret/keychain/env secret reads, no exchange/API I/O, no real order submission, no production runtime loop.
Before any 24h stage, keep the long-run health/resource/report guard enabled in controlled validation summary (`health_summary`, `process_resource_summary`, `progress_summary`, `artifact_summary`).
CPR-29 raises the duration guard to 86400 without executing the 24h target run; the actual 24h run is a separate stage.
CPR-27 adds a lightweight checkpoint/heartbeat/progress summary in controlled validation (`heartbeat_mode=step_boundary`) for deterministic step-boundary progress visibility.
After CPR-29 guard update, `long_run_ready=true` is expected when no blockers are present in bounded sanity.
24h controlled validation milestone is closed after two successful 86400s controlled validation runs (CPR-30 and CPR-31).
72h escalation remains a separate stage and requires resource health summary validation before any 259200 guard change.
Resource health summary is report-only and non-live; fields may be unavailable/null where safe cross-platform sampling is not available.
`long_run_ready` and `long_run_blockers` are tracked under `summary.health_summary.*`.
24h/72h still require separate duration patches/seals, and no-live/no-exchange/no-real-orders boundary remains unchanged.

## Installer fingerprint readiness contract (PACKAGING-READINESS-2)

Dostępna jest komenda local-only: `python scripts/installer_fingerprint_readiness.py --json`.
Kontrakt ma charakter bezpiecznego readiness check dla instalatora/first-run UX:
- nie aktywuje licencji,
- nie czyta sekretów/keychain/env values,
- nie wymaga API keys,
- nie wykonuje exchange/API I/O,
- nie uruchamia runtime loop,
- nie eksponuje raw machine identifiers (tylko ewentualny masked preview).



## Packaged config readiness (PACKAGING-READINESS-3)

Uruchom: `python scripts/packaged_config_readiness.py --config config/e2e/demo_paper.yml --json`.
Kontrakt jest static/config-only: nie czyta sekretów, keychain ani wartości env, nie wymaga API keys do instalacji, nie wykonuje exchange/API I/O i nie uruchamia runtime loop.
Bezpieczny domyślny tryb po instalacji to demo/paper/offline; onboarding credentiali jest oddzielony od sukcesu instalacji.
Packaging nie powinien bundle'ować: `.env`, lokalnej DB (`trading.db`), logów ani raportów.


## Security packaging readiness manifest (SECURITY-PACKAGING-2)

Uruchom: `python scripts/security_packaging_readiness.py --config config/e2e/demo_paper.yml --json`.
Manifest agreguje `installer_fingerprint_readiness` + `packaged_config_readiness` i zwraca machine-readable status sekcji: contracts, artifact hygiene, safe default launch, release integrity.
Komenda nie buduje EXE/installera, nie czyta sekretów/keychain/env values, nie wymaga API keys, nie wykonuje exchange/API I/O i nie uruchamia runtime loop.
Sekcje artifact hygiene i release integrity są readiness summary; pełne leak/exclude tests pozostają osobnym etapem.
