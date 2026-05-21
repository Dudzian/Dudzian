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

6. **Cleanup po testach/audycie**

   ```bash
   rm -rf bot_core/logs
   git checkout -- reports/ci/decision_feed_metrics.json || true
   git checkout -- reports/exchanges/signal_quality || true
   rm -f trading.db
   git clean -fd -- reports/exchanges/signal_quality || true
   rm -f test-results/qml/timeline.ndjson
   ```
