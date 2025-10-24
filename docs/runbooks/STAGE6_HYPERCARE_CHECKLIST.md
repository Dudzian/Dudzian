# STAGE6 – Checklist: Hypercare Orchestrator

## Cel
Wykonać kompletny cykl hypercare Stage6 (Observability + Resilience + Portfolio)
jednym poleceniem, uzyskując podpisany raport zbiorczy oraz powiązane
artefakty audytowe.

## Prerekwizyty
- Aktualne raporty wejściowe dla Observability (definicje/metyki SLO –
  repozytoryjny plik `config/observability/slo.yml` i plik metryk
  `var/metrics/stage6_measurements.json` wygenerowany według runbooka
  Observability; jeśli otrzymujesz metryki z innego środowiska, skopiuj je
  lub przekonwertuj do tej ścieżki – możesz użyć
  `python scripts/sync_stage6_metrics.py --source <plik> --output var/metrics/stage6_measurements.json`,
  który tworzy katalog docelowy, sprawdza poprawność JSON oraz liczbę
  pomiarów), Resilience (plan
  failover, manifesty paczek, polityka) oraz Portfolio
  (alokacje, Market Intel, raporty SLO/Stress Lab).
- Raport Market Intel wygenerowany do oczekiwanej lokalizacji hypercare:
  ```bash
  python scripts/build_market_intel_metrics.py \
    --environment binance_paper \
    --governor stage6_core \
    --output var/market_intel/stage6_core_market_intel.json
  ```
  Dostosuj `--environment`/`--governor` do konfiguracji portfela.
- Szablon konfiguracji hypercare dostępny w `config/core.yaml`
  (możesz go skopiować i uzupełnić o konkretne ścieżki środowiskowe; w razie
  potrzeby rozszerz struktury sekcji o dodatkowe pola specyficzne dla Stage6).
- Klucze HMAC umieszczone w `secrets/hmac/` i przypisane do komponentów Stage6.
- Ścieżki docelowe w `var/audit/...` z uprawnieniami zapisu.

> **Uwaga:** Wszystkie skrypty Stage6 uruchamiamy poprzez `python <ścieżka_do_skryptu>` (alias `python3` w aktywnym venv). Bezpośrednie `./scripts/...` omija ustawienia środowiska i nie jest wspierane.

## Procedura
1. Przygotuj plik konfiguracyjny YAML/JSON zawierający sekcje `summary`,
   `observability`, `resilience` oraz `portfolio`. Repozytorium udostępnia
   startowy szablon w `config/core.yaml`, który możesz skopiować i dostosować.
   Przykład minimalny (YAML):
   ```yaml
   summary:
     path: var/audit/stage6/hypercare_summary.json
     signing:
       key_path: secrets/hmac/stage6_summary.key
       key_id: stage6
   observability:
     definitions: config/observability/slo.yml
     metrics: var/metrics/stage6_measurements.json
     slo:
       json: var/audit/observability/slo_report.json
       csv: var/audit/observability/slo_report.csv
   resilience:
     bundle:
       source: var/audit/resilience
       output_dir: var/resilience
     audit:
       json: var/audit/resilience/audit_summary.json
     failover:
       plan: data/stage6/resilience/failover_plan.json
       json: var/audit/resilience/failover_summary.json
   portfolio:
     core_config: config/core.yaml
     environment: stage6-demo
     governor: default
     inputs:
       allocations: var/portfolio/allocations.yaml
       market_intel: var/audit/market_intel/report.json
       portfolio_value: 125000
       slo_report: var/audit/observability/slo_report.json
       stress_report: var/audit/risk/stress_lab.json
   ```
2. Upewnij się, że w katalogu `var/metrics/` znajduje się aktualny plik
   `stage6_measurements.json`. Jeżeli otrzymujesz go z innego środowiska,
   skopiuj lub zsynkuj plik do tej lokalizacji, np.:
   ```bash
   python scripts/sync_stage6_metrics.py \
     --source /mnt/backup/stage6_measurements.json \
     --output var/metrics/stage6_measurements.json
   ```
   W przypadku generowania nowego zestawu pomiarów wykonaj procedurę z
   runbooka Observability (kroki przygotowania metryk) i zapisz wynik w tej
   ścieżce. Skrypt `python scripts/sync_stage6_metrics.py` potwierdzi liczbę
   odczytanych pomiarów, a `python scripts/run_stage6_hypercare_cycle.py`
   zakończy się błędem, jeśli plik będzie nieobecny, podając tę ścieżkę oraz
   przykładowe polecenie kopiujące.
3. Uruchom orchestratora Stage6, wskazując przygotowany plik konfiguracyjny
   (domyślnie `config/core.yaml`):
   ```bash
   python scripts/run_stage6_hypercare_cycle.py --config config/core.yaml
   ```
   Skrypt wykona wszystkie cykle, zapisze raport zbiorczy i podpis HMAC, a w
   przypadku ostrzeżeń/błędów wypisze szczegóły w konsoli.
4. Zweryfikuj podpisany raport zbiorczy (opcjonalnie wymagaj podpisu):
   ```bash
   python scripts/verify_stage6_hypercare_summary.py \
     var/audit/stage6/hypercare_summary.json \
     --hmac-key-file secrets/hmac/stage6_summary.key \
     --require-signature
   ```
   Polecenie potwierdzi integralność raportu, wypisze wykryte ostrzeżenia lub
   błędy oraz może być archiwizowane w logach hypercare.
5. W razie potrzeby powtórz wykonanie dla środowisk testowych/produkcyjnych,
   modyfikując sekcję `portfolio` oraz ścieżki artefaktów.
6. Po uzyskaniu raportu Stage6 dołącz go do pełnego przeglądu hypercare zgodnie
   z runbookiem `FULL_HYPERCARE_CHECKLIST.md` (skrypt
   `python scripts/run_full_hypercare_summary.py`).

## Automatyzacja i CI

- **Planowany workflow GitHub Actions.** Repozytorium udostępnia workflow
  `Stage6 hypercare cycle` (`deploy/ci/github_actions_stage6_hypercare.yml`),
  który codziennie o 03:20 UTC uruchamia polecenie z predefiniowanym plikiem
  Stage6 (`config/core.yaml`):

  ```bash
  python scripts/run_stage6_hypercare_cycle.py --config config/core.yaml
  ```

  Aby uruchomić go ręcznie, przejdź do zakładki **Actions → Stage6 hypercare
  cycle** i wybierz **Run workflow**. Podsumowanie joba automatycznie wypisuje
  skrócony raport JSON (z kodem `json`), status ogólny raportu oraz tabelę
  statusów komponentów. Sekcje „Problemy” i „Ostrzeżenia” prezentują kluczowe
  komunikaty znalezione w raporcie, a tabela artefaktów z manifestu (kolumny:
  typ pliku, cele podpisu, rozmiar i skrót SHA-256) pozwala szybko ocenić
  kompletność archiwum bez pobierania paczki.
- **Artefakty raportów z CI.** Po każdym przebiegu workflow zestaw plików z
  katalogu `var/audit/stage6/` (raporty JSON oraz towarzyszące podpisy HMAC
  `*.sig`/`*.hmac`) wraz z plikiem `artifact_manifest.json` jest publikowany
  jako artefakt o nazwie `stage6-hypercare-<run_number>`. Manifest zawiera
  teraz dla każdego pliku jego rozmiar w bajtach, skrót SHA-256 oraz typ pliku
  (`kind = audit_json` lub `signature`). Dla podpisów manifest dopisuje pole
  `targets` wskazujące pliki, które podpis obejmuje, co ułatwia automatyczną
  walidację kompletności zestawu. Raport można pobrać przez interfejs Actions
  (**Artifacts → Download**) lub poleceniem:

  ```bash
  gh run download <run-id> --name stage6-hypercare-<run_number>
  ```

  Po rozpakowaniu znajdziesz najświeższy raport hypercare Stage6 oraz podpis,
  a także plik `artifact_manifest.json` z listą wszystkich spakowanych plików.
  Manifest ułatwia szybkie sprawdzenie, czy w artefakcie znajdują się np.
  podpisy HMAC lub raporty cząstkowe, a dodatkowe pola `size_bytes`, `sha256`,
  `kind` i `targets` pozwalają potwierdzić integralność oraz zgodność podpisów
  (te same informacje są widoczne w tabeli podsumowania workflow) przed ich
  weryfikacją narzędziem
  `python scripts/verify_stage6_hypercare_summary.py`.

- **Tabela spójności podpisów w podsumowaniu CI.** Podsumowanie joba zawiera
  teraz dodatkową tabelę „Spójność podpisów”, która dla każdego raportu JSON
  wypisuje, czy znaleziono odpowiadające podpisy (`*.sig`/`*.hmac`) oraz czy
  manifest wskazuje właściwy plik. Osierocone podpisy również są wypisane, co
  przyspiesza diagnozę brakujących raportów bez pobierania artefaktów.

## Artefakty/Akceptacja
- `var/audit/stage6/hypercare_summary.json` z podsumowaniem komponentów i
  listą ostrzeżeń/błędów.
- Plik podpisu HMAC (`*.sig`) odpowiadający raportowi zbiorczemu.
- Raport weryfikacji (stdout/JSON) potwierdzający poprawność podpisu i statusy
  komponentów, dołączony do logów hypercare.
- Raporty cząstkowe z sekcji Observability (`slo_report.json/.csv`,
  override, anotacje), Resilience (audit/failover/self-healing) oraz Portfolio
  (podsumowanie, ewentualny CSV, wpis decision logu) – zgodnie z checklistami
  dedykowanymi dla poszczególnych modułów.
- Zapis w decision logu Stage6 zawierający identyfikatory wygenerowanych
  artefaktów i numer podpisu raportu zbiorczego.
