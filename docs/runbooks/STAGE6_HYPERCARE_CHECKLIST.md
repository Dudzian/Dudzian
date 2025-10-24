# STAGE6 – Checklist: Hypercare Orchestrator

## Cel
Wykonać kompletny cykl hypercare Stage6 (Observability + Resilience + Portfolio)
jednym poleceniem, uzyskując podpisany raport zbiorczy oraz powiązane
artefakty audytowe.

## Prerekwizyty
- Aktualne raporty wejściowe dla Observability (definicje/metyki SLO –
  repozytoryjny plik `config/observability/slo.yml`), Resilience (plan
  failover, manifesty paczek, polityka) oraz Portfolio
  (alokacje, Market Intel, raporty SLO/Stress Lab).
- Szablon konfiguracji hypercare dostępny w `config/stage6/hypercare.yaml`
  (możesz go skopiować i uzupełnić o konkretne ścieżki środowiskowe).
- Klucze HMAC umieszczone w `secrets/hmac/` i przypisane do komponentów Stage6.
- Ścieżki docelowe w `var/audit/...` z uprawnieniami zapisu.

> **Uwaga:** Wszystkie skrypty Stage6 uruchamiamy poprzez `python <ścieżka_do_skryptu>` (alias `python3` w aktywnym venv). Bezpośrednie `./scripts/...` omija ustawienia środowiska i nie jest wspierane.

## Procedura
1. Przygotuj plik konfiguracyjny YAML/JSON zawierający sekcje `summary`,
   `observability`, `resilience` oraz `portfolio`. Repozytorium udostępnia
   startowy szablon w `config/stage6/hypercare.yaml`, który możesz skopiować i
   dostosować. Przykład minimalny (YAML):
   ```yaml
   summary:
     path: var/audit/stage6/hypercare_summary.json
     signing:
       key_path: secrets/hmac/stage6_summary.key
       key_id: stage6
   observability:
     definitions: config/observability/slo.yml
     metrics: var/audit/observability/metrics.json
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
2. Uruchom orchestratora Stage6, wskazując przygotowany plik konfiguracyjny
   (domyślnie `config/stage6/hypercare.yaml`):
   ```bash
   python scripts/run_stage6_hypercare_cycle.py --config config/stage6/hypercare.yaml
   ```
   Skrypt wykona wszystkie cykle, zapisze raport zbiorczy i podpis HMAC, a w
   przypadku ostrzeżeń/błędów wypisze szczegóły w konsoli.
3. Zweryfikuj podpisany raport zbiorczy (opcjonalnie wymagaj podpisu):
   ```bash
   python scripts/verify_stage6_hypercare_summary.py \
     var/audit/stage6/hypercare_summary.json \
     --hmac-key-file secrets/hmac/stage6_summary.key \
     --require-signature
   ```
   Polecenie potwierdzi integralność raportu, wypisze wykryte ostrzeżenia lub
   błędy oraz może być archiwizowane w logach hypercare.
4. W razie potrzeby powtórz wykonanie dla środowisk testowych/produkcyjnych,
   modyfikując sekcję `portfolio` oraz ścieżki artefaktów.
5. Po uzyskaniu raportu Stage6 dołącz go do pełnego przeglądu hypercare zgodnie
   z runbookiem `FULL_HYPERCARE_CHECKLIST.md` (skrypt
   `python scripts/run_full_hypercare_summary.py`).

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
