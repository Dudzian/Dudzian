# Specyfikacja Etapu 5 – Optymalizacja kosztów i rozszerzony decision engine

## 1. Kontekst
Etap 5 ma na celu przejście z fazy wdrożenia wielostrate-gicznego (Etap 4) do operacyjnie dojrzałej platformy klasy OEM, która potrafi aktywnie optymalizować koszty transakcyjne, bilansować decyzje inwestycyjne oraz utrzymywać rygorystyczne wymagania compliance w pełnym cyklu demo → paper → live. Wszystkie prace muszą zachować pełną pracę offline, podpisy HMAC oraz zabezpieczenia mTLS/RBAC.

## 2. Zakres
- **Strumień TCO (Total Cost of Ownership / Transaction Cost Optimization):**
  - Modelowanie i raportowanie kosztów (prowizje, slippage, funding) na poziomie strategii, profili ryzyka oraz scheduler-a.
  - Implementacja modułu analitycznego `bot_core/tco/` z raportami CSV/PDF podpisanymi HMAC.
  - Integracja danych kosztowych z `run_oem_acceptance.py` oraz pipeline'ami CI.
- **Rozszerzony decision engine:**
  - Dodanie warstwy `DecisionOrchestrator` oceniającej warianty wejść/wyjść strategii z uwzględnieniem kosztów, ryzyka i limitów compliance.
  - Wsparcie dla symulacji scenariuszy (stress testy TCO, worst case latency) oraz gatingu przed przejściem live.
  - Rozszerzenie decision logu o metadane TCO oraz podpisy kroków orkiestracji.
  - Konfiguracja `decision_engine` w `config/core.yaml` opisująca progi orkiestrowania (koszt, edge, limity ryzyka, stress testy) oraz nowy skrypt smoke `python scripts/run_decision_engine_smoke.py`.
- **Observability & Compliance+:**
  - Monitorowanie SLO (latencja, koszt/ trade, fill rate) z bundlowanym dashboardem „Stage5 – Compliance & Cost Control” oraz checklistą `docs/runbooks/STAGE5_COMPLIANCE_CHECKLIST.md`.
  - Moduł rotacji kluczy i przypomnienia (CLI `python scripts/rotate_keys.py`, alerty) wraz z checklistą audytową oraz playbookiem `docs/runbooks/STAGE5_SUPPORT_PLAYBOOK.md`.
  - Generowanie podpisanych raportów PDF/CSV dla audytorów (koszty, decyzje, incydenty) oraz paczek obserwowalności (`python scripts/export_observability_bundle.py`).
- **Enablement operacyjny:**
  - Checklisty demo→paper→live rozszerzone o progi TCO i audyty decision engine.
  - Warsztaty L1/L2 i scenariusze szkoleniowe dla analityków kosztów oraz operatorów compliance.

## 3. Zależności
- Dane rynkowe muszą dostarczać szczegółowe informacje o prowizjach, płynności i finansowaniu (rozszerzenie manifestów w `data/backtests/`).
- Runtime wymaga rozszerzenia `ThresholdRiskEngine` o limity kosztowe oraz integracji z `DecisionOrchestrator`.
- Observability opiera się na istniejącej infrastrukturze Stage4 (Prometheus, Grafana, bundling HMAC) i wymaga aktualizacji workflow CI.
- Bezpieczeństwo i compliance opierają się na modułach fingerprint/licencji oraz decision logu podpisywanego HMAC.

## 4. Definicje ukończenia (Definition of Done)
- **Kod:** moduły TCO i decision engine posiadają testy jednostkowe/integracyjne oraz pokrycie ≥85%, włączone do pipeline'ów CI i smoke testów.
- **Konfiguracja:** `config/core.yaml` zawiera progi kosztowe, definicje SLO oraz parametry decision engine (np. minimalny spread, maks. koszt slippage na profil). Wszystkie konfiguracje podpisane i zwalidowane przez bundler OEM.
- **Dokumentacja:** spec, runbooki (audyt TCO, rotacja kluczy, szkolenia) oraz raporty techniczne dostępne w `docs/architecture/` i `docs/runbooks/` (w tym `STAGE5_COMPLIANCE_CHECKLIST.md`, `STAGE5_SUPPORT_PLAYBOOK.md`, `docs/training/stage5_workshop.md`).
- **Testy:** pipeline CI uruchamia `run_tco_analysis.py`, `run_decision_engine_smoke.py` oraz `validate_compliance_reports.py`; wyniki archiwizowane i podpisane HMAC.
- **Operacje i bezpieczeństwo:** checklista demo→paper→live zawiera sekcję TCO/DecisionOrchestrator, a moduł rotacji kluczy generuje wpisy w decision logu i alerty SLO.

## 5. Kamienie milowe
1. **Discovery i baza danych TCO:** zmapowanie źródeł kosztów, definicja KPI, aktualizacja manifestów danych, szkic raportów (szczegóły w `docs/architecture/stage5_discovery.md`).
2. **Moduł TCO + raporty:** implementacja `bot_core/tco`, generatory CSV/PDF, integracja z Paper Labs i bundlerem OEM.
3. **DecisionOrchestrator + risk integration:** rozszerzenie runtime, symulacje, gating compliance, aktualizacja decision logu.
4. **Observability & Compliance+:** dashboard Stage5, monitoring SLO, bundling raportów, moduł rotacji kluczy.
5. **Enablement i audyty:** szkolenia, runbooki, playbooki L1/L2, testy regresyjne i finalny audit acceptance Etapu 5, w tym
   podpisany log warsztatów (`python scripts/log_stage5_training.py`).

## 6. Ryzyka i mitigacje
- **Niepełne dane kosztowe:** fallback do modeli estymacyjnych i ujednoliconego API prowizji; wymuszenie checklisty danych przed iteracją.
- **Złożoność decision engine:** podział na mikroserwisy w ramach repo `bot_core`, hermetyzacja interfejsów oraz regresje testowe.
- **Compliance i audyt:** podpisy HMAC dla raportów, rotacja kluczy i dedykowany CLI audytowy. Regularne dry-run’y acceptance.
- **Wydajność:** testy obciążeniowe TCO/decision engine, monitor SLO i mechanizmy rollbacku.

## 7. Harmonogram wysokiego poziomu
- Tydzień 1: Discovery TCO, aktualizacja manifestów danych, definicja KPI i szkic raportów.
- Tydzień 2: Implementacja modułu TCO i generatorów raportów, testy jednostkowe, integracja z CI.
- Tydzień 3: DecisionOrchestrator – implementacja, symulacje, integracja z risk engine i decision logiem.
- Tydzień 4: Observability & Compliance+ – dashboardy, SLO, rotacja kluczy, bundling obserwowalności.
- Tydzień 5: Enablement – runbooki, playbooki, szkolenia, finalny audit i podpis Etapu 5.
