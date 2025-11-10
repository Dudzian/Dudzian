# Roadmap: Integracja adapterów, marketplace i metryki pokrycia

## Cel nadrzędny
Skonsolidować trzy równoległe inicjatywy — adaptację środowisk paper/live, rozwój marketplace presetów oraz raportowanie metryk pokrycia — tak, aby w ciągu jednego kwartału dostarczyć spójny ekosystem wdrożeń, kontroli jakości i raportowania.

## Oś czasu i kamienie milowe
| Miesiąc | Kluczowe rezultaty | Zależności krzyżowe |
| --- | --- | --- |
| Tydzień 1 | Audyt adapterów, zdefiniowanie KPI oraz metryk pokrycia. | Wspólna definicja KPI z analityką i finansami. |
| Tygodnie 2–4 | Projekt unified adapter API, architektura katalogu publicznego, rozszerzenia ETL. | Zasoby DEV/UX, sandboxy giełd, zgody prawne. |
| Tygodnie 5–9 | Implementacja adapterów z failover, wdrożenie wersjonowania presetów, pipeline raportowy. | Dostęp do środowisk staging, inżynierowie danych. |
| Tygodnie 10–11 | Walidacja E2E, beta-testy marketplace, integracja raportów kwartalnych. | Harmonogram publikacji raportów, compliance. |
| Tydzień 12 | Retrospekcja, stabilizacja wsparcia L2/L3, przygotowanie rolloutów marketingowych. | Zespół komunikacji, CS. |

---

## 1. Integracja adapterów paper/live z testami failover

### Zakres i cele
- Zapewnienie pełnej zgodności funkcjonalnej pomiędzy trybami paper oraz live.
- Wprowadzenie automatycznych testów failover z kontrolą MTTR i MTTD.
- Ujednolicenie konfiguracji i obserwowalności adapterów.

### Etapy
1. **Discovery (Tydzień 1)**
   - Audyt checklisty onboardingowej i harnessów testowych.
   - Zebranie wymagań API (limity, auth, rate limits) dla każdej giełdy.
   - Utworzenie katalogu scenariuszy awarii (HTTP 5xx, degradacja latencji, utrata połączenia).
2. **Design (Tygodnie 2–3)**
   - Aktualizacja abstrakcji adapterów (interfejs, contract tests, DI).
   - Rozszerzenie schematów konfiguracyjnych o parametry środowiskowe i plany eskalacji.
   - Specyfikacja testów failover z wykorzystaniem chaos runnera oraz fault injection.
3. **Implementation (Tygodnie 4–7)**
   - Implementacja sterowników paper/live ze wspólnymi walidatorami i telemetry hooks.
   - Budowa harnessu failover (syntetyczne błędy, replay logów, throttling simulator).
   - Integracja z CI (dedykowany stage dla chaos tests, alerting na Slack/PagerDuty).
4. **Validation & Launch (Tygodnie 8–9)**
   - Testy integracyjne w stagingu z obciążeniem i symulacją awarii.
   - Publikacja runbooków (L1/L2), checklist GA, aktualizacja dokumentacji klientów.

### Artefakty do dostarczenia
- Aktualizowany schemat `adapters.yaml` z parametrami środowiskowymi.
- `failover_test_matrix.xlsx` z opisem scenariuszy i wyników.
- Dashboard w Grafanie: MTTR, MTTD, procent scenariuszy pokrytych.

### Ryzyka i mitigacje
- **Brak sandboxów giełd** → plan awaryjny: mocki kontraktowe + symulacja w VCR.
- **Niedostępne limity API** → dynamiczne throttlingi + kontakt z account managerem.
- **Chaos runner zbyt inwazyjny** → feature flagi i stopniowa eskalacja injekcji.

---

## 2. Rozbudowa marketplace presetów (katalog publiczny, wersjonowanie, recenzje offline)

### Zakres i cele
- Udostępnienie katalogu presetów w trybie publicznym z filtrowaniem i kuracją.
- Wprowadzenie wersjonowania semantycznego, changelogów i mechanizmów rollback.
- Umożliwienie recenzji offline z podpisanymi pakietami i ścieżką audytu.

### Etapy
1. **Research (Tydzień 1)**
   - Inwentaryzacja aktualnego workflow publikacji i kontroli dostępu.
   - Analiza wymogów prawnych (compliance, licencje, PII).
   - Benchmark UX vs. konkurencja (filtrowanie, rekomendacje, tagging).
2. **Architecture (Tygodnie 2–4)**
   - Projekt usługi katalogu (REST + cache), modelu danych i ACL.
   - Rozszerzenie schematu presetów (`preset.json`) o pola wersji, kompatybilności, checksum.
   - Specyfikacja formatu paczek recenzji (ZIP + podpis GPG, manifest JSON).
3. **Build (Tygodnie 5–9)**
   - Implementacja API katalogu, indeksu wyszukiwarki i synchronizacji cache.
   - Dodanie CLI/CI do zarządzania wersjami: `preset publish`, `preset deprecate`, `preset rollback`.
   - Stworzenie modułów UI: karta wersji, timeline zmian, import recenzji offline.
4. **Rollout (Tygodnie 10–11)**
   - Beta-test publicznego katalogu (wybrani partnerzy), feedback loop.
   - Włączenie alertów o nowych wersjach i automatycznych downgrade warnings.
   - Publikacja polityk recenzji, integracja z logami audytowymi.

### Artefakty do dostarczenia
- Dokument API (`marketplace_catalog_openapi.yaml`).
- Biblioteka komponentów UI dla kart presetów i timeline.
- Skrypty migracyjne bazy (`V2025_07__preset_versions.sql`).

### Metryki sukcesu
- ≥90% presetów z poprawnym versioningiem i changelogiem.
- Średni czas publikacji < 5 minut od merge do katalogu publicznego.
- ≥3 recenzje offline dla każdej wersji major w ciągu 14 dni.

### Ryzyka
- **Opóźnienia w audycie bezpieczeństwa** → równoległe przygotowanie checklist + rezerwowy slot audytorów.
- **Brak zasobów UX** → wykorzystanie design systemu i komponentów reusable.
- **Niska adopcja recenzji offline** → program incentivów + szkolenia partnerów.

---

## 3. Integracja metryk pokrycia z raportami kwartalnymi

### Zakres i cele
- Zdefiniowanie spójnych wskaźników pokrycia dla giełd, klas aktywów i presetów.
- Automatyzacja pipeline ETL z kwartalnymi snapshotami i walidacją danych.
- Włączenie metryk do dashboardów BI oraz raportów PDF/HTML.

### Etapy
1. **Alignment (Tydzień 1)**
   - Warsztat z analityką i finansami: definicje metryk, KPI bazowe.
   - Mapowanie źródeł danych (telemetria, logi wdrożeń, marketplace) do modelu raportowego.
   - Uzgodnienie SLA aktualizacji (np. T+2 dni po końcu kwartału).
2. **Data Pipeline Enhancements (Tygodnie 2–4)**
   - Rozszerzenie jobów ETL (Airflow/Prefect) o agregacje per giełda/preset.
   - Normalizacja danych referencyjnych (słowniki giełd, taksonomia presetów).
   - Zapis snapshotów kwartalnych w hurtowni (`coverage_quarterly_fact`).
3. **Reporting Integration (Tygodnie 5–6)**
   - Aktualizacja dashboardów BI (Looker/Metabase) o sekcje pokrycia.
   - Dodanie modułu generowania narracji (NLG) do raportów kwartalnych.
   - Integracja z workflow publikacji raportów (approval gate, archiwizacja).
4. **Governance & QA (Tygodnie 7–8)**
   - Implementacja testów jakości danych (freshness, completeness, drift).
   - Ustalenie właścicieli (Data Steward, Reporter) i procedur eskalacji.
   - Dokumentacja procesu, checklist release'owych, onboarding nowych członków.

### Artefakty do dostarczenia
- Definicje metryk (`coverage_metrics_dictionary.md`).
- Notebook walidacyjny (`coverage_validation.ipynb`) z testami regresji danych.
- Szablon raportu (`quarterly_report_template_v2.pptx` + sekcja coverage).

### KPI i monitoring
- 100% raportów kwartalnych zawiera sekcję coverage, automatycznie aktualizowaną.
- Walidacje ETL przechodzą przed publikacją (zielony status w 3 kolejnych cyklach).
- Różnica QoQ pomiędzy systemem automatycznym a manualnym < ±2%.

### Ryzyka
- **Opóźnienia danych źródłowych** → fallback do danych H-1, oznaczenie w raporcie.
- **Zmiany w definicji KPI** → proces change management z zatwierdzeniem CDO.
- **Przeciążenie pipeline** → autoskalowanie workerów + monitoring kosztów.

---

## Koordynacja międzyzespołowa
- Cotygodniowy steering committee (adaptery, marketplace, analityka, compliance).
- Wspólna tablica ryzyk i zależności w Jira/Notion, aktualizowana co piątek.
- Synchronizacja komunikacji do klientów: kalendarz release notes, newsletter, webinar.

## Plan komunikacji i wsparcia
- **Support**: przygotowanie L2/L3 playbooków, szkoleń i bazy wiedzy.
- **Monitoring**: centralny dashboard OKR + alerting (Slack/Email) na regresje KPI.
- **Retrospekcja**: po wdrożeniu kwartalnym warsztat lessons learned, aktualizacja roadmapy.

## Kryteria sukcesu
- Adaptery paper/live osiągają pełną parytet funkcjonalny, a testy failover zamykają się w <1h MTTR.
- Marketplace oferuje katalog publiczny z aktywnym wersjonowaniem i audytowalnymi recenzjami offline.
- Raporty kwartalne prezentują metryki pokrycia w sposób spójny, automatyczny i zgodny z definicjami KPI.
