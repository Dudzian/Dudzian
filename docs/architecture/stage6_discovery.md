# Stage 6 – Discovery i przygotowanie zakresu

## 1. Cel discovery
Celem discovery Etapu 6 jest określenie wymagań dla kolejnego kroku rozwoju platformy OEM po domknięciu Etapu 5. Discovery ma
zebrać twarde dane o efektywności kosztowej i decyzyjnej, ustalić priorytety produktowe oraz oszacować wpływ na istniejące
moduły (strategiczne, ryzyka, egzekucyjne, runtime, compliance). Faza kończy się zatwierdzoną specyfikacją techniczno-
operacyjną oraz roadmapą implementacji.

## 2. Kluczowe pytania badawcze
- Jakie są docelowe cele biznesowe Etapu 6 (np. auto-rebalancing portfela, inteligentne strojenie strategii, automatyczna
  kalibracja kosztów)?
- Jakie są obecne ograniczenia decyzji i kosztów wykazane w raportach TCO/DecisionOrchestratora z Etapu 5?
- Jakie dodatkowe dane rynkowe/offchain muszą być gromadzone (np. depth-of-book, dane funding, wskaźniki sentymentu)?
- Jakie ryzyka compliance i bezpieczeństwa pojawią się przy bardziej autonomicznej orkiestracji (np. wymagania dotyczące
  overrides operatora, dodatkowe podpisy)?
- W jaki sposób należy rozbudować pipeline demo → paper → live, aby uwzględniał nowe testy regresyjne i scenariusze awaryjne?

## 3. Artefakty discovery
- **Analiza danych**: zrzuty raportów TCO/SLO/DecisionOrchestratora, analiza luk oraz propozycje KPI Etapu 6.
- **Mapowanie modułów**: macierz wpływu na istniejące komponenty (`bot_core/strategies`, `bot_core/risk`, `bot_core/execution`,
  `bot_core/tco`, `bot_core/decision`, `bot_core/runtime`).
- **Warsztaty**: zespół tradingowy, risk, compliance, operatorzy L1/L2 — protokoły spotkań podpisane HMAC i zarejestrowane w
  decision logu (wymagane użycie `python scripts/log_stage5_training.py`).
- **Draft spec**: szkic specyfikacji Etapu 6 obejmujący zakres, kamienie milowe, definicje ukończenia, ryzyka i harmonogram.
- **Plan danych**: wymagane rozszerzenia manifestów Parquet/SQLite, źródeł offchain i pipeline'ów walidacyjnych.

## 4. Harmonogram discovery (proponowany)
| Tydzień | Aktywności |
| --- | --- |
| 1 | Analiza raportów Etapu 5, identyfikacja luk kosztowych i decyzyjnych, przygotowanie KPI. |
| 2 | Warsztaty z zespołami trading/risk/compliance, rejestracja w decision logu, aktualizacja mapy modułów. |
| 3 | Opracowanie planu danych, draft specyfikacji i aktualizacja roadmapy demo → paper → live. |
| 4 | Review z interesariuszami (product, compliance, operations), podpisanie discovery reportu i publikacja specyfikacji Etapu 6. |

## 5. Definition of Ready dla Etapu 6
Discovery uznaje się za zakończone, gdy:
- Powstała specyfikacja `docs/architecture/stage6_spec.md` z zatwierdzonym zakresem i kamieniami milowymi.
- Przygotowano checklistę operacyjną Stage6 (draft) oraz aktualizację runbooków demo → paper → live.
- Udokumentowano ryzyka i zależności wraz z planem mitigacji.
- Wszystkie artefakty discovery znajdują się w `var/audit/stage6_discovery/<timestamp>/` i posiadają podpisy HMAC.
