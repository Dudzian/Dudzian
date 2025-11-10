# Stage6 Stress Lab – whitepaper produktowy

## Cel dokumentu
Ten dokument streszcza przewagi Stage6 Stress Lab na tle konkurencji klasy CryptoHopper oraz wskazuje, jak wykorzystać raporty audytowe i runbooki operacyjne do komunikacji wartości z klientami OEM.

## Podstawy technologiczne
- **Pełna integracja z Market Intelligence i Portfolio Governor** – ustalenia warsztatowe z 7 czerwca 2024 r. potwierdzają, że Stress Lab stanowi integralny element pipeline'u decyzyjnego (zacieśnianie progów płynności, blackoutów i dyspersji oraz korekta parametrów portfolio).[^audit-workshop]
- **Offline-first i podpisy HMAC** – każdy raport generowany przez `run_stress_lab.py` jest podpisany, co umożliwia dystrybucję w środowiskach bez dostępu do chmury.[^paper-trading-runbook]
- **Checklisty hypercare** – runbooki Stage6 definiują obowiązki operacyjne (audyt progów, walidacja bundli, komunikacja marketplace), co zapewnia spójność z wymogami OEM.[^marketing-runbook]

## Architektura Stress Lab
1. **Warstwa symulacyjna** – moduł `bot_core.risk.stress_lab` generuje scenariusze multi-market oraz metryki płynności i ryzyka. Raporty JSON/CSV są wzbogacone o rekomendacje override.
2. **Warstwa podpisu** – CLI obsługuje zarówno klucz lokalny (`--signing-key-path`), jak i zaczytanie z konfiguracji `core.yaml`, wymuszając integralność raportów.
3. **Warstwa dystrybucji** – raporty trafiają do katalogu audytu (`var/audit/stage6/…`) oraz do pakietu marketingowego poprzez automatyczny workflow (patrz sekcja „Automatyzacja eksportu”).

## Przewagi Stage6 vs CryptoHopper
| Obszar | Stage6 Stress Lab | CryptoHopper | Wpływ biznesowy |
| --- | --- | --- | --- |
| Zakres scenariuszy | Multi-market, blackout infrastrukturalny, funding, latency, dyspersja, override guardrails. | Paper trading, backtesty historyczne, brak podpisów HMAC. | Stage6 umożliwia compliance-ready audyty i eskalacje w czasie rzeczywistym. |
| Integracja portfelowa | Decyzje warsztatowe przekładają się na automatyczne zmiany w Portfolio Governor (rebalance 20 min, guardraile wag).[^audit-workshop-portfolio] | Manualne mapowanie sygnałów na strategie. | Skraca czas reakcji desków i zapewnia spójność danych. |
| Observability | Raporty Stress Lab trafiają do benchmarku i runbooków hypercare, a UI wizualizuje metryki cyklu decyzyjnego.[^runtime-status] | Dashboard online bez audytowych artefaktów offline. | Pełna przejrzystość dla audytu oraz partnerów OEM. |

## Proces komunikacji wartości
1. **Przygotowanie raportu** – `python scripts/run_stress_lab.py run --config config/core.yaml --output reports/stress_lab/latest.json --signing-key-path secrets/stress_lab.hmac`.
2. **Walidacja** – podpis sprawdzany przez `python scripts/verify_stage6_thresholds.py` oraz checklistę `docs/runbooks/marketplace_marketing.md` (sekcja 2).[^marketing-runbook-checklist]
3. **Dystrybucja** – raport i podpis trafiają do pakietu marketingowego (whitepaper + case studies) oraz artefaktów release.

## Narracja sprzedażowa
- **Redukcja ryzyka operacyjnego** – skrócenie tolerancji blackoutów do 40 min i czasy reakcji guardrail (override latency 180 ms) pokazują, że Stage6 reaguje szybciej niż standardowe boty chmurowe.[^audit-workshop-stress]
- **Dowód zgodności** – każdy raport Stress Lab jest powiązany z dziennikiem decyzji i checklistą hypercare, co ułatwia due diligence OEM.[^benchmark-comparison]
- **Elastyczność deploymentu** – offline-first pozwala na pełną autonomię lokalną bez wymogu łączności chmurowej, zachowując przy tym integrację z UI Stage6.

## Wymagania operacyjne
- Aktualizacja progów Stress Lab w konfiguracji core wymaga równoczesnej aktualizacji runbooków i dokumentacji marketingowej.
- Raporty muszą być przechowywane wraz z podpisami (`*.sig`) i manifestem (`*.manifest.json`).[^audit-manifest]
- Marketing publikuje streszczenie wyników w cyklu T+7 po release (zgodnie z runbookiem).

## Załączniki
- Raport warsztatowy: `var/audit/stage6/2024-06-07-stage6-operator-workshop.md`.
- Checklisty audytowe: `var/audit/stage6/stage6_preliminary_checklist.md`.
- Runbook marketingowy: `docs/runbooks/marketplace_marketing.md`.
- Benchmark vs CryptoHopper: `docs/benchmark/cryptohopper_comparison.md`.

[^audit-workshop]: `var/audit/stage6/2024-06-07-stage6-operator-workshop.md` – sekcja „Stress Lab” i decyzje operacyjne.
[^paper-trading-runbook]: `docs/runbooks/paper_trading.md` – sekcja podpisów HMAC dla raportów Stress Lab.
[^marketing-runbook]: `docs/runbooks/marketplace_marketing.md` – checklisty marketingowe i hypercare.
[^audit-workshop-portfolio]: `var/audit/stage6/2024-06-07-stage6-operator-workshop.md` – sekcja „Portfolio Governor”.
[^runtime-status]: `docs/runtime/status_review.md` – tabela luk UI i integracji metryk.
[^marketing-runbook-checklist]: `docs/runbooks/marketplace_marketing.md` – tabela checklisty publikacji.
[^audit-workshop-stress]: `var/audit/stage6/2024-06-07-stage6-operator-workshop.md` – sekcja „Stress Lab”, decyzje dotyczące progów.
[^benchmark-comparison]: `docs/benchmark/cryptohopper_comparison.md` – sekcje Compliance i Stress Lab.
[^audit-manifest]: `var/audit/stage6/stage6_preliminary_checklist.manifest.json` – wymagane artefakty audytowe.
