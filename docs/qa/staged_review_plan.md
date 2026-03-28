# Etapowy plan przeglądu jakości i hardeningu (duży codebase)

## Założenia
- Zakres repo jest duży (setki katalogów, >800 plików kodu i testów), dlatego plan jest podzielony na małe, niezależne paczki.
- Pojedyncza paczka obejmuje jeden moduł logiczny albo 10–25 plików / ~10k–30k LOC wraz z lokalnymi testami.
- Celem jest sekwencja: najpierw obszary o najwyższym ryzyku operacyjnym, bezpieczeństwa i poprawności transakcyjnej, potem stabilizacja i utrzymanie.

## 1) Repo map (główne obszary odpowiedzialności)

### A. Krytyczna ścieżka transakcyjna
- `bot_core/runtime`, `bot_core/trading`, `bot_core/execution`, `bot_core/exchanges`, `bot_core/auto_trader`, `bot_core/decision`
- Odpowiedzialność: uruchamianie runtime, routing decyzji, egzekucja zleceń, integracje exchange, automatyzacja handlu.

### B. Ryzyko, bezpieczeństwo i zgodność
- `bot_core/risk`, `bot_core/security`, `bot_core/compliance`, `core/security`, `tests/security`, `tests/compliance`
- Odpowiedzialność: limity ryzyka, walidacje bezpieczeństwa, polityki compliance i audytowalność.

### C. Dane, modele i AI
- `bot_core/data`, `bot_core/ai`, `core/ml`, `docs/ml`, `tests/ai`, `tests/ml`
- Odpowiedzialność: pipeline danych, inferencja, komponenty ML/AI, integralność wejść/wyjść modeli.

### D. Obserwowalność i odporność operacyjna
- `bot_core/observability`, `bot_core/monitoring`, `bot_core/resilience`, `core/telemetry`, `deploy/monitoring`
- Odpowiedzialność: metryki, alerty, tracing, odporność na awarie i degradacje.

### E. API/UI i warstwa usług
- `bot_core/api`, `bot_core/services`, `bot_core/ui`, `tests/ui*`, `tests/services`
- Odpowiedzialność: interfejsy użytkownika, API, kontrakty wejścia/wyjścia i ergonomia operacyjna.

### F. Konfiguracja, infrastruktura i release
- `bot_core/config`, `core/config`, `deploy/*`, `scripts/build`, `scripts/ci`, `scripts/deploy`, `tests/ci`, `tests/deploy`
- Odpowiedzialność: konfiguracja środowisk, pipeline build/release, powtarzalność wdrożeń.

### G. Jakość testów i aktywa wspierające
- `tests/*`, `tests/integration`, `tests/e2e`, `tests/smoke`, `tests/performance`, `tests/load`
- Odpowiedzialność: pokrycie krytycznych ścieżek, stabilność testów i koszt utrzymania.

## 2) Review batches (małe paczki)

> Rozmiary są estymacjami operacyjnymi na podstawie struktury i objętości katalogów.

### Batch 1 — Exchange adapters (core execution edge)
- Ścieżki:
  - `bot_core/exchanges`
  - `tests/exchanges`
- Szacowany rozmiar: ~103 pliki, ~30k LOC (dzielone na 2 podetapy)
- Podział roboczy:
  - 1A: konektory + auth/signing (10–20 plików)
  - 1B: retry/rate-limit/error mapping + testy lokalne (10–25 plików)
- Typ pracy: correctness review, hardening, test review
- Ryzyko/wartość: najwyższe ryzyko błędnej egzekucji i błędów integracyjnych; duży wpływ finansowy.

### Batch 2 — Runtime orchestration
- Ścieżki:
  - `bot_core/runtime`
  - `tests/runtime`
- Szacowany rozmiar: ~80 plików, ~25k–30k LOC (2–3 podetapy)
- Typ pracy: correctness review, quality review, hardening
- Ryzyko/wartość: centralny orchestrator, awarie tu propagują się globalnie.

### Batch 3 — Trading + execution pipeline
- Ścieżki:
  - `bot_core/trading`
  - `bot_core/execution`
  - `tests/trading`
  - `tests/execution`
- Szacowany rozmiar: ~30–40 plików, ~13k LOC
- Typ pracy: correctness review, hardening, test review
- Ryzyko/wartość: logika decyzja→zlecenie→potwierdzenie; kluczowa poprawność finansowa.

### Batch 4 — Auto-trader control loop
- Ścieżki:
  - `bot_core/auto_trader`
  - `tests/auto_trader`
- Szacowany rozmiar: ~20–25 plików, ~23k LOC
- Typ pracy: correctness review, hardening
- Ryzyko/wartość: pętle automatyczne i mechanizmy sterowania mogą amplifikować błędy.

### Batch 5 — Risk engine
- Ścieżki:
  - `bot_core/risk`
  - `tests/risk`
  - `docs/risk` (tylko spójność reguł)
- Szacowany rozmiar: ~20–30 plików, ~11k LOC
- Typ pracy: correctness review, hardening, test review
- Ryzyko/wartość: bezpośrednio ogranicza potencjalne straty i przekroczenia limitów.

### Batch 6 — Security controls
- Ścieżki:
  - `bot_core/security`
  - `core/security`
  - `tests/security`
  - `scripts/security`
- Szacowany rozmiar: ~60–80 plików, ~20k+ LOC (2 podetapy)
- Typ pracy: hardening, correctness review, build/release review
- Ryzyko/wartość: ekspozycja na incydenty bezpieczeństwa; wysoka wartość redukcji ryzyka.

### Batch 7 — Compliance & auditability
- Ścieżki:
  - `bot_core/compliance`
  - `tests/compliance`
  - `docs/compliance`
  - `audit/*`
- Szacowany rozmiar: ~15–30 plików, ~5k–10k LOC + artefakty
- Typ pracy: correctness review, quality review, test review
- Ryzyko/wartość: wymagania regulacyjne i dowodowość działań.

### Batch 8 — Decision engine
- Ścieżki:
  - `bot_core/decision`
  - `tests/decision`
  - `data/decision_engine`
- Szacowany rozmiar: ~35–45 plików, ~10k LOC
- Typ pracy: correctness review, quality review, test review
- Ryzyko/wartość: jakość decyzji wpływa na wyniki i bezpieczeństwo strategii.

### Batch 9 — AI/ML runtime integration
- Ścieżki:
  - `bot_core/ai`
  - `core/ml`
  - `tests/ai`
  - `tests/ml`
- Szacowany rozmiar: ~50–65 plików, ~28k LOC
- Typ pracy: correctness review, hardening, test review
- Ryzyko/wartość: ryzyka dryfu, fallbacków i niejawnych błędów inferencji.

### Batch 10 — Data ingestion and data quality
- Ścieżki:
  - `bot_core/data`
  - `tests/data`
  - `docs/data`
- Szacowany rozmiar: ~30–40 plików, ~5k–8k LOC
- Typ pracy: quality review, correctness review, test review
- Ryzyko/wartość: jakość danych warunkuje poprawność downstream.

### Batch 11 — Observability & resilience
- Ścieżki:
  - `bot_core/observability`
  - `bot_core/monitoring`
  - `bot_core/resilience`
  - `core/telemetry`
  - `tests/monitoring`, `tests/observability`
- Szacowany rozmiar: ~30–45 plików, ~9k–12k LOC
- Typ pracy: hardening, quality review, test review
- Ryzyko/wartość: skraca MTTR i ogranicza koszty incydentów.

### Batch 12 — API/services/ui contracts
- Ścieżki:
  - `bot_core/api`
  - `bot_core/services`
  - `bot_core/ui`
  - `tests/ui`, `tests/ui_backend`, `tests/ui_pyside`, `tests/services`
- Szacowany rozmiar: ~35–55 plików, ~12k LOC
- Typ pracy: quality review, correctness review, test review
- Ryzyko/wartość: stabilność interfejsów i kompatybilność kontraktów.

### Batch 13 — Config and runtime settings
- Ścieżki:
  - `bot_core/config`
  - `core/config`
  - `env.example`, `config.toml`
- Szacowany rozmiar: ~10–20 plików, ~10k LOC
- Typ pracy: hardening, quality review
- Ryzyko/wartość: błędy konfiguracji często powodują ciche awarie produkcyjne.

### Batch 14 — Build/CI/deploy pipeline
- Ścieżki:
  - `scripts/build`, `scripts/ci`, `scripts/deploy`, `scripts/packaging`
  - `deploy/*`
  - `tests/ci`, `tests/deploy`, `tests/packaging`
- Szacowany rozmiar: ~80–120 plików, ~30k+ LOC (3 podetapy)
- Typ pracy: build/release review, hardening, test review
- Ryzyko/wartość: łańcuch dostarczania i niezawodność release.

### Batch 15 — Strategy layer & backtest coupling
- Ścieżki:
  - `bot_core/strategies`
  - `bot_core/backtest`
  - `tests/strategies`, `tests/backtest`
- Szacowany rozmiar: ~60–75 plików, ~11k–14k LOC
- Typ pracy: quality review, correctness review, test review
- Ryzyko/wartość: średnie ryzyko produkcyjne, duża wartość jakości modeli strategii.

### Batch 16 — Reporting, docs consistency, non-critical support
- Ścieżki:
  - `bot_core/reporting`, `core/reporting`, `tests/reporting`
  - `docs/reporting`, wybrane `docs/runbooks`
- Szacowany rozmiar: ~20–35 plików, ~6k LOC
- Typ pracy: quality review, test review
- Ryzyko/wartość: niższe ryzyko runtime, wysoka wartość operacyjna i komunikacyjna.

## 3) Recommended order (od najwyższego ryzyka/wpływu)
1. Batch 1 — Exchange adapters
2. Batch 2 — Runtime orchestration
3. Batch 3 — Trading + execution pipeline
4. Batch 5 — Risk engine
5. Batch 6 — Security controls
6. Batch 4 — Auto-trader control loop
7. Batch 8 — Decision engine
8. Batch 9 — AI/ML runtime integration
9. Batch 10 — Data ingestion and data quality
10. Batch 11 — Observability & resilience
11. Batch 13 — Config and runtime settings
12. Batch 14 — Build/CI/deploy pipeline
13. Batch 12 — API/services/ui contracts
14. Batch 15 — Strategy layer & backtest coupling
15. Batch 7 — Compliance & auditability
16. Batch 16 — Reporting & support

> Uwaga: Batch 7 (compliance) może być przesunięty wyżej, jeśli presja regulacyjna/audytowa jest bieżącym priorytetem biznesowym.

## 4) First batch to inspect now
**Rekomendacja startu: Batch 1 (Exchange adapters, etap 1A).**

Zakres pierwszego sprintu przeglądu (bez refaktoru i bez pełnego test suite):
- 10–20 plików z `bot_core/exchanges` skoncentrowanych na:
  - podpisywaniu/auth,
  - mapowaniu błędów API,
  - idempotencji i retry.
- Lokalne testy z `tests/exchanges` powiązane z wybranymi adapterami.

Checklista pracy w pierwszej paczce:
1. Zmapować invariants: kolejność składania zleceń, pre/post-condition, oczekiwane kody błędów.
2. Zweryfikować scenariusze timeout/rate-limit/network split.
3. Sprawdzić, czy testy lokalne pokrywają ścieżki błędów krytycznych.
4. Przygotować listę ryzyk i rekomendacji naprawczych (bez implementacji na tym etapie).
