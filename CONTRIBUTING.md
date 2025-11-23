# Wkład w projekt bot_core

Dziękujemy za chęć współtworzenia bot_core! Poniżej znajdziesz minimalny
proces kontrybucji obowiązujący w repozytorium.

## Proces review

1. Utwórz branch funkcjonalny i utrzymuj zmiany w małych, dobrze opisanych
   commitach.
2. Przed złożeniem PR uzupełnij opis o zakres zmian, wpływ na bezpieczeństwo oraz
   informację o scenariuszach testowych (wszystkie muszą działać w trybie demo).
3. PR wymagają minimum **dwóch** niezależnych review – jednego technicznego i
   jednego z zespołu bezpieczeństwa/compliance.
4. Reviewerzy sprawdzają zgodność z dokumentacją architektury, wymaganiami KYC/AML
   oraz checklistą bezpieczeństwa. Zmiany są mergowane dopiero po zatwierdzeniu
   przez oba zespoły.

## Checklista bezpieczeństwa przed PR

- [ ] Potwierdziłem uruchomienie pipeline'u wyłącznie w trybie demo/testnet.
- [ ] Zweryfikowałem, że nie są używane produkcyjne klucze API ani dane klientów.
- [ ] Oceniłem wpływ zmian na zarządzanie ryzykiem i dodałem brakujące alerty/logi.
- [ ] Sprawdziłem, czy konfiguracja wymusza flagi `require_demo_mode` oraz
      odpowiednie limity pozycji.
- [ ] Zgłosiłem potencjalne incydenty bezpieczeństwa zespołowi `#sec-alerts`.

## Testy lokalne (obowiązkowe)

Przed otwarciem PR uruchom wszystkie poniższe polecenia:

```bash
python scripts/lint_paths.py
ruff check \
  bot_core/trading/engine.py \
  bot_core/strategies/base \
  bot_core/config/loader.py \
  bot_core/strategies/catalog.py
mypy
pytest tests/test_pipeline_paper.py
```

Jeżeli dodajesz dodatkowe testy jednostkowe/integracyjne, dopisz je do sekcji
"Test Plan" w opisie PR. Pamiętaj, aby **nie** przełączać środowiska na live
bez pisemnego zatwierdzenia zespołu compliance po zakończeniu testów demo.

Lint layoutu kończy się błędem za każdym razem, gdy w repozytorium pojawią się
zakazane katalogi (prefiksy legacy z poprzedniego etapu) albo pliki wykonywalne wewnątrz `archive/`.
Pozostałości należy usunąć lub przenieść do dokumentacji historycznej przed
wysłaniem zmian do review.

Importy z `archive/**` są zabronione w kodzie runtime/CI (patrz
`scripts/check_no_archive_imports.py` oraz test `tests/qa/test_no_archive_imports.py`).
Jeżeli potrzebujesz materiałów referencyjnych, trzymaj je w `archive/` bez
wiązań importowych do aktywnego kodu.

### Statyczne typowanie

- `python -m mypy` obejmuje teraz pakiety `bot_core.auto_trader`, `bot_core.ai`,
  `bot_core.risk`, `bot_core.execution`, warstwy `core/config`, `core/reporting`,
  `core/security`, `core/licensing`, a także kluczowe moduły packagingu desktopu
  (`deploy/packaging/*`) oraz wybrane skrypty runtime (`run_cloud_service.py`,
  `run_stage6_resilience_cycle.py`, `run_stage6_observability_cycle.py`,
  `run_local_bot.py`, `list_exchange_adapters.py`, `run_ai_governor_cycle.py`,
  `validate_marketplace_presets.py`).
- `reports/__init__.py` utrzymuje katalog artefaktów raportowych w zakresie mypy,
  aby potwierdzić kompletność instalacji z pełnymi informacjami o typach.
- Wyjątki importowe dopuszczalne są wyłącznie w dedykowanych override'ach dla
  bibliotek bez stubów (np. PySide6/shiboken6, pandas, fastapi, grpc, shap,
  jsonschema); nie używamy globalnego `ignore_missing_imports` ani
  `follow_imports="silent"`.
- Lista `disable_error_code` w override'ach jest ograniczona do uzasadnionych
  wyjątków dla dynamicznych API (m.in. `attr-defined`, kompatybilność `arg-type`
  z Qt/QML); nowe tłumienia zgłaszaj w review wraz z kodami błędów i powodem,
  zamiast dodawać globalne wyłączenia.

### Użycie słowa „legacy”

- W kodzie wykonywalnym (np. `bot_core`, `core`, `scripts/`, `ui/pyside_app`, `proto/`,
  testy) słowo „legacy” jest zakazane i test `tests/qa/test_no_legacy_tokens.py` zablokuje merge.
- Dopuszczalne konteksty to wyłącznie dokumentacja (`docs/**`), archiwum (`archive/**`)
  oraz pliki zaczynające się od `README`. Każda taka wzmianka musi wskazywać konkretną
  lokalizację w `archive/` (np. `archive/ui_cpp_preserved.md`) albo sekcję migracji.
- Poza powyższą listą nie dodajemy nowych odwołań – w razie potrzeby rozszerzenia
  allowlisty należy dołączyć zmianę w teście QA i uzasadnienie w opisie PR.
