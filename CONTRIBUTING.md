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
