# Paper Labs – Symulacje ryzyka

Dokument opisuje proces analizy scenariuszy ryzyka uruchamianych w ramach
pipeline'u „Paper Labs”. Każde uruchomienie wymaga przejścia przez checklistę
wejścia, walidację wyników oraz finalną akceptację compliance.

## Checklist wejść

Przed startem zadania Paper Labs należy potwierdzić, że:

- [ ] Potwierdzono aktualność scenariuszy smoke testu zdefiniowanych w
      `bot_core/risk/simulation.py::DEFAULT_SMOKE_SCENARIOS` lub dostarczono
      własny plik Parquet na potrzeby bieżącego uruchomienia.
- [ ] Udostępniono konfigurację profilu manualnego (maksymalnie cztery pozycje,
      dźwignia 4.0, limity dzienne 5%) w repozytorium CI.
- [ ] Operator Paper Labs posiada ważne poświadczenia do przestrzeni sekretów
      `ci/paper`.
- [ ] Katalog wyjściowy `var/paper_smoke_ci/risk_simulation` jest czysty lub
      zawiera historyczne raporty z poprzednich uruchomień (do celów audytu).

## Wyniki symulacji

Silnik `ThresholdRiskEngine` generuje dwa artefakty raportowe:

1. `risk_simulation_report.json` – pełne dane decyzji z każdego scenariusza.
2. `risk_simulation_report.pdf` – skrócony raport kontrolny gotowy do wysyłki.

Wyniki muszą zawierać poniższe elementy:

- komplet czterech scenariuszy (konserwatywny, zbalansowany, agresywny,
  manualny);
- informację o liczbie zaakceptowanych i odrzuconych zleceń;
- listę powodów odrzuceń wraz z podpowiedziami korekt (jeśli wystąpiły);
- flagę trybu awaryjnego (force liquidation) dla każdego profilu.

## Akceptacja compliance

Dział compliance zatwierdza raport wyłącznie, jeśli wszystkie z poniższych
warunków są spełnione:

1. Raport JSON i PDF zostały wygenerowane w tym samym przebiegu pipeline'u.
2. W scenariuszu manualnym nie wystąpiły odrzucenia z powodu braku limitów.
3. Wartości limitów profilu manualnego są zgodne z polityką klienta
   (zdefiniowaną w sekcji checklisty wejść).
4. Status kroku „Run risk simulation suite” w GitHub Actions zakończył się
   sukcesem i artefakty zostały zarchiwizowane.

Po pozytywnej weryfikacji compliance dołącza krótką notatkę do `docs/audit/`
ze wskazaniem numeru przebiegu (`github.run_number`) i daty akceptacji.
