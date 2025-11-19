# Stage5 – lista kontrolna szkoleń i decision logu

## Cel
Zapewnić udokumentowanie wszystkich szkoleń Stage5 (compliance, operacyjne,
rotacje kluczy) wraz z metadanymi i podpisem HMAC, aby spełnić wymagania
hypercare oraz audytu.

## Przygotowanie
1. Zbierz listę uczestników, prowadzących i zakres tematów szkolenia.
2. Przygotuj klucz HMAC dla artefaktów `var/audit/stage5/training` (rotowany
   zgodnie z polityką bezpieczeństwa).
3. Ustal identyfikator sesji, np. `S5-TRAIN-2024-05-15`.

## Kroki
1. Uruchom rejestrację szkolenia:
   ```bash
   python -m scripts.log_stage5_training \
     S5-TRAIN-2024-05-15 "Stage5 Compliance" "Anna Trainer" \
     --summary "Przegląd wymagań Stage5" \
     --participants "Anna,Bob" \
     --topics "SLO,Resilience" \
     --materials "slides.pdf,playbook.md" \
     --compliance-tags "stage5,compliance" \
     --signing-key-env STAGE5_TRAINING_HMAC
   ```
2. Zweryfikuj, że plik został zapisany w `var/audit/stage5/training/` i zawiera
   podpis HMAC.
3. Przekaż uczestnikom streszczenie decyzji i zadania następcze z pola
   `actions`.

## Artefakty / Akceptacja
- Plik JSON z logiem szkolenia, np.
  `var/audit/stage5/training/training_S5-TRAIN-2024-05-15.json`.
- Podpis HMAC w polu `signature` (algorytm `HMAC-SHA256`).
- Lista działań następczych z przypisanymi właścicielami (pole `actions`).
- Notatka w decision logu hypercare, że szkolenie zostało zrealizowane.
