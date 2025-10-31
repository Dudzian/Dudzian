# Walidacja jakości danych treningowych

Niniejszy moduł opisuje zestaw walidacji uruchamianych przed każdą iteracją retreningu
Decision Engine. Walidacja jest wykonywana przez `core.data.validators.DatasetValidator`
i rejestrowana w katalogu `logs/data/validation/` w formacie JSON.

## Zakres kontroli

1. **Reguła `missing_data` (błędy krytyczne)**
   - brak jakichkolwiek wektorów w zbiorze treningowym,
   - niepoprawne targety (wartości `NaN`, `inf`, typy nienumeryczne),
   - cechy o wartościach `None`, `NaN`, `inf` lub nienumerycznych.
   - wykryte problemy powodują przerwanie retreningu i zapis szczegółów w raporcie.

2. **Reguła `feature_anomalies` (ostrzeżenia)**
   - cechy o zerowej zmienności (stałe kolumny),
   - cechy o bardzo szerokim zakresie wartości względem odchylenia standardowego.
   - ostrzeżenia nie blokują treningu, ale pojawiają się w raporcie jako sygnał do audytu.

## Format raportu

Każde uruchomienie tworzy plik `dataset_validation_YYYYmmddTHHMMSSffffffZ.json` zawierający:

- znacznik czasu wykonania,
- status (`passed` lub `failed`),
- metadane datasetu (liczba wierszy, symbole, itp.),
- listę problemów wykrytych przez reguły wraz z poziomem istotności.

Przykładowy fragment raportu:

```json
{
  "generated_at": "2025-05-18T14:12:30.102938Z",
  "status": "failed",
  "dataset_metadata": {
    "row_count": 1,
    "symbols": ["BTCUSDT"]
  },
  "issues": [
    {
      "rule": "missing_data",
      "severity": "error",
      "message": "Wykryto nieprawidłowe wartości targetów (NaN/inf).",
      "details": {"rows": [0]}
    }
  ]
}
```

## Integracja z pipeline treningowym

`TrainingPipeline` uruchamia walidację automatycznie. W przypadku błędów podnoszony jest
`DatasetValidationError`, który zawiera ścieżkę do pliku raportu. Logi sukcesu i ostrzeżeń
są również zapisywane i dołączane do raportów retreningu (`validation_log_path`).

## Rekomendacje operacyjne

- monitoruj katalog `logs/data/validation/` w systemach OEM i dołącz raporty do zgłoszeń
  serwisowych,
- w przypadku ostrzeżeń o zerowej zmienności rozważ ponowną inżynierię cech lub sanity check
  źródeł danych,
- błędy `missing_data` wymagają natychmiastowej diagnostyki źródeł OHLCV lub procesu ETL,
  aby zapewnić ciągłość działania bota.
