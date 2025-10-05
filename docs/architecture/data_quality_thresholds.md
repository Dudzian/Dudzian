# Progi jakości danych OHLCV per profil ryzyka

## Kontekst

Monitoring pokrycia danych OHLCV zasila pre-checki (`paper_precheck`), ręczne
kontrole (`check_data_coverage`) oraz automatyczny runner alertów. Aby uniknąć
rozbieżności pomiędzy środowiskami, progi jakości są definiowane w profilach
ryzyka i dziedziczone przez środowiska, które nie definiują własnych limitów.

## Ustalony zestaw progów

| Profil ryzyka  | max_gap_minutes | Opis                                   | min_ok_ratio | Uzasadnienie |
|----------------|-----------------|---------------------------------------|--------------|--------------|
| conservative   | 1440            | Alert po >24 h bez nowych świec       | 0.92         | codzienny refresh + ≥92% OK |
| balanced       | 2160            | Tolerancja do 36 h (opóźnienia nocne) | 0.90         | ochrona D1 przy ≥90% OK |
| aggressive     | 2880            | Do 48 h dla rynków agresywnych        | 0.85         | tolerancja 48h, ≥85% OK |
| manual         | 4320            | Do 72 h przy kontroli ręcznej         | 0.75         | kontrola ręczna, ≥75% OK |

`min_ok_ratio` kontroluje udział wpisów z wynikiem OK względem całkowitej liczby
sprawdzonych kombinacji symbol/interwał. Gdy manifest zawiera mniej wpisów OK
niż próg, runner oznacza naruszenie nawet przy braku błędów strukturalnych.

## Integracja w konfiguracji

- Progi są ustawiane w sekcji `risk_profiles` (`config/core.yaml`).
- Loader konfiguracji automatycznie wstrzykuje progi do środowisk, które nie
  definiują własnych limitów.
- Skrypty i runner alertów raportują wartości progów w JSON, co ułatwia audyt i
  kalibrację.
- Raporty `check_data_coverage`/`coverage_alert_runner` zawierają statystyki luk
  (medianę, percentyle, maksimum), dzięki czemu alerty Telegram/e-mail od razu
  wskazują skalę problemu bez zaglądania do logów.

## Dalsze kroki kalibracyjne

1. Po zasileniu magazynu danych paper/testnet należy zebrać statystyki
   historyczne (`worst_gap_minutes`, `ok_ratio`) dla każdej giełdy i interwału.
   Pomocny jest skrypt `coverage_gap_report`, który generuje raporty JSON z
   percentylami luk oraz rozbiciem na interwały.
2. W razie fałszywych alarmów dostosować progi środowiskowe lub wprowadzić
   dodatkowe różnicowanie (np. osobne wartości dla interwałów 1h/15m).
3. Zespół ryzyka powinien co najmniej raz na kwartał przeglądać progi i raporty
   z runnera, aby weryfikować, czy spełniają one wymogi polityki ekspozycji.

