# Stage6 operator workshop – 2024-06-07

**Uczestnicy:** Trading (M. Krawiec), Risk (A. Nowicka), Compliance (J. Szafran), Stage6 Ops (facylitacja).

## Kontekst
Spotkanie poświęcone było zgraniu konfiguracji Stage6 Market Intelligence, Stress Lab oraz Portfolio Governora z aktualnym profilem ryzyka i wymaganiami zgodności OEM. Analizie poddano ostatnie raporty stress lab z 2024-05-29 oraz logi decyzji governora.

## Ustalenia

### Market Intelligence
- Trading wymagał zwiększenia udziału danych wolumenowych dla SOLUSDT jako proxy rotacji altów.
- Risk zasugerował podniesienie wag bazowych raportów, aby sygnały SLO miały większy wpływ na rebalans.
- Compliance przypomniał o konieczności odzwierciedlenia decyzji w manifestach audytowych.

**Decyzje:**
1. Dodajemy SOLUSDT do listy symboli obowiązkowych w agregacji Market Intel.
2. Podnosimy wagę domyślną agregatów do 1.15, aby portfolio governor miał wyższy priorytet dla świeżych metryk.
3. Zachowujemy dotychczasowy katalog wynikowy oraz ścieżki audytu.

### Stress Lab
- Risk wskazał, że dotychczasowe limity dopuszczały zbyt głębokie spadki płynności zanim pipeline sygnalizował naruszenie.
- Trading poprosił o obniżenie tolerancji na blackouty do 40 minut, aby szybciej angażować desk.
- Compliance poprosił o niższe progi fundingowych i dyspersji, aby szybciej eskalować kwestie zgodności.

**Decyzje:**
1. Zacieśniamy wszystkie główne progi Stress Lab: płynność 55%, spready 45 bps, zmiana wolatylności 80%, sentyment 0.50, funding 25 bps, latency 150 ms, blackout 40 min, dyspersja 50 bps.
2. Dla scenariusza blackoutowego utrzymujemy margines awaryjny, ale korygujemy override latency do 180 ms i blackout do 55 minut (wcześniej 220/75).
3. Raport końcowy musi zawierać sekcję podpisaną HMAC, a weryfikacja progu ma być wykonywana przed publikacją raportu.

### Portfolio Governor
- Trading zaproponował skrócenie interwału rebalansowania do 20 minut i obniżenie wygładzenia, aby szybciej reagować na sygnały z Market Intel.
- Risk nalegał na wyższe koszty transakcyjne i minimalny próg sygnału 0.08, aby ograniczyć nadmierne zmiany wag.
- Compliance zatwierdził nowe limity wag dla strategii, pod warunkiem aktualizacji runbooków kontrolnych.

**Decyzje:**
1. Rebalance co 20 minut, smoothing 0.55, default baseline 0.28, min 0.08, max 0.50.
2. Minimalny score 0.08, koszt bazowy 5.0 bps, scoring: alpha 0.9, cost 1.3, slo 1.2, risk 0.8.
3. Nowe limity strategii:
   - Trend: baseline 0.38, min 0.22, max 0.60, max sygnałów 3, mnożnik 1.25.
   - Mean reversion: baseline 0.27, min 0.10, max 0.45, max sygnałów 5, mnożnik 1.45.
   - Vol target: baseline 0.23, min 0.12, max 0.40, max sygnałów 3, mnożnik 1.15.
   - Cross exchange: baseline 0.12, min 0.05, max 0.28, max sygnałów 4, mnożnik 1.30.
4. Każda zmiana wchodzi do użytku po aktualizacji konfiguracji i podpisaniu raportu compliance.

## Zadania następcze
- [x] Zaktualizować `config/core.yaml` z nowymi progami.
- [x] Wdrożyć aktualizacje w runbookach Stage6 (Portfolio & Stress Lab).
- [x] Przygotować wspólny skrypt audytowy do kontroli progów Stage6 (`scripts/verify_stage6_thresholds.py`).
- [x] Rozszerzyć weryfikator o eksport raportu JSON dla archiwizacji audytowej.
- [ ] Odtworzyć raport Stress Lab po wdrożeniu (planowane na 2024-06-08).

