# Runbook reagowania na alerty bezpieczeństwa

Ten dokument opisuje standard operacyjny reagowania na alerty bezpieczeństwa dla zespołu Dudzian.

## 1. Klasyfikacja alertów

| Poziom | Kryteria | Wymagany czas reakcji |
| --- | --- | --- |
| **Krytyczny** | aktywne wykorzystanie podatności, wyciek danych, kompromitacja kluczy | do 30 minut |
| **Wysoki** | nowe podatności o wysokim CVSS, naruszenia kontroli dostępu | do 4 godzin |
| **Średni** | ostrzeżenia narzędzi SAST/DAST, nieudane próby logowania masowego | do 1 dnia roboczego |
| **Niski** | błędy konfiguracyjne, ostrzeżenia jakościowe | do 3 dni roboczych |

## 2. Kanały zgłoszeń

1. Powiadomienia automatyczne z pipeline'ów bezpieczeństwa (Slack `#sec-alerts`, mailing `security@dudzian.example`).
2. Zgłoszenia manualne od zespołu operacji lub supportu (`Jira/SOC`).
3. Eskalacje partnerów zewnętrznych (CERT, dostawcy chmurowi).

Wszystkie zgłoszenia muszą zostać zarejestrowane w narzędziu śledzenia incydentów wraz z priorytetem i opisem.

## 3. Proces reagowania

1. **Triaging**
   - Dyżurny inżynier bezpieczeństwa ocenia ważność alertu, potwierdza priorytet i przypisuje właściciela.
   - Weryfikuje dostępność dodatkowych danych (logi z `logs/security/`, raporty z CI, metryki).
2. **Zawężenie i potwierdzenie**
   - Reprodukuje problem (jeżeli możliwe) na środowisku izolowanym.
   - Potwierdza zakres wpływu: systemy, konta, dane.
3. **Działania natychmiastowe**
   - Izolacja zagrożonych zasobów (wyłączenie usług, rotacja kluczy, blokada kont).
   - Aktualizacja firewall/WAF oraz polityk dostępu.
4. **Remediacja trwała**
   - Przygotowanie i wdrożenie poprawek (kod, konfiguracja, aktualizacje zależności).
   - Uruchomienie testów regresyjnych i bezpieczeństwa po wdrożeniu.
5. **Komunikacja**
   - Aktualizacje co 30/60 minut dla alertów krytycznych/wysokich.
   - Informacje dla interesariuszy (produkt, operacje, compliance).
6. **Zamknięcie**
   - Potwierdzenie usunięcia zagrożenia i brak kolejnych alertów.
   - Uzupełnienie dokumentacji incydentu i oznaczenie jako zamknięty w narzędziu śledzącym.

## 4. Analiza po-incydentowa (Postmortem)

- Termin: maksymalnie 5 dni roboczych od zamknięcia alertu krytycznego/wysokiego.
- Wymagane elementy:
  - Oś czasu zdarzeń.
  - Analiza przyczyn źródłowych (RCA) z przypisaniem kontrolom zapobiegawczym.
  - Lekcje wyniesione i zadania zapobiegające ponownemu wystąpieniu.
  - Aktualizacje runbooków i automatyzacji.

## 5. Monitoring i metryki

- `MTTA` (średni czas reakcji) oraz `MTTR` (średni czas rozwiązania) liczone miesięcznie.
- Liczba otwartych alertów według poziomu ważności.
- Pokrycie pipeline'ów bezpieczeństwa (SAST, DAST, skanowanie sekretów, IaC).

## 6. Kontakt i dyżury

- Lista dyżurujących inżynierów przechowywana w `docs/operations/oncall.md`.
- Kanał eskalacyjny: `+48 123 456 789` (24/7 SOC) oraz `incident@dudzian.example`.
- W przypadku braku odpowiedzi w ciągu 15 minut – eskalacja do Security Lead oraz CTO.

## 7. Przegląd runbooka

- Runbook jest przeglądany co kwartał przez Security Lead i aktualizowany w repozytorium.
- Zmiany są zatwierdzane w ramach przeglądu technicznego (minimum 2 recenzentów z zespołu bezpieczeństwa).

