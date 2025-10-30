# Plan wsparcia i obsługi zgłoszeń

## Cel dokumentu
Plan opisuje zasady wsparcia technicznego dla użytkowników końcowych bota handlowego, w tym poziomy SLA, kanały kontaktu oraz wzorce zgłoszeń.

## Zakres wsparcia
- Instalacja i aktualizacja aplikacji desktopowej.
- Aktywacja licencji i odcisku sprzętowego.
- Integracja z giełdami oraz konfiguracja strategii.
- Diagnostyka błędów runtime, alertów i marketplace.

## Kanały kontaktu
| Kanał | Opis | Godziny pracy |
|-------|------|---------------|
| Portal wsparcia (ticket) | Dostępny w panelu klienta OEM, umożliwia zgłoszenie incydentu lub prośby o zmianę. | 24/7 (odpowiedź zgodnie z SLA) |
| E-mail | support@example.com (automatyczna rejestracja zgłoszenia). | 24/7 |
| Hotline awaryjny | Numer dostępny po aktywacji licencji premium. | 8:00–20:00 CET |
| Kanał społecznościowy | Discord / Telegram (społeczność, brak gwarantowanego SLA). | Brak gwarancji |

## Poziomy SLA
| Priorytet | Opis | Czas reakcji | Czas obejścia | Czas rozwiązania |
|-----------|------|--------------|---------------|------------------|
| P1 – Krytyczny | Brak działania aplikacji, niemożność składania zleceń. | 1 h | 4 h | 24 h |
| P2 – Wysoki | Błędy funkcji kluczowych (np. brak połączenia z giełdą). | 4 h | 1 dzień | 3 dni |
| P3 – Średni | Błędy funkcji niekrytycznych, problemy konfiguracyjne. | 1 dzień | 3 dni | 7 dni |
| P4 – Niski | Sugestie rozwojowe, pytania informacyjne. | 3 dni | n/d | n/d |

## Proces obsługi zgłoszeń
1. **Rejestracja** – użytkownik zgłasza incydent poprzez portal lub e-mail, wypełniając formularz według szablonu.
2. **Triaga** – zespół wsparcia określa priorytet, potwierdza dane logowania i powiązanie licencji.
3. **Diagnoza** – inżynier wsparcia analizuje logi (`logs/`), raporty monitoringu oraz metryki.
4. **Akcja korygująca** – wdrożenie poprawki, obejście lub przekazanie do zespołu R&D.
5. **Zamknięcie** – potwierdzenie rozwiązania przez użytkownika, aktualizacja bazy wiedzy.

## Szablon zgłoszenia
```
[Tytuł]
Opis problemu:
Kroki odtworzenia:
Oczekiwany rezultat:
Rzeczywisty rezultat:
Logi (załączniki):
Wersja aplikacji i systemu:
Priorytet (P1–P4):
Dodatkowe informacje (zrzuty ekranu, preset strategii):
```

## Eskalacja
- Brak odpowiedzi w zadeklarowanym czasie reakcji → eskalacja do koordynatora wsparcia.
- Nierozwiązane incydenty P1/P2 po czasie obejścia → eskalacja do CTO i zespołu R&D.

## Raportowanie i przeglądy
- Miesięczne raporty KPI (czas reakcji, czas rozwiązania, liczba incydentów).
- Przegląd kwartalny jakości wsparcia wraz z planem usprawnień.

## Materiały dodatkowe
- Troubleshooting: `docs/user_manual/troubleshooting.md`.
- Procedury bezpieczeństwa: `docs/security/runbook.md`.
- Instrukcje instalacji: `docs/deployment/installer_build.md`.

