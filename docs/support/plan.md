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

## Checklist aktualizacji benchmarku Stage6
> Checklistę traktuj jako wymaganie bramki release – dopiero po jej odhaczeniu podpisujemy releas hypercare.

1. Zweryfikuj aktualność danych w `docs/benchmark/cryptohopper_comparison.md` (obszar strategia, automatyzacja, UI, compliance) po każdym releasie hypercare.
2. Zaktualizuj status priorytetów (pokrycie giełdowe, marketplace presetów, integracja UI ↔ runtime, przewagi compliance) i oznacz zmiany w dzienniku releasu.
3. Potwierdź, że artefakty audytowe Stage6 zawierają podpisy HMAC i są zarchiwizowane w `var/audit/` wraz z raportami benchmarku.
4. Zaktualizuj tablicę wyników i harmonogram działań korygujących w `docs/benchmark/cryptohopper_comparison.md` (statusy 🟢/🟡/🔴, odpowiedzialni, cele metryk).
5. Dodaj wpis do sekcji „Historia aktualizacji benchmarku” z datą releasu, opisem zmian i linkami do artefaktów (hypercare, marketplace, audyt compliance, testy UI).
6. Przekaż aktualizację zespołowi produktowemu podczas przeglądu wsparcia, linkując do zaktualizowanego benchmarku i status_review.
7. Potwierdź synchronizację z `docs/runtime/status_review.md` – rozbieżności otwierają zadania follow-up.
8. Przygotuj krótkie porównanie dla klientów pytających o alternatywy (CryptoHopper/Gunbot) – odwołuj się do `config/marketplace/catalog.md` (persony strategii) oraz najnowszego `reports/exchanges/<data>.csv` z checklistami HyperCare.

### Raportowanie benchmarku
- **Odpowiedzialny operacyjny:** Owner Stage6 Support (koordynuje aktualizację checklisty i benchmarku).
- **Artefakty wymagane przy releasie:**
  - `var/audit/hypercare/<data>/summary.json` (podpisany raport cyklu hypercare).
  - `reports/exchanges/<data>.csv` (stan adapterów live/paper) oraz log aktualizacji adapterów.
  - `reports/strategy/presets_<data>.md` (lista presetów publicznych z recenzjami) wraz z potwierdzeniem marketingu.
  - `reports/ui/tests/<build_id>/grpc_feed.json` z p95 opóźnień feedu i odniesieniem do wyników testów UI w CI.
  - `var/audit/compliance/<okres>.pdf` lub JSON z wynikami audytu decyzji.
- **Dystrybucja:** pakiet benchmarku (tabela wyników + historia) archiwizujemy w `var/audit/benchmark/<data>/` i wysyłamy do zespołów produktowych w ramach notatki release’owej.

### Referencje rynkowe i porównania konkurencji
- **Katalog strategii:** `config/marketplace/catalog.md` (oraz podpis `.sig`) zawiera ≥15 strategii z personami – wykorzystuj go w odpowiedziach do klientów pytających o pokrycie scenariuszy CryptoHopper/Gunbota.
- **Raport giełdowy:** `reports/exchanges/2025-01-15.csv` (oraz nowsze snapshoty) dostarcza checklisty futures Deribit/BitMEX oraz statusy HyperCare (failover, latencja, koszty); dołączaj je do zgłoszeń o niezawodność infrastruktury.
- **Benchmark konkurencji:** `docs/benchmark/cryptohopper_comparison.md` opisuje różnice wobec CryptoHoppera i Gunbota – wsparcie powinno cytować ten dokument przy eskalacjach produktowych i aktualizować wnioski w bazie wiedzy.

## Materiały dodatkowe
- Troubleshooting: `docs/user_manual/troubleshooting.md`.
- Procedury bezpieczeństwa: `docs/security/runbook.md`.
- Instrukcje instalacji: `docs/deployment/installer_build.md`.

