# Stage6 Resilience Drill – lista kontrolna

## Cel
Potwierdzić skuteczność procedur failover i zebrać podpisane artefakty raportowe przed wejściem w tryb live.

## Przygotowanie
- [ ] Zweryfikowano aktualność konfiguracji `resilience` w `config/core.yaml` (ścieżki datasetów, progi).
- [ ] Zrotowano klucz HMAC dla raportów resilience (`STAGE6_RESILIENCE_SIGNING_KEY`).
- [ ] Zapełniono katalog `var/audit/stage6/resilience` i zapewniono uprawnienia 700.

## Wykonanie
1. [ ] Uruchom `python scripts/failover_drill.py --config config/core.yaml --fail-on-breach`.
2. [ ] Sprawdź logi w konsoli (brak błędów I/O, potwierdzenie zapisania raportu i podpisu).
3. [ ] Zbierz raport JSON (`resilience_failover_report.json`) oraz podpis `.sig` do artefaktów.
4. [ ] Zbuduj paczkę resilience: `python scripts/export_resilience_bundle.py --version <tag> --output-dir var/audit/stage6/resilience --signing-key-env STAGE6_RESILIENCE_SIGNING_KEY`.
5. [ ] Zweryfikuj paczkę offline poleceniem `python scripts/verify_resilience_bundle.py --bundle var/audit/stage6/resilience/resilience-bundle-<tag>.tar.gz --signing-key-env STAGE6_RESILIENCE_SIGNING_KEY`.
6. [ ] Jeżeli którykolwiek drill posiada status `failed`, eskaluj do L2 i powtórz po weryfikacji przyczyn.

## Artefakty / Akceptacja
- Raport JSON + podpis HMAC oraz podpisana paczka `resilience-bundle-<tag>.tar.gz` w katalogu `var/audit/stage6/resilience`.
- Zaktualizowany decision log (manualny wpis operatora z wynikami drill'u).
- Zrzut ekranu z dashboardu Stage6 Observability++ (potwierdzenie metryk failover).

## Notatki
> **Uwaga:** Wszystkie skrypty Stage6 uruchamiamy poprzez `python <ścieżka_do_skryptu>` (alias `python3` w naszym venv). Bezpośrednie `./scripts/...` omija aktywne środowisko i nie jest wspierane.
- Lista fallbacków i progi muszą odpowiadać aktualnym wymaganiom compliance.
- W przypadku braku danych telemetrycznych dopuszczalne jest użycie datasetów offline opisanych w `data/stage6/resilience/`.
