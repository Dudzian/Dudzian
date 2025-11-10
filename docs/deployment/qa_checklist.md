# Lista kontrolna QA dla wydań offline

1. **Walidacja fingerprintu**
   - Uruchom `deploy/offline/wizard.py --fingerprint ... --fingerprint-keys ...`
     i upewnij się, że podpis HMAC jest rozpoznawany.
   - Zweryfikuj, że raportowane ostrzeżenia zostały zapisane w logu instalatora.

2. **Walidacja licencji OEM**
   - Z weryfikacją fingerprintu: `--license ... --license-keys ... --fingerprint ...`
   - Sprawdź, czy profil licencji zgadza się z docelowym środowiskiem.

3. **Test aktualizacji offline**
   - Przygotuj manifest i katalog z plikami aktualizacji.
   - Uruchom kreator z parametrem `--apply-update` i przejrzyj raport końcowy.
   - Jeśli aktualizacja wymaga wielu magazynów kluczy, podaj każdą ścieżkę
     `--update-keys` (argument można powtarzać).
   - Jeżeli dystrybuujesz spakowaną paczkę (ZIP/TAR), użyj `--update-archive`
     zamiast `--update-payload` – kreator rozpakowuje archiwum i zachowuje
     strukturę katalogów w `updates/<manifest>/`.
   - Upewnij się, że pliki pojawiły się w katalogu `updates/` oraz, że zachowane są
     oryginalne nazwy manifestów i podpisów.
   - Zweryfikuj, że w `logs/offline_installer_summary.json` zapisane zostało
     podsumowanie próby wraz z timestampem oraz ścieżką `payload_archive`, jeśli
     użyto archiwum.

4. **Sprawdzenie lokalnych ścieżek konfiguracyjnych**
   - Potwierdź, że pliki logów trafiają do `logs/` w katalogu instalacyjnym.
   - Zaktualizuj `config/marketplace/repository.json` lub inne pliki do trybu
     lokalnego (brak zewnętrznych URL-i).

5. **Uruchomienie testów regresyjnych**
   - `pytest tests/ui` – minimalny smoke test UI.
   - `python scripts/build/build_cross_platform_installers.py --skip-windows` –
     weryfikacja ścieżki Linux/macOS.

6. **Materiały do wydania**
   - Zbierz wygenerowane instalatory z katalogu `dist/installers/`.
   - Zarchiwizuj raport JSON z kreatora (`logs/offline_installer_summary.json` lub
     ścieżkę przekazaną przez `--summary-path`).
   - Zaktualizuj dokumentację klienta (np. README wydania) informacją o wersji.
