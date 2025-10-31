# Aktualizacje offline (.kbot)

Poniższa instrukcja opisuje proces przygotowania i instalacji podpisanych pakietów
aktualizacji przeznaczonych do dystrybucji offline. Pakiety w formacie `.kbot`
umożliwiają dostarczenie aktualizacji modeli, strategii lub zasobów aplikacji bez
konieczności łączenia się z serwerami zewnętrznymi.

## Tworzenie pakietu `.kbot`

1. Przygotuj katalog z zasobami, które mają zostać zaktualizowane, np.:

   ```text
   payload/
     models/
       regime_v3.bin
     strategies/
       breakout.json
   ```

2. Uruchom skrypt `scripts/package_update.py`, wskazując katalog źródłowy, plik
   wynikowy oraz podstawowe metadane pakietu:

   ```bash
   python scripts/package_update.py \
       payload/ build/releases/regime_update.kbot \
       --package-id regime-update \
       --version 1.0.0 \
       --fingerprint HWID-123456 \
       --metadata kanał=stable opis="Aktualizacja modeli" \
       --signing-key-file secrets/offline_update.key \
       --signing-key-id desk-prod
   ```

   * `--package-id` – unikalny identyfikator pakietu.
   * `--version` – wersja wyświetlana w UI.
   * `--fingerprint` – opcjonalne powiązanie z konkretnym urządzeniem.
   * `--metadata` – lista par `klucz=wartość` dodawanych do manifestu.
   * `--signing-key` lub `--signing-key-file` – klucz HMAC wykorzystywany do
     podpisania manifestu (`bot_core.security.signing`).
   * `--signing-key-id` – identyfikator klucza zapisywany w podpisie.

   Skrypt tworzy archiwum `.kbot` zawierające manifest (`manifest.json`),
   podpis (`manifest.sig`) oraz spakowaną zawartość (`payload.tar`).

## Import pakietu na stanowisku operatorskim

1. Przenieś plik `.kbot` na komputer docelowy i uruchom aplikację desktopową.
2. W menu ustawień otwórz dialog **Aktualizacja offline (.kbot)**.
3. Wskaż plik pakietu oraz – jeżeli to wymagane – wprowadź klucz podpisu HMAC.
4. Kliknij **Importuj**. Kontroler `OfflineUpdateController` zweryfikuje podpis,
   fingerprint oraz sumy kontrolne, a następnie rozpakowuje pakiet do katalogu
   `var/updates/packages/<id>-<version>`.
5. Po poprawnym imporcie w panelu aktualizacji pojawi się informacja o nowym
   pakiecie, a logi trafią do katalogu `logs/ui/updates/offline_update.log`
   tworzonego automatycznie przez `OfflineUpdateController`.

## Integracja z pipeline CI/CD

* Testy jednostkowe `tests/update/test_offline_updater.py` weryfikują integralność
  procesu budowania i importu pakietów.
* Do automatycznego publikowania paczek offline można wykorzystać powyższy
  skrypt w ramach kroku `release` w pipeline (np. tworząc artefakt `.kbot`).
* Zaleca się przechowywanie kluczy HMAC w `secrets/` lub bezpiecznym magazynie,
  aby nie trafiły do repozytorium.

## Dodatkowe wskazówki bezpieczeństwa

* Pakiet zostanie odrzucony, jeżeli fingerprint z manifestu różni się od
  oczekiwanego wartości urządzenia lub podpis kryptograficzny jest nieprawidłowy.
* Wygenerowany manifest zapisywany jest w katalogu docelowym wraz z sumą
  kontrolną artefaktów, dzięki czemu moduły C++ (OfflineUpdateManager) mogą
  korzystać z dotychczasowych mechanizmów walidacji.

