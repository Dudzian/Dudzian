# OEM License Provisioning – Runbook

## Cel
Zapewnienie w pełni offline'owego procesu wydawania licencji dla bundla Core OEM z kontrolą podpisów HMAC, rotacją kluczy i archiwizacją artefaktów.

## Checklista operacyjna
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| Przygotuj klucz HMAC (`signing.key`) ≥32 bajty oraz zaktualizuj rejestr rotacji (`var/licenses/key_rotation.json`). | Bezpieczeństwo | `signing.key`, `var/licenses/key_rotation.json` | Security Lead |
| Zweryfikuj status rotacji (`python scripts/oem_provision_license.py --rotation-log ... --key-id ... --valid-days ... --fingerprint ... --signing-key-path ... --dry-run`*) i potwierdź, że klucz nie jest przeterminowany. | Bezpieczeństwo | Log skryptu | Security Lead |
| Uruchom `python scripts/oem_provision_license.py` z docelowym fingerprintem i parametrami profilu. | Operator OEM | `var/licenses/registry.jsonl`, logi CLI | OEM Ops |
| (opcjonalnie) Wygeneruj kod QR (`--emit-qr`) i wydruk papierowy dla klienta. | Operator OEM | Wydruk QR | OEM Ops |
| (opcjonalnie) Wyeksportuj licencję na nośnik (`--usb-target /media/.../license.json`). | Operator OEM | Plik licencyjny na USB | OEM Ops |
| Zaimportuj licencję w kliencie Qt (ekran aktywacji) i zweryfikuj fingerprint/profil. | Operator OEM | Zrzut ekranu UI, `var/licenses/active/license.json` | Product Owner |
| Załącz artefakty do biletu provisioningowego (hash licencji, podpisane logi, numer zamówienia). | Operator OEM | Hash SHA-384, log CLI | Product Owner |
| Zaktualizuj decision log (`var/licenses/registry.jsonl`, `audit/decision_logs/*.jsonl`) podpisem i odnotuj kto zatwierdził wydanie. | Operator OEM | Decision log | Compliance |

\* *Tryb dry-run*: ustaw `--no-mark-rotation`, aby wygenerować podpis bez zapisu licencji (służy do testów).* 

## Parametry skryptu `python scripts/oem_provision_license.py`
- `--signing-key-path`: klucz HMAC (min. 32 bajty, najlepiej 48 bajtów SHA-384).
- `--key-id`: identyfikator klucza wykorzystywany w rotacji i audycie.
- `--fingerprint` lub `--fingerprint-file`: źródło fingerprintu klienta; jeśli brak parametrów, narzędzie wygeneruje fingerprint lokalnej maszyny.
- `--valid-days`: ważność licencji (domyślnie 365 dni); dla licencji próbnych ustaw np. `30`.
- `--feature`: wielokrotny parametr określający odblokowane moduły (np. `daemon`, `ui`, `paper-labs`).
- `--emit-qr`: drukuje kod QR w terminalu (wymaga opcjonalnej biblioteki `qrcode`).
- `--usb-target`: zapisuje licencję na wskazanym nośniku (np. pamięć USB air-gapped).
- `--rotation-log`: ścieżka do logu rotacji klucza podpisującego licencje.
- `--no-mark-rotation`: pomija aktualizację logu rotacji (np. w testach).

## Walidacja po provisioning
1. Otwórz `var/licenses/registry.jsonl` i upewnij się, że nowa linia zawiera oczekiwany fingerprint, profil i daty `issued_at`/`expires_at`.
2. Zweryfikuj podpis HMAC (`python -m scripts.oem_provision_license --fingerprint ... --signing-key-path ... --no-mark-rotation` + `jq`/`python` do przeliczenia hashy).
3. Sprawdź log rotacji (`var/licenses/key_rotation.json`) i potwierdź zaktualizowaną datę.
4. (Jeśli użyto QR/USB) wykonaj testowe importy licencji po stronie klienta i dołącz zrzuty ekranów do biletu.
5. Zaktualizuj runbook instalacyjny (`OEM_INSTALLER_ACCEPTANCE.md`) o numer licencji i fingerprint urządzenia.

## Self-service instalacji presetów Marketplace
Operatorzy i zespoły wsparcia mogą w pełni samodzielnie przeprowadzić instalację
pakietu Marketplace na stanowisku klienta (offline), korzystając z nowego
workflow CLI/UI:

1. Przygotuj artefakty:
   - plik licencji OEM (`licenses/<preset_id>.json`) z wygenerowanego procesu
     provisioning (sekcja powyżej);
   - katalog manifestu Marketplace (np. podpisany pakiet OEM zawierający
     `catalog.yaml` i folder `presets/` lub `artifacts/`).
2. Uruchom komendę instalacyjną, podając ścieżki do repozytorium presetów,
   indeksu licencji oraz docelowego portfela:

   ```bash
   python scripts/ui_marketplace_bridge.py \
       --presets-dir data/strategies \
       --licenses-path var/marketplace_licenses.json \
       --fingerprint DEVICE-XYZ \
       --signing-key=catalog=$(cat config/marketplace/keys/catalog.hex) \
       install \
       --preset-id automation-ai \
       --portfolio-id master-portfolio \
       --license-json /media/offline/licenses/automation-ai.json \
       --licenses-dir var/licenses/presets \
       --catalog-path /media/offline/catalog \
       --preferences-json docs/samples/preferences_balanced.json
   ```

   Pole `--preferences-json` przyjmuje budżet i target ryzyka użytkownika, które
   zostaną zmapowane na parametry strategii oraz zapisane w
   `.meta/preferences.json` (np. maksymalna liczba pozycji, mnożnik ryzyka).
3. Polecenie generuje raport JSON zawierający wynik weryfikacji podpisu,
   ścieżki do zaktualizowanych store'ów (`assignments`, `preferences`, indeks
   licencji) oraz listę przypisanych portfeli. Raport należy załączyć do biletu
   hypercare lub dokumentacji audytowej.
4. Po instalacji można zweryfikować przydziały i licencje z poziomu UI (zakładka
   Marketplace) – backend korzysta z tych samych store'ów, więc operacja CLI nie
   wymaga restartu aplikacji.

## Artefakty wyjściowe
- `var/licenses/registry.jsonl` – podpisane dokumenty licencyjne.
- `var/licenses/key_rotation.json` – log rotacji klucza HMAC.
- `logs/oem_provisioning/*.log` – logi CLI (jeśli włączono redirect).
- Wydruk QR lub plik `license.json` z nośnika offline (USB/dysk szyfrowany).

## Checklisty bezpieczeństwa
- Klucz HMAC przechowywany w sejfie offline, używany tylko na stacji air-gapped.
- Weryfikacja, że fingerprint klienta pochodzi z zaufanego kanału (np. podpisany request JSON).
- Po provisioning usuń tymczasowe pliki i odmontuj nośniki USB.
- Wpisz operację do decision log wraz z identyfikatorem operatora i czasem wykonania.
