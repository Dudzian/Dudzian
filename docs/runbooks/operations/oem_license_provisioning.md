# Runbook: Provisioning licencji OEM

## Kontekst

Moduł `bot_core.security.fingerprint` udostępnia deterministyczny fingerprint sprzętowy
podpisany HMAC-em z kluczami rotowanymi w `RotationRegistry`. Fingerprint jest używany
do generowania licencji OEM zapisywanych w `var/licenses/registry.jsonl`. Proces
provisioning musi działać na Windows, macOS i Linux oraz obsługiwać zarówno przesyłanie
fingerprintu w postaci QR, jak i na nośniku USB.

## Generowanie fingerprintu na stanowisku klienta

1. Upewnij się, że plik `config/oem_fingerprint_keys.json` zawiera aktywny klucz w formacie:

   ```json
   { "keys": { "fp-key": "hex:0123..." } }
   ```

2. W aplikacji desktopowej otwórz zakładkę **Aktywacja** (przycisk w górnym pasku).
   Komponent korzysta z `ActivationController`, który wykonuje polecenie
   `python -m bot_core.security.fingerprint --key fp-key=… --rotation-log var/licenses/fingerprint_rotation.json`.

3. W widoku aktywacji pojawią się:
   - fingerprint SHA-256,
   - komponenty CPU/TPM/MAC/dongle z wartościami RAW i digest,
   - lista lokalnych licencji (jeśli istnieją).

4. Fingerprint można wyeksportować do pliku JSON (np. na pendrive) lub zeskanować kodem QR
   wygenerowanym z zawartości JSON (base64).

## Provisioning po stronie OEM

Do wystawienia licencji użyj narzędzia `python scripts/oem_provision_license.py`.

### Wniosek licencyjny z pliku JSON/YAML

Operator może przygotować kompletny wniosek w pliku (np. `request.json` lub
`request.yaml`) i przekazać go jako pierwszy argument polecenia. Przykład
minimalnego pliku JSON:

```json
{
  "fingerprint": "ABCD-EFGH",
  "issuer": "QA-DEPARTMENT",
  "profile": "paper",
  "bundle_version": "1.2.3",
  "features": ["daemon", "ui"],
  "valid_days": 30,
  "signing_key_path": "var/licenses/signing.key",
  "key_id": "lic-2024",
  "output": "var/licenses/registry.jsonl",
  "rotation_log": "var/licenses/license_rotation.json"
}
```

Polecenie uruchomione jako `python scripts/oem_provision_license.py request.json`
zachowuje się tak, jak gdyby wszystkie wartości zostały podane wprost jako
flagi CLI. Parametry przekazane jednocześnie w pliku i poprzez flagi są
nadpisywane przez flagi (np. `--issuer`). Plik YAML może wykorzystywać te same
klucze (np. `bundle_version: 1.2.3`).

### Walidacja fingerprintu i wydanie licencji (USB)

```bash
python scripts/oem_provision_license.py \
  --fingerprint /media/OEM/FINGERPRINT.json \
  --mode usb \
  --fingerprint-key fp-key=hex:0123... \
  --license-key lic-2024=hex:abcd... \
  --license-rotation-log var/licenses/license_rotation.json \
  --registry var/licenses/registry.jsonl \
  --usb-output /media/OEM/Licenses
```

Skrypt:
- weryfikuje podpis fingerprintu,
- sprawdza politykę OEM (CPU z allow-listy, obecność MAC, TPM i dongla w trybie USB),
- podpisuje licencję kluczem `lic-2024`,
- dopisuje wpis JSONL do `var/licenses/registry.jsonl`,
- zapisuje artefakt licencyjny w katalogu USB (nazwany `LICENSE_ID.json`).

### Tryb QR

Jeżeli fingerprint został dostarczony jako ciąg base64 (np. skan QR):

```bash
python scripts/oem_provision_license.py \
  --fingerprint 'eyJwYXlsb2FkIjog...==' \
  --mode qr \
  --fingerprint-key fp-key=... \
  --license-key lic-2024=...
```

Skrypt wypisze na stdout zakodowany payload licencji, który można przekazać do generatora QR.

### Walidacja rejestru

CI oraz operatorzy mogą zweryfikować spójność podpisów:

```bash
python scripts/oem_provision_license.py \
  --registry var/licenses/registry.jsonl \
  --license-key lic-2024=... \
  --fingerprint-key fp-key=... \
  --validate-registry
```

Polecenie zwraca kod wyjścia `0`, jeśli wszystkie wpisy mają poprawne podpisy HMAC.

## Integracja z CI i UI

- Workflow `deploy/ci/github_actions_paper_smoke.yml` wykonuje `--validate-registry`
  z kluczem przekazanym jako `secrets.OEM_LICENSE_KEY`. Walidacja jest pomijana, gdy
  rejestr nie istnieje lub sekret nie został zdefiniowany.
- UI Qt udostępnia dialog aktywacji (`ActivationDialog`) z listą licencji i możliwością
  odświeżenia fingerprintu. Dane są pobierane poprzez `ActivationController`, który
  uruchamia moduł fingerprintu i wczytuje JSONL z rejestru.

## Rotacja kluczy

- Fingerprint: rotacja kontrolowana wpisami w `var/licenses/fingerprint_rotation.json`
  (domyślna długość życia 180 dni). Klucze przekazywane przez `--key` muszą być
  aktualizowane zgodnie z polityką bezpieczeństwa.
- Licencje: rejestr rotacji w `var/licenses/license_rotation.json`. Skrypt provisioning
  wykorzystuje najświeższy klucz nieprzeterminowany, a brak wpisu traktuje jako wymagającą
  natychmiastową rotację.

## Diagnostyka

- `ActivationController` loguje błędy (kategoria `bot.shell.activation`).
- Skrypt provisioning zwraca kod `2` i komunikat na stderr w razie naruszenia polityki OEM
  lub braku kluczy. Błędy JSONL sygnalizowane są z numerem linii.
- W razie potrzeby użyj `python -m bot_core.security.fingerprint --help` oraz
  `python scripts/oem_provision_license.py --help`.

## Sprzątanie raportów w UI

Panel desktopowy udostępnia zakładkę **Raporty** (przycisk w górnym pasku). Widok
korzysta z `ReportCenterController`, który komunikuje się z mostkiem CLI
`bot_core.reporting.ui_bridge`. Operator może odświeżyć listę artefaktów z katalogu
`var/reports`, sprawdzić metadane (ścieżka, rozmiar, liczba plików) oraz usunąć
zaznaczony raport. Usunięcie wymaga potwierdzenia w oknie dialogowym i automatycznie
odświeża listę. Mostek waliduje, że wskazana ścieżka znajduje się w katalogu raportów,
co zapobiega przypadkowemu usunięciu niepowiązanych plików.

