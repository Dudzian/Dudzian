# Bezpieczeństwo i zgodność

Ten dokument opisuje, jak chronić klucze API, rotować je automatycznie i zapewniać zgodność z wymaganiami KYC/AML.

## Rotacja kluczy API

- Klucze przechowywane lokalnie szyfrowane są przez `bot_core.security.SecretManager` z backendem `EncryptedFileSecretStorage` (AES-256-GCM).
- Harmonogram rotacji prowadź w pliku `var/security/rotation.json` przy użyciu `bot_core.security.RotationRegistry`.
- Helper `KryptoLowca.security.KeyRotationManager` łączy magazyn sekretów z rejestrem rotacji i udostępnia metody `ensure_exchange_rotation()` oraz `ensure_secret_rotation()`, co upraszcza pisanie skryptów cron/CLI.
- Klucze przechowywane lokalnie szyfrowane są przez `bot_core.security.file_storage.EncryptedFileSecretStorage` (AES-256-GCM, PBKDF2 390k iteracji).
- `KeyRotationManager` zapisuje metadane rotacji w pliku `api_keys.enc.rotation.json`.
- Aby wymusić rotację raz na 30 dni, uruchom:

```bash
python -m KryptoLowca.scripts.rotate_keys --password <hasło>
```

(Przykładowy skrypt jest częścią roadmapy; w środowisku CI/cron możesz wywołać helper zapisujący wpis w `RotationRegistry`).

## Sekrety i skarbiec

`bot_core.security.SecretManager` ujednolica dostęp do sekretów:

| Backend | Zastosowanie | Uwagi |
| --- | --- | --- |
| `ENV` | Małe instalacje / development | Prosty magazyn implementujący interfejs `SecretStorage` |
| `EncryptedFileSecretStorage` | Stacje robocze bez dostępu do chmury | Szyfrowany plik `.enc` + metadane rotacji |
| `KeyringSecretStorage` | macOS / Windows / Linux z GUI | Wykorzystuje natywne API systemowe |
| Backend niestandardowy | Integracja z Vault/S3 | Dostarcz własną implementację `SecretStorage` |

Przykład użycia (plik):

```python
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import EncryptedFileSecretStorage, SecretManager
from KryptoLowca.security import KeyRotationManager

storage = EncryptedFileSecretStorage("/secure/secrets.enc", passphrase="<hasło>")
manager = SecretManager(storage, namespace="dudzian.trading")
manager.store_exchange_credentials(
    "binance",
    ExchangeCredentials(
        key_id="api-key",
        secret="sekret",
        passphrase=None,
        environment=Environment.PAPER,
        permissions=("trade", "read"),
    ),
)

creds = manager.load_exchange_credentials("binance", expected_environment=Environment.PAPER)
print(creds.key_id)

rotation = KeyRotationManager(manager)
rotation.ensure_exchange_rotation(
    "binance",
    expected_environment=Environment.PAPER,
    rotation_callback=lambda payload: payload,
)
```

## Compliance (KYC/AML)
- Flagi `CoreConfig.runtime_entrypoints[*].compliance` wraz z profilami ryzyka w `CoreConfig.risk_profiles` blokują tryb LIVE do czasu spełnienia wymagań.
- Każde naruszenie limitu ryzyka zapisuje się w tabeli `risk_audit_logs` oraz trafia do centralnych logów (Vector → Loki).
- Użytkownik powinien prowadzić dziennik decyzji (eksport z Grafany lub bazy `risk_audit_logs`) – to minimalny wymóg audytowy.

## Disaster Recovery

1. Konfiguracja oraz klucze trzymane są w wolumenie `/data` (podmontowany w Docker Compose).
2. Bazy SQLite (`trading.db`, `telemetry.sqlite`) należy backupować co najmniej raz dziennie.
3. Logi trafiają do katalogu `/app/logs` i są streamowane do Vector/Loki – pozwala to odtworzyć historię w razie awarii.

## Rekomendowane praktyki

- Zawsze testuj nowe wersje na koncie demo lub paper tradingu.
- Używaj oddzielnych kluczy API dla tradingu automatycznego, z włączonym limitem wypłat = 0.
- Rotuj hasła i klucze min. co 30 dni; stosuj unikalne hasła menedżerów haseł.
- Monitoruj logi alertów (Slack/email/webhook) – reaguj na naruszenia `reduce-only` i `cooldown`.
