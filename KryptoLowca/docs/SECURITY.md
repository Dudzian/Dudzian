# Bezpieczeństwo i zgodność

Ten dokument opisuje, jak chronić klucze API, rotować je automatycznie i zapewniać zgodność z wymaganiami KYC/AML.

## Rotacja kluczy API

- Klucze przechowywane lokalnie szyfrowane są przez `bot_core.security.file_storage.EncryptedFileSecretStorage` (AES-256-GCM, PBKDF2 390k iteracji).
- `KeyRotationManager` zapisuje metadane rotacji w pliku `api_keys.enc.rotation.json`.
- Aby wymusić rotację raz na 30 dni, uruchom:

```bash
python -m KryptoLowca.scripts.rotate_keys --password <hasło>
```

(Przykładowy skrypt jest częścią roadmapy; w środowisku CI/cron można wywołać klasę `KeyRotationManager.ensure_rotation`).

## Sekrety i skarbiec

`SecretManager` ujednolica dostęp do sekretów:

| Backend | Zastosowanie | Uwagi |
| --- | --- | --- |
| `ENV` | Małe instalacje / development | Sekrety trzymane w zmiennych środowiskowych |
| `FILE` | Stacje robocze bez dostępu do chmury | Plik `.secrets.json` + metadane rotacji |
| `VAULT` | HashiCorp Vault | Wymagany pakiet `hvac` i token z polityką RW |
| `AWS` | AWS Secrets Manager | Wymaga `boto3` oraz uprawnień `secretsmanager:*` |

Przykład użycia (plik):

```python
from KryptoLowca.security import SecretManager, SecretBackend

manager = SecretManager(backend=SecretBackend.FILE, file_path="/secure/secrets.json")
manager.set_secret("BINANCE_API_KEY", "...")
print(manager.get_secret("BINANCE_API_KEY"))
```

## Compliance (KYC/AML)

- Flagi `StrategyConfig` (`api_keys_configured`, `compliance_confirmed`, `acknowledged_risk_disclaimer`) blokują tryb LIVE do czasu spełnienia wymagań.
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
