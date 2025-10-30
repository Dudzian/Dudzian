# Format pakietów aktualizacji offline

Pakiety aktualizacji dystrybuowane lokalnie zawierają obecnie:

1. `manifest.json` – metadane aktualizacji wraz z listą artefaktów,
2. pliki `payload` (`.tar`) oraz opcjonalne łatki różnicowe (`.patch`),
3. opcjonalny manifest integralności (lista docelowych plików z sumami SHA-256).

## Manifest

```json
{
  "id": "desktop-shell",
  "version": "2024.05",
  "platform": "linux-x86_64",
  "runtime": "bot-shell",
  "artifacts": [
    {
      "path": "payload.tar",
      "sha384": "…",
      "sha256": "…",
      "size": 10485760,
      "type": "full"
    },
    {
      "path": "delta.patch",
      "sha384": "…",
      "sha256": "…",
      "size": 524288,
      "type": "diff",
      "base_id": "desktop-shell@2024.04"
    }
  ],
  "integrity_manifest": {
    "path": "integrity.json",
    "sha256": "…"
  },
  "signature": {
    "algorithm": "hmac-sha384",
    "value": "base64…",
    "key_id": "oem-updates"
  }
}
```

* każdy artefakt musi posiadać hash `sha384` lub `sha256` (rekomendowane oba),
* pola `type=diff` sygnalizują łatki różnicowe (OfflineUpdateManager sprawdzi zgodność z wersją bazową),
* podpis HMAC chroni cały manifest – weryfikowany jest kluczem `BOT_CORE_UPDATE_HMAC_KEY`.

## Narzędzie CLI `scripts/update_package.py`

Budowa pakietu:

```bash
python3 scripts/update_package.py build \
  --output-dir var/updates/packages/demo \
  --package-id desktop-shell \
  --version 2024.05 \
  --platform linux-x86_64 \
  --runtime bot-shell \
  --payload build/payload.tar \
  --diff build/delta.patch \
  --base-id desktop-shell@2024.04 \
  --integrity-manifest build/integrity.json \
  --key "hex:…" --key-id oem-updates
```

Walidacja pakietu (włączana przez OfflineUpdateManager przed instalacją):

```bash
python3 scripts/update_package.py verify \
  --package-dir var/updates/packages/demo \
  --key "$BOT_CORE_UPDATE_HMAC_KEY"
```

CLI zwraca wynik w formacie JSON na STDOUT, co pozwala UI prezentować użytkownikowi pełny łańcuch audytu.

