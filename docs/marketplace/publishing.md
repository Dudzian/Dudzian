# Publikacja paczek Marketplace

Dokument opisuje referencyjny workflow przygotowywania i publikacji paczek w Marketplace konfiguracji. Proces obejmuje walidację metadanych, budowanie artefaktów, podpisy kryptograficzne, testy jakości (w tym eksperymenty A/B) oraz finalne wydanie w repozytorium `config/marketplace/`.

## 1. Przygotowanie metadanych

Każda paczka opisująca konfigurację korzysta ze schematu `bot_core.config_marketplace.schema.MarketplacePackageMetadata`. Najważniejsze pola:

- **`schema_version`** – wersja schematu (obecnie `1.0`). Aktualizuj przy zmianie formatu.
- **`version`** oraz opcjonalny **`revision`** – wersjonowanie zgodne z SemVer. Używamy `MAJOR.MINOR.PATCH` + dodatkowy identyfikator builda.
- **`license`** – obiekt `LicenseInfo` z nazwą, identyfikatorem SPDX, flagami `redistributable` oraz `commercial_use`.
- **`data_requirements`** – lista `DataAssetRequirement` opisująca strumienie danych, retencję, dostęp i uwagi dot. danych wrażliwych.
- **`distribution`** – artefakty (np. paczka konfiguracyjna) z sekcjami `integrity` (algorytm + digest) i `signature` (HMAC z `key_id`).
- **`security`** – polityka fingerprintu sprzętowego (`mode`, `allowed_fingerprints`, `require_strict_match`).

Metadane zapisujemy w katalogu `config/marketplace/catalog.json`. Do repozytorium dodajemy również fizyczne artefakty (np. `packages/<id>/bundle.json`).

## 2. Budowanie paczki

1. Utwórz katalog `config/marketplace/packages/<package_id>/` i umieść w nim spakowaną konfigurację.
2. Oblicz skrót SHA (zalecany `sha256`).
3. Zbuduj podpis HMAC:
   ```bash
   python - <<'PY'
   import base64, hashlib, hmac, json
   from pathlib import Path
   payload = {
       "package_id": "mean_reversion.v1",
       "version": "1.0.0",
       "artifact": "config-bundle",
       "uri": "packages/mean_reversion/bundle.json",
       "sha256": "<DIGEST>"
   }
   key = Path("config/marketplace/keys/dev-hmac.key").read_bytes()
   mac = hmac.new(key, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256)
   print(base64.b64encode(mac.digest()).decode())
   PY
   ```
4. Uzupełnij sekcję `signature` w katalogu (`key_id`, `algorithm`, `value`, `signed_fields`).

## 3. Walidacja lokalna

CLI `scripts/marketplace_cli.py` udostępnia pomocnicze komendy:

```bash
# Lista paczek
python scripts/marketplace_cli.py list

# Podgląd pełnych metadanych
python scripts/marketplace_cli.py show mean_reversion.v1

# Walidacja podpisów i fingerprintów
python scripts/marketplace_cli.py validate --key dev-hmac:config/marketplace/keys/dev-hmac.key
```

Walidator (`bot_core.security.marketplace_validator.MarketplaceValidator`) sprawdza:

- zgodność skrótu artefaktu z deklaracją `integrity`;
- podpis HMAC względem podanego klucza (`key_id` -> ścieżka);
- dopasowanie fingerprintu sprzętowego (`HwIdProvider`).

## 4. Testy jakości i eksperymenty A/B

1. **Walidacja funkcjonalna** – uruchom zestawy testów jednostkowych/integrowanych dla konfiguracji w środowisku staging (np. `pytest tests/marketplace/<package_id>`).
2. **Eksperyment A/B** – przygotuj scenariusz porównawczy:
   - Grupa kontrolna: aktualnie wdrożona konfiguracja.
   - Grupa eksperymentalna: nowa paczka.
   - Monitoruj metryki (np. Sharpe, max drawdown, latency) co najmniej przez 48h.
   - Zapisz raport w `docs/marketplace/reports/<package_id>/<version>/ab-test.md` i podlinkuj w `release_notes`.
3. **Walidacja bezpieczeństwa** – potwierdź, że klucze HMAC są rotowane, a fingerprinty obejmują tylko autoryzowany sprzęt.

## 5. Publikacja

1. Zaktualizuj `config/marketplace/catalog.json` (dodaj/edytuj wpis paczki, ustaw `generated_at`).
2. Upewnij się, że repozytorium posiada aktualny `repository.json` z adresem zdalnego katalogu.
3. Wygeneruj paczkę dystrybucyjną (np. `tar.gz`) wraz z podpisem.
4. Wdróż katalog na serwer dystrybucyjny (`remote_index_url`).
5. Po publikacji wykonaj `python scripts/marketplace_cli.py sync` na środowisku docelowym i potwierdź, że UI prezentuje kartę w zakładce **Marketplace**.

## 6. Najlepsze praktyki

- Zachowuj zgodność wersji `schema_version` – zmiany w schemacie wprowadzaj wraz z migracją katalogu.
- Przechowuj klucze HMAC w bezpiecznym magazynie (`secrets/` lub zewnętrzny KMS). W plikach repozytorium trzymaj wyłącznie klucze testowe.
- Uzupełniaj `release_notes` z odnośnikiem do raportów testów A/B i dokumentacji.
- Używaj `security.allowed_fingerprints` do ograniczenia instalacji na sprzęcie kontrolowanym przez organizację. W przypadku paczek otwartych ustaw `mode: "none"`.
- Regularnie uruchamiaj walidator i testy regresyjne przed każdą publikacją.

## 7. Integracja z UI

Zakładka **Marketplace** (`ui/qml/components/MarketplaceView.qml`) odczytuje lokalny katalog i wyświetla karty ofert. Po każdej synchronizacji CLI odświeża dane (widok automatycznie odczytuje zmiany co minutę lub po kliknięciu „Odśwież”). Zdarzenie `packageActivated` można powiązać z pipeline instalacji (np. wywołaniem CLI `marketplace_cli.py` w trybie `sync`/`validate`).

---
W razie pytań kontaktuj się z zespołem Marketplace Guild (`marketplace@example.com`).
