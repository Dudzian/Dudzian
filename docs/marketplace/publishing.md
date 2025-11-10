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

Nowy proces bazuje na skrypcie `scripts/build_marketplace_catalog.py`, który podpisuje wszystkie presety oraz odświeża katalog w jednym kroku.

1. Upewnij się, że sekcja `catalog.distribution` w specyfikacji zawiera `uri`, `signature.key_id` oraz opis artefaktu.
2. Uruchom:
   ```bash
   python scripts/build_marketplace_catalog.py \
     --private-key config/marketplace/keys/dev-presets-ed25519.key \
     --key-id dev-presets \
     --signing-key dev-hmac:config/marketplace/keys/dev-hmac.key
   ```
3. Skrypt zapisze podpisane dokumenty w `config/marketplace/packages/…`, policzy sumy SHA-256 i uzupełni podpis HMAC (`signature.value`).
4. Wygenerowany `config/marketplace/catalog.json` będzie miał schemat `1.1` oraz zaktualizowane znaczniki `release`, `exchange_compatibility` i `versioning`.
5. Zweryfikuj wynik `git status` – brak zmian po uruchomieniu oznacza, że repozytorium jest zsynchronizowane z katalogiem publicznym.

## 3. Walidacja lokalna

CLI `scripts/marketplace_cli.py` udostępnia pomocnicze komendy:

```bash
# Lista paczek
python scripts/marketplace_cli.py list

# Podgląd pełnych metadanych
python scripts/marketplace_cli.py show mean_reversion.v1

# Walidacja podpisów, fingerprintów oraz metadanych release/versioning
python scripts/marketplace_cli.py validate --key dev-hmac:config/marketplace/keys/dev-hmac.key
```

Walidator (`bot_core.security.marketplace_validator.MarketplaceValidator`) sprawdza:

- zgodność skrótu artefaktu z deklaracją `integrity`;
- podpis HMAC względem podanego klucza (`key_id` -> ścieżka);
- dopasowanie fingerprintu sprzętowego (`HwIdProvider`).
- poprawność sekcji `release` (zatwierdzenia) i `versioning` (referencje `package@version`).

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
