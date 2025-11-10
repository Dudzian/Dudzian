# Marketplace – proces zgłaszania i recenzji presetów

Ten dokument opisuje kompletny przepływ dodawania nowych presetów do katalogu Marketplace, od przygotowania specyfikacji po marketing release'u.

## 1. Struktura repozytorium

- **`config/marketplace/presets/`** – źródłowe specyfikacje presetów. Każdy plik zawiera sekcję `preset` (ładunek instalacyjny) oraz sekcję `catalog` z metadanymi publikacji.
- **`config/marketplace/packages/`** – podpisane artefakty generowane z presetów (nie modyfikujemy ręcznie).
- **`config/marketplace/catalog.json`** – indeks publicznych paczek generowany automatycznie na podstawie specyfikacji.
- **`config/marketplace/keys/`** – klucze testowe do podpisów Ed25519 (presety) oraz HMAC (artefakty katalogu).

## 2. Przygotowanie zgłoszenia

1. Utwórz nowy plik w `config/marketplace/presets/<kategoria>/<nazwa>.json`.
2. Wypełnij sekcję `preset` (strategia, guardraile, metadane) oraz sekcję `catalog`:
   - `release.review_status`, `release.approved_at` i lista `reviewers` – kompletna ścieżka akceptacji.
   - `exchange_compatibility` – tablica obsługiwanych giełd z polami `status` i `last_verified_at`.
   - `versioning.source` – ścieżka do bieżącego pliku presetu (relatywnie względem katalogu `presets`).
   - `distribution[].signature.key_id` – identyfikator klucza HMAC używanego do podpisu artefaktu.
3. Dołącz odnośniki do raportów QA/Stress Lab w `release_notes` lub `documentation_url`.

## 3. Generowanie katalogu i artefaktów

1. Uruchom lokalnie:
   ```bash
   python scripts/build_marketplace_catalog.py \
     --private-key config/marketplace/keys/dev-presets-ed25519.key \
     --key-id dev-presets \
     --signing-key dev-hmac:config/marketplace/keys/dev-hmac.key
   ```
2. Skrypt wykona następujące kroki:
   - Podpisze każdy preset Ed25519 i zapisze go do `config/marketplace/packages/...`.
   - Policzy sumy SHA-256 i podpisy HMAC zadeklarowane w `catalog.distribution[].signature`.
   - Zbuduje znormalizowany `config/marketplace/catalog.json` (schemat `1.1`).
3. Po uruchomieniu sprawdź `git status` – brak zmian oznacza, że artefakty są aktualne.

## 4. Walidacja QA

- **Walidator CLI**: `python scripts/marketplace_cli.py validate --key dev-hmac:config/marketplace/keys/dev-hmac.key` – weryfikuje podpisy, fingerprinty i metadane release.
- **Test regresyjny**: `pytest tests/test_marketplace_catalog.py` – kontroluje skróty, podpisy oraz obecność źródeł `versioning.source`.
- **Workflow CI**: zadanie `marketplace-catalog` w `.github/workflows/ci.yml` powtarza powyższe kroki i publikuje artefakt `marketplace-packages`.

## 5. Publikacja i marketing

1. Po merge'u pipeline CI wygeneruje podpisane paczki i zaktualizowany katalog.
2. Runbook marketingowy (`docs/runbooks/marketplace_marketing.md`) opisuje wymagane materiały (komunikat release, listing w newsletterze, aktualizacja porównania CryptoHopper).
3. Po wydaniu zsynchronizuj katalog u klientów: `python scripts/marketplace_cli.py sync`.

## 6. Najczęstsze problemy

| Problem | Diagnoza | Rozwiązanie |
| --- | --- | --- |
| Brak `last_verified_at` dla statusu `certified` | Walidator zakończy się błędem | Uzupełnij timestamp potwierdzający testy funkcjonalne |
| `validate` zgłasza brak `versioning.source` | Spec nie wskazuje ścieżki źródłowej | Ustaw relatywną ścieżkę (np. `strategies/mean_reversion_v1.json`) |
| `marketplace-catalog` zmienia katalog w CI | Ręcznie edytowano `catalog.json` | Uruchom `build_marketplace_catalog.py` lokalnie i dołącz wynik w PR |

---
W razie pytań kontakt: `marketplace@example.com` (Marketplace Guild).
