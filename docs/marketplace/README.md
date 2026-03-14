# Marketplace – proces zgłaszania i recenzji presetów

Ten dokument opisuje kompletny przepływ dodawania nowych presetów do katalogu Marketplace, od przygotowania specyfikacji po marketing release'u.

## 1. Struktura repozytorium

- **`config/marketplace/presets/`** – źródłowe specyfikacje presetów. Każdy plik zawiera sekcję `preset` (ładunek instalacyjny) oraz sekcję `catalog` z metadanymi publikacji.
- **`config/marketplace/packages/`** – podpisane artefakty generowane z presetów (nie modyfikujemy ręcznie).
- **`config/marketplace/catalog.json`** – indeks publicznych paczek generowany automatycznie na podstawie specyfikacji.
- **`config/marketplace/catalog.md`** – podpisany (HMAC) katalog Markdown wykorzystywany przez marketing i support do szybkiego przeglądu person i budżetów.
- **`config/marketplace/keys/`** – klucze testowe do podpisów Ed25519 (presety) oraz HMAC (artefakty katalogu).
- **`config/marketplace/reviews/`** – podpisane recenzje community synchronizowane do klientów UI/HyperCare.

## 2. Przygotowanie zgłoszenia

1. Utwórz nowy plik w `config/marketplace/presets/<kategoria>/<nazwa>.json`.
2. Wypełnij sekcję `preset` (strategia, guardraile, metadane) oraz sekcję `catalog`:
   - `release.review_status`, `release.approved_at` i lista `reviewers` – kompletna ścieżka akceptacji.
   - `exchange_compatibility` – tablica obsługiwanych giełd z polami `status` i `last_verified_at`.
   - `versioning.source` – ścieżka do bieżącego pliku presetu (relatywnie względem katalogu `presets`).
   - `distribution[].signature.key_id` – identyfikator klucza HMAC używanego do podpisu artefaktu.
   - `metadata.user_preferences` – lista person (co najmniej jedna na strategię) z polami `persona`, `risk_target`, `recommended_budget`, `holding_period` i `notes`, aby UI/marketing mógł prezentować blur + FontAwesome.
3. Dołącz odnośniki do raportów QA/Stress Lab w `release_notes` lub `documentation_url`.

## 3. Generowanie katalogu i artefaktów

1. Uruchom lokalnie:
   ```bash
   python scripts/build_marketplace_catalog.py \
     --private-key config/marketplace/keys/dev-presets-ed25519.key \
     --key-id dev-presets \
     --signing-key dev-hmac:config/marketplace/keys/dev-hmac.key \
     --catalog-signature-key dev-hmac
   ```
2. Skrypt wykona następujące kroki:
   - Podpisze każdy preset Ed25519 i zapisze go do `config/marketplace/packages/...`.
   - Policzy sumy SHA-256 i podpisy HMAC zadeklarowane w `catalog.distribution[].signature`.
   - Zbuduje znormalizowany `config/marketplace/catalog.json` (schemat `1.1`) oraz `config/marketplace/catalog.md` wraz z plikami `.sig` dzięki opcji `--catalog-signature-key`.
   - Zweryfikuje, że katalog zawiera ≥15 strategii z kompletnymi metadanymi person (`user_preferences`).
3. Po uruchomieniu sprawdź `git status` – brak zmian oznacza, że artefakty są aktualne.
4. Po zmianach w `.gitattributes` (szczególnie na Windows/CI) wykonaj jednorazowo `git add --renormalize .`, aby working tree został przeliczony według nowych reguł EOL i nie utrzymywał starych CRLF.

## 4. Bundlowanie release'u

1. Uruchom `python scripts/build_release_bundle.py --installer-root deploy/packaging/samples` aby zsynchronizować `config/marketplace/catalog.*` oraz całe `packages/**` z paczkami instalatora (katalog `config/marketplace` w bundlu). Skrypt weryfikuje przy tym, że katalog zawiera co najmniej 15 strategii z recenzjami QA (`release.reviewers` z rolą `QA`/`quality`).
2. Flagi `--check-only --require-clean` blokują merge, gdy `config/marketplace/catalog.md(.sig)` różnią się od wersji w repozytorium lub brakuje recenzji QA – dokładnie w ten sposób działa krok „Validate release bundle catalog” w jobie `marketplace-catalog` workflowa `.github/workflows/ci.yml`.
3. Job `deploy` tego samego workflowa ponawia budowę katalogu, uruchamia `build_release_bundle.py` dla katalogu `deploy/packaging/samples` i publikuje artefakt `marketplace-catalog-release`, który można dołączyć do release notes.

## 4a. Rollout presetów (produkcyjny)

1. **Podpisz i zbuduj katalog**:
   ```bash
   python scripts/build_marketplace_catalog.py \
     --private-key config/marketplace/keys/dev-presets-ed25519.key \
     --key-id dev-presets \
     --signing-key dev-hmac:config/marketplace/keys/dev-hmac.key \
     --catalog-signature-key dev-hmac
   ```
2. **Zweryfikuj podpisy katalogu i presetów** (CLI używany także w CI):
   ```bash
   python scripts/validate_marketplace_presets.py \
     --presets config/marketplace/presets \
     --hmac-key config/marketplace/keys/dev-hmac.key \
     --ed25519-key config/marketplace/keys/dev-presets-ed25519.pub \
     --catalog config/marketplace/catalog.json \
     --catalog-markdown config/marketplace/catalog.md
   ```
   Walidator sprawdza podpisy HMAC/Ed25519 dla `catalog.json` i `catalog.md` oraz kompletność metadanych QA (`release.review_status`, `release.reviewers`, `exchange_compatibility`).
3. **Opublikuj artefakty**: dołącz `config/marketplace/catalog.json(.sig)` i `config/marketplace/catalog.md(.sig)` do bundla instalatora (`build_release_bundle.py`) oraz pipeline’u marketingowego (`marketplace-catalog` artifact `marketplace-catalog-release`).
4. **Synchronizacja u klientów**: job `marketplace_cli.py sync` (lub manualny `python scripts/marketplace_cli.py sync --source <URL>`) aktualizuje katalog i recenzje w instalacjach OEM; UI PySide6 weryfikuje podpisy katalogu przed załadowaniem presetów.

## 5. Recenzje community

1. Każdy preset może mieć dowolną liczbę recenzji w plikach `config/marketplace/reviews/<preset_id>.json`. Recenzje muszą być podpisane HMAC kluczem `dev-hmac` (lub produkcyjnym) – podpisujemy je poleceniem `python scripts/ui_marketplace_bridge.py submit-review --preset-id <id> --rating <1-5> --comment "..." --review-key-id dev-hmac` (CLI sam dopisze wpis w repozytorium i odświeży `.meta/reviews.json`).
2. Po zmianach w katalogu recenzji należy zsynchronizować je lokalnie i w UI: `python scripts/ui_marketplace_bridge.py --presets-dir config/marketplace/presets --signing-key dev-hmac=$(cat config/marketplace/keys/dev-hmac.key) sync-reviews --source-dir config/marketplace/reviews`.
3. Pipeline `marketplace-catalog` oraz ręczne wydania kończą się krokiem `python scripts/marketplace_cli.py sync --source config/marketplace/catalog.json --force`, aby klienci OEM pobrali świeży katalog + recenzje.

## 6. Walidacja QA

- **Walidator CLI**: `python scripts/marketplace_cli.py validate --key dev-hmac:config/marketplace/keys/dev-hmac.key` – weryfikuje podpisy, fingerprinty i metadane release.
- **Test regresyjny**: `pytest tests/test_marketplace_catalog.py` – kontroluje skróty, podpisy oraz obecność źródeł `versioning.source`.
- **Workflow CI**: zadanie `marketplace-catalog` w `.github/workflows/ci.yml` powtarza powyższe kroki, uruchamia `build_release_bundle.py --check-only --require-clean` i publikuje artefakt `marketplace-packages`.
- **Automatyzacja release'u**: job `deploy` w tym samym workflowie kopiuje katalog i paczki do bundla instalatora oraz publikuje podpisane `catalog.json`/`catalog.md` jako artefakt `marketplace-catalog-release`.
- **Przegląd marketingowy**: upewnij się, że `config/marketplace/catalog.md` oraz `config/marketplace/catalog.md.sig` są zaktualizowane i odzwierciedlają pełen zestaw ≥15 strategii z personami.

## 7. Publikacja i marketing

1. Po merge'u pipeline CI wygeneruje podpisane paczki i zaktualizowany katalog.
2. Runbook marketingowy (`docs/runbooks/marketplace_marketing.md`) opisuje wymagane materiały (komunikat release, listing w newsletterze, aktualizacja porównania CryptoHopper).
3. Po wydaniu zsynchronizuj katalog u klientów: `python scripts/marketplace_cli.py sync`.

## 8. Najczęstsze problemy

| Problem | Diagnoza | Rozwiązanie |
| --- | --- | --- |
| Brak `last_verified_at` dla statusu `certified` | Walidator zakończy się błędem | Uzupełnij timestamp potwierdzający testy funkcjonalne |
| `validate` zgłasza brak `versioning.source` | Spec nie wskazuje ścieżki źródłowej | Ustaw relatywną ścieżkę (np. `strategies/mean_reversion_v1.json`) |
| `marketplace-catalog` zmienia katalog w CI | Ręcznie edytowano `catalog.json` | Uruchom `build_marketplace_catalog.py` lokalnie i dołącz wynik w PR |
| `build_release_bundle.py --check-only --require-clean` zwraca błąd „Zmodyfikowany katalog Marketplace” | Wygenerowane artefakty `config/marketplace/catalog.md(.sig)` różnią się od tego, co jest zatwierdzone w repo (guard fail-fast w CI) | Uruchom ponownie `build_marketplace_catalog.py`, sprawdź `git diff -- config/marketplace/catalog.md config/marketplace/catalog.md.sig`, a następnie dodaj i zatwierdź zaktualizowane `catalog.json(.sig)`, `catalog.md(.sig)` i `config/marketplace/packages/**`; jeśli różnice dotyczą tylko EOL na Windowsie, wykonaj `git add --renormalize .` |

---
W razie pytań kontakt: `marketplace@example.com` (Marketplace Guild).

## Utrzymanie presetów giełdowych (spec-hash)

Przed commitem (lub po zmianach w `config/exchanges` / logice generatora) zregeneruj podpisane presety giełdowe:

1. Zainstaluj zależności (jeśli nie są dostępne): `pip install -r requirements.txt`.
2. Uruchom rekonsyliację z domyślnymi ścieżkami repozytorium:  
   `python scripts/reconcile_exchange_presets.py`
3. Przejrzyj zmiany: `git status` (katalog `config/marketplace/presets/exchanges` powinien zostać zaktualizowany).
4. Zweryfikuj tylko ten test:  
   `pytest tests/marketplace/test_exchange_presets_repository.py::test_committed_exchange_presets_are_signed_and_current`
5. Jeśli test przejdzie, dołącz wygenerowane presety do commita/PR.
