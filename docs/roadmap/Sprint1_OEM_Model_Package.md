# Sprint 1 – "Pakiet modeli OEM gotowy do inference"

## Cel sprintu
Dostarczyć komplet artefaktów modeli AI (manifesty, sumy kontrolne, podpisy), zintegrowanych z `ai_models/` i `bot_core.ai_models`, aby domyślna instalacja bota wykonywała inference na produkcyjnych modelach, a nie w trybie fallback.

## Założenia wstępne
- Modele OEM są trenowane poza repozytorium (pipeline MLOps out-of-tree) i publikowane jako spakowane bundla `.tar.zst`.
- Każdy pakiet modeli zawiera co najmniej: `manifest.json`, katalog `artifacts/` (pliki wag, tokenizery, metadane) oraz podpis `manifest.sig` (Ed25519 + SHA-384).
- Klucz weryfikacyjny podpisów (`oem_model_signing.pub`) trafia do repozytorium w katalogu `ai_models/keys/`.
- Wymuszamy flagę środowiskową `BOT_CORE_REQUIRE_OEM_MODELS=true` na wszystkich buildach CI.

## Struktura prac
### Epik A – Pipeline bundlowania modeli
1. **Repozytorium artefaktów**
   - Przygotować katalog `ai_models/bundles/` z `.gitkeep` i strukturą pod wersje (`vX_Y_Z/`).
   - Dodać skrypt `scripts/models/build_oem_bundle.py` generujący pakiet (kompresja, manifest, podpisywanie).
   - Ustandaryzować `manifest.json` (pola: `model_family`, `version`, `hash_alg`, `artifacts[*]`, `inference_contract`).
2. **Kontrola integralności**
   - Rozszerzyć `bot_core/ai/models.py` o funkcję `verify_bundle_signature(manifest_path, signature_path, public_key)`.
   - Dodać obsługę błędów w `AIManager` (`MissingOEMBundle`, `InvalidOEMSignature`).
3. **Publikacja artefaktów**
   - Dodać workflow `deploy/ci/github_actions_oem_models.yml` generujący bundle na tagach `models/v*`.
   - Artefakty workflow zapisywać do `ai_models/bundles/latest/` i commitować w gałęzi release.

### Epik B – Wymuszenie obecności modeli w CI
1. **Konfiguracja**
   - Ustawić w `pytest.ini` lub `pyproject.toml` domyślną flagę `BOT_CORE_REQUIRE_OEM_MODELS=1` dla testów.
   - W plikach CI (`.github/workflows/*.yml`, `deploy/ci/*.yml`) dodać eksport zmiennej środowiskowej.
2. **Testy strażnicze**
   - Rozszerzyć `tests/ai/test_packaged_models_presence.py` o walidację manifestu i podpisu (użyć generowanej pary kluczy testowych).
   - Dodać test CLI `pytest tests/cli/test_verify_oem_bundle.py` uruchamiający `scripts/models/build_oem_bundle.py --validate`.
3. **Fail-fast**
   - W `bot_core/ai/manager.py` rzucać `RuntimeError` przy braku modeli, gdy wymuszenie jest aktywne.
   - Zabezpieczyć `ai_models/__init__.py`, aby logował krytyczne błędy i nie ukrywał wyjątków.

### Epik C – Runbook i proces operacyjny
1. **Runbook publikacyjny**
   - Utworzyć dokument `docs/runbooks/OEM_MODEL_RELEASE.md` opisujący kroki: trening, generowanie bundla, podpis, walidacja, dystrybucja.
   - Dodać tabelę kontroli jakości (accuracy, drift, sanity trading) i wymagane raporty dołączane do wydania.
2. **Aktualizacja dokumentacji**
   - Zaktualizować `docs/deployment/offline_packaging.md` o sekcję "Modele OEM" (umieszczenie bundla, weryfikacja podpisu, ścieżki).
   - Uzupełnić `docs/deployment/desktop_install.md` o krok weryfikacji modeli po instalacji.
3. **Checklisty compliance**
   - Rozszerzyć `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` o podpunkt "Potwierdź integralność modeli OEM" (skrypt `python scripts/models/verify_oem_bundle.py`).
   - Dodać wzór wpisu do `audit/decision_logs/demo.jsonl` z hashami bundla modeli.

## Harmonogram i zależności
| Zadanie | Czas (md) | Zależności | Output |
| --- | --- | --- | --- |
| Budowa skryptu bundla (`build_oem_bundle.py`) | 2 | Spec manifestu | Plik w `scripts/models/`, unit testy |
| Walidacja podpisów w `bot_core/ai/models.py` | 1.5 | Klucz publiczny | Funkcja + testy |
| Workflow CI `oem_models` | 1 | Skrypt bundla | Plik YAML + artefakty |
| Rozszerzenie testów obecności | 1 | Flaga CI | Testy `pytest` |
| Runbook `OEM_MODEL_RELEASE.md` | 1 | Spec procesu | Dokument w `docs/runbooks/` |
| Aktualizacja checklist | 0.5 | Runbook | Sekcje w dokumentach |

## Definition of Done
- `pytest -k oem` przechodzi lokalnie i w CI z aktywną flagą `BOT_CORE_REQUIRE_OEM_MODELS=1`.
- Workflow CI generuje bundla z podpisem testowym; artefakt trafia do `ai_models/bundles/latest/`.
- `AIManager` uruchomiony z domyślnej konfiguracji znajduje i weryfikuje bundla, a fallback jest wyłączony w logach.
- Dokumentacja (runbook + deployment) zawiera instrukcje weryfikacji i aktualizacji modeli.
- W `audit/decision_logs/demo.jsonl` istnieje zaktualizowany przykład wpisu z hashami modeli.

## Ryzyka i mitigacje
| Ryzyko | Mitigacja |
| --- | --- |
| Brak stabilnego procesu podpisywania | Przygotować tryb awaryjny (klucz operatorski) i rotację klucza w runbooku |
| Duży rozmiar bundla | Wprowadzić opcję dzielenia artefaktów i kompresję Zstandard |
| Różnice w środowiskach CI/offline | Testować bundla w trybie offline (air-gapped) przy pomocy `pytest --offline` |

