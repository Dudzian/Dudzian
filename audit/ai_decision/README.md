# Artefakty audytu AI Decision Engine

Katalog przechowuje artefakty wspierające audyt pipeline'u AI:

- `data_quality/` – raporty kontroli kompletności i jakości danych (`*.json`).
- `drift/` – raporty dryfu cech wraz z podpisami compliance.
- `walk_forward/` – wyniki walidacji walk-forward wykonywane przez scheduler.
- `models/` – podpisane pakiety artefaktów modeli wygenerowane przez `bot_core.ai.generate_model_artifact_bundle`.

Każdy pakiet modelu zawiera strukturę `models/<model>/<timestamp>/` z następującymi plikami:

- `<nazwa>.json` – artefakt `ModelArtifact` zgodny z `docs/schemas/model_artifact.schema.json`.
- `<nazwa>.metadata.json` – streszczenie metadanych treningu (liczba wierszy, `feature_scalers`, `metrics`).
- `checksums.sha256` – sumy kontrolne SHA-256 dla wszystkich plików pakietu.
- `<nazwa>.sig` – podpis HMAC (`algorithm=HMAC-SHA256`) potwierdzający integralność artefaktu.

Pakiety są wymagane przez checklistę `docs/architecture/iteration_gate_checklists.md` przed aktywacją modelu w Decision Engine. Wpisy decision journalu powinny wskazywać katalog pakietu oraz identyfikator podpisu compliance.
