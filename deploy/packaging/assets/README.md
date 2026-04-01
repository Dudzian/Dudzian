# Packaging assets layout

- `demo/` – sample/demo payloads used by demo profiles and `build_core_bundle.py --dry-run` defaults.
- `prod/` – production profile asset roots (kept separate from demo payloads).
  Production profiles intentionally keep `wheels_extra = []` until real offline wheels
  are explicitly provisioned in `assets/prod/wheels/`.

Test fixtures remain under `tests/assets/` and are intentionally outside `deploy/packaging/assets/`.
