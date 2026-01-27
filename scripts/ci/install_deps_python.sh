#!/usr/bin/env bash
set -euo pipefail

wheelhouse_args=()
wheelhouse_enabled="${WHEELHOUSE_ENABLED:-false}"
wheelhouse_dir="${WHEELHOUSE_DIR:-}"
wheelhouse_artifact="${WHEELHOUSE_ARTIFACT:-}"
runner_os="${RUNNER_OS:-}"
wheelhouse_enforce="false"

if [[ "${wheelhouse_enabled}" == "true" ]] && [[ -n "${wheelhouse_dir}" ]]; then
  wheelhouse_args+=(--wheelhouse "${wheelhouse_dir}")
  if [[ "${runner_os}" != "Windows" ]]; then
    wheelhouse_args+=(--require-wheelhouse)
    wheelhouse_enforce="true"
  fi
fi

echo "[install-deps] wheelhouse_enabled=${wheelhouse_enabled}"
echo "[install-deps] wheelhouse_dir=${wheelhouse_dir}"
echo "[install-deps] wheelhouse_artifact=${wheelhouse_artifact}"
echo "[install-deps] wheelhouse_enforce=${wheelhouse_enforce}"
python - <<'PY'
import sys
print(f"[install-deps] python={sys.version}")
PY

python -m pip install --upgrade pip
python scripts/ci/pip_install.py "${wheelhouse_args[@]+"${wheelhouse_args[@]}"}" -- -e ".[dev]"
