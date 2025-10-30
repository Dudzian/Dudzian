#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROFILE=${PROFILE:-"$ROOT_DIR/deploy/packaging/profiles/windows.toml"}

if [[ ! -f "$PROFILE" ]]; then
  echo "Brak profilu: $PROFILE" >&2
  exit 1
fi

python "$ROOT_DIR/scripts/build_installer_from_profile.py" \
  --profile "$PROFILE" \
  --version "${VERSION:-0.0.0-dev}" \
  --platform windows \
  "$@"
