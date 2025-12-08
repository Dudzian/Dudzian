#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE_DIR=${WHEELHOUSE_DIR:-}
if [[ ${1:-} == "--wheelhouse" ]]; then
  WHEELHOUSE_DIR=${2:-}
  if [[ -z "${WHEELHOUSE_DIR}" ]]; then
    echo "--wheelhouse requires a path" >&2
    exit 1
  fi
  shift 2
fi

JOB=${1:-}
if [[ -z "${JOB}" ]]; then
  echo "Usage: $0 [--wheelhouse <path>] <job-name>" >&2
  exit 1
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACT_ROOT="artifacts/local-ci/${JOB}/${TIMESTAMP}"
LOG_FILE="${ARTIFACT_ROOT}/run.log"
mkdir -p "${ARTIFACT_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv-${JOB}}

run() {
  echo "[run] $*" | tee -a "${LOG_FILE}"
  "$@" 2>&1 | tee -a "${LOG_FILE}"
}

activate_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    run "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  run python -m pip install --upgrade pip
}

pip_install() {
  activate_venv
  if [[ -n "${WHEELHOUSE_DIR:-}" && -d "${WHEELHOUSE_DIR}" ]]; then
    run python scripts/ci/pip_install.py --wheelhouse "${WHEELHOUSE_DIR}" -- "$@"
  else
    run python scripts/ci/pip_install.py -- "$@"
  fi
}

install_dev_deps() {
  activate_venv
  pip_install .[dev]
}

ui_packaging_macos() {
  install_dev_deps
  export QT_DESKTOP_MODULES=${QT_DESKTOP_MODULES:-"qtcharts qtdeclarative qtquickcontrols2"}
  QT_PREFIX=${Qt6_DIR:-${QT_ROOT_DIR:-}}  # optional override
  mkdir -p ui/build-macos ui/install-macos "${ARTIFACT_ROOT}"
  run python scripts/packaging/qt_bundle.py \
    --platform macos \
    --build-dir ui/build-macos \
    --install-dir ui/install-macos \
    --artifact-dir "${ARTIFACT_ROOT}" \
    ${QT_PREFIX:+--qt-prefix ${QT_PREFIX}}
  run ls -la "${ARTIFACT_ROOT}"
}

prepare_wheelhouse() {
  activate_venv
  TARGET_WHEELHOUSE=${WHEELHOUSE_DIR:-wheelhouse}
  run python scripts/ci/build_wheelhouse.py --wheelhouse "${TARGET_WHEELHOUSE}" --pyside6-version "${PYSIDE6_VERSION:-6.7.0}" --only-binary :all:
  run ls -la "${TARGET_WHEELHOUSE}"
}

case "${JOB}" in
  ui-packaging-macos)
    ui_packaging_macos
    ;;
  prepare-wheelhouse)
    prepare_wheelhouse
    ;;
  *)
    echo "Unknown job: ${JOB}" | tee -a "${LOG_FILE}"
    exit 1
    ;;
 esac

echo "Logs saved to ${LOG_FILE}" | tee -a "${LOG_FILE}"
