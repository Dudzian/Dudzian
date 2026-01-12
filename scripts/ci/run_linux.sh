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

prepare_pyside6_wheel() {
  activate_venv
  echo "HTTPS_PROXY=${HTTPS_PROXY:-}"
  echo "HTTP_PROXY=${HTTP_PROXY:-}"
  echo "PIP_INDEX_URL=${PIP_INDEX_URL:-}"
  echo "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-}"
  echo "NO_PROXY=${NO_PROXY:-}"
  run python -m pip config debug -v
  run python scripts/ci/build_wheelhouse.py --wheelhouse wheelhouse --pyside6-version "${PYSIDE6_VERSION:-6.7.0}" --only-binary :all:
  run ls -la wheelhouse
}

ui_packaging_linux() {
  install_dev_deps
  QT_PREFIX=${Qt6_DIR:-${QT_ROOT_DIR:-}}  # optional override
  export QT_DESKTOP_MODULES=${QT_DESKTOP_MODULES:-"qtcharts qtdeclarative qtquickcontrols2"}
  mkdir -p ui/build-linux ui/install-linux "${ARTIFACT_ROOT}"
  run python scripts/packaging/qt_bundle.py \
    --platform linux \
    --build-dir ui/build-linux \
    --install-dir ui/install-linux \
    --artifact-dir "${ARTIFACT_ROOT}" \
    ${QT_PREFIX:+--qt-prefix ${QT_PREFIX}}
  run ls -la "${ARTIFACT_ROOT}"
}

ui_native_tests() {
  install_dev_deps
  export CMAKE_GENERATOR=Ninja
  mkdir -p ui/build-tests
  if command -v ninja >/dev/null 2>&1; then
    :
  else
    run sudo apt-get update
    run sudo apt-get install -y ninja-build
  fi
  run cmake -S ui -B ui/build-tests -G Ninja -DBUILD_TESTING=ON ${Qt6_DIR:+-DCMAKE_PREFIX_PATH=${Qt6_DIR}}
  run cmake --build ui/build-tests
  run ctest --test-dir ui/build-tests --output-on-failure
}

bot_core_fast_tests() {
  install_dev_deps
  run python scripts/lint_paths.py
  run python scripts/generate_trading_stubs.py --skip-cpp
  mkdir -p test-results
  export PYTEST_FAST=1
  run pytest --fast --maxfail=1 --durations=10 --junitxml=test-results/pytest.xml
}

lint_and_test() {
  install_dev_deps
  run python scripts/lint_paths.py
  run pre-commit run --all-files --show-diff-on-failure
  run pytest \
    --cov=bot_core.strategies \
    --cov=bot_core.runtime.multi_strategy_scheduler \
    --cov=bot_core.runtime.journal \
    --cov-config=.coveragerc \
    --cov-report=xml \
    --cov-report=term \
    --cov-fail-under=75 \
    tests/test_pipeline_paper.py \
    tests/test_risk_profiles.py \
    tests/test_mean_reversion_strategy.py \
    tests/test_volatility_target_strategy.py \
    tests/test_cross_exchange_arbitrage_strategy.py \
    tests/test_multi_strategy_scheduler.py \
    tests/test_backtest_dataset_library.py \
    tests/test_telemetry_risk_profiles.py \
    tests/test_trading_decision_journal.py \
    tests/test_smoke_demo_strategies_cli.py
}

prepare_wheelhouse() {
  activate_venv
  TARGET_WHEELHOUSE=${WHEELHOUSE_DIR:-wheelhouse}
  run python scripts/ci/build_wheelhouse.py --wheelhouse "${TARGET_WHEELHOUSE}" --pyside6-version "${PYSIDE6_VERSION:-6.7.0}" --only-binary :all:
  run ls -la "${TARGET_WHEELHOUSE}"
}

release_quality_gates() {
  install_dev_deps
  run sudo apt-get update
  run sudo apt-get install -y \
    libegl1 \
    libgl1 \
    libpulse0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-xinerama0
  run python scripts/lint_paths.py
  pip_install mypy
  run mypy
  run pytest -m e2e_demo_paper --maxfail=1 --disable-warnings
  run pytest tests/test_paper_execution.py
  run pytest tests/integration/test_execution_router_failover.py
}

qml_collect_only() {
  install_dev_deps
  set +e
  local output
  output=$(python -m pytest --collect-only -q tests/ui/qml/test_risk_panels.py 2>&1 | tee -a "${LOG_FILE}")
  local status=${PIPESTATUS[0]}
  set -e
  if echo "${output}" | grep -Fq "found no collectors"; then
    echo "QML collect-only failed: found no collectors" | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
  if [[ ${status} -ne 0 && ${status} -ne 5 ]]; then
    echo "QML collect-only failed with status ${status}" | tee -a "${LOG_FILE}" >&2
    exit "${status}"
  fi
}

case "${JOB}" in
  prepare-pyside6-wheel)
    prepare_pyside6_wheel
    ;;
  prepare-wheelhouse)
    prepare_wheelhouse
    ;;
  ui-packaging-linux)
    ui_packaging_linux
    ;;
  ui-native-tests)
    ui_native_tests
    ;;
  bot-core-fast-tests)
    bot_core_fast_tests
    ;;
  lint-and-test)
    lint_and_test
    ;;
  release-quality-gates)
    release_quality_gates
    ;;
  qml-collect-only)
    qml_collect_only
    ;;
  *)
    echo "Unknown job: ${JOB}" | tee -a "${LOG_FILE}"
    exit 1
    ;;
 esac

echo "Logs saved to ${LOG_FILE}" | tee -a "${LOG_FILE}"
