# CI Command Map

Summary of workflows and commands for key jobs mentioned in failing checks.

## `.github/workflows/ci.yml`

- **Prepare PySide6 wheels (Linux)**
  - Python 3.11, caches wheelhouse, downloads `PySide6==${PYSIDE6_VERSION}` via `pip download`, uploads artifact `pyside6-wheels-linux-${PYSIDE6_VERSION}`.
- **Lint and Test**
  - `python scripts/lint_paths.py`
  - Install: `pip install .[dev] pytest pytest-cov pre-commit`
  - Pre-commit hooks (ruff + mypy): `pre-commit run --all-files --show-diff-on-failure`
  - Pytest selection with coverage threshold 75%: listed strategy/backtest tests.
- **Release Quality Gates**
  - Installs `libegl1` on Ubuntu.
  - Installs `. [dev]` and `mypy`; runs `mypy`, `pytest -m e2e_demo_paper`, `pytest tests/test_paper_execution.py`, `pytest tests/integration/test_execution_router_failover.py`.
- **Bot Core Fast Tests**
  - `python scripts/lint_paths.py`
  - Install `. [dev]`, generate trading stubs (`python scripts/generate_trading_stubs.py --skip-cpp`), then `pytest --fast --maxfail=1 --durations=10 --junitxml=test-results/pytest.xml` with `PYTEST_FAST=1`.
- **UI Native Tests**
  - Install build deps (protobuf/grpc/archive), install Qt 6.5.3 desktop modules, configure/build CMake (`cmake -S ui -B ui/build-tests -G Ninja -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=$Qt6_DIR`), run `ctest --test-dir ui/build-tests --output-on-failure`.
- **UI Packaging** (matrix: linux/windows/macos)
  - Python 3.11; macOS installs protobuf via brew.
  - Install Qt 6.5.3 (desktop modules; Windows limited modules).
  - Build bundle: `python scripts/packaging/qt_bundle.py --platform <platform> --build-dir ui/build-<platform> --install-dir ui/install-<platform> --artifact-dir artifacts/ui/<platform>`.
  - Uploads artifact `ui-<platform>`.
- **Prepare PySide6 wheels**
  - As above; prerequisite for QML/UI tests.

