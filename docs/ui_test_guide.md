# UI test guide

This document describes how to run and debug the UI test suites (QML and native/ctest) locally.

CI reference (current workflows):

- Python setup uses `actions/setup-python@v6`.
- Linux `UI Native Tests` (ctest) and Linux `UI Packaging` install a full desktop Qt SDK via pinned `aqtinstall` (non-Node flow), then export `QT_ROOT_DIR` and `Qt6_DIR` from that SDK path.
- `UI / QML tests` in `main.yml` rely on Python dependencies (`pip install -e ".[test]"`) and do not call `install-qt-action` directly.

## Common prerequisites

- Python 3.11+
- System libraries used by Qt: `libegl1`, `libgl1`, `libpulse0`, `libxkbcommon-x11-0`, `libxcb-cursor0`, `libxcb-xinerama0`
- Pinned `PySide6` runtime installed locally for Python-side UI runtime
- Full Qt desktop SDK installed locally (for CMake/native builds), e.g. via `aqtinstall` into `$HOME/.local/qt/<version>/gcc_64`
- For headless environments, a virtual display (e.g., `xvfb` with a 1920x1080x24 screen)

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libpulse0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-xinerama0 xvfb
```

To start a virtual display that matches CI:

```bash
export DISPLAY=:99
Xvfb "$DISPLAY" -screen 0 1920x1080x24 > /tmp/xvfb.log 2>&1 &
```

Use `QT_QPA_PLATFORM=offscreen` and `QT_QUICK_BACKEND=software` to keep rendering deterministic:

```bash
export QT_QPA_PLATFORM=offscreen
export QT_QUICK_BACKEND=software
```

## QML tests (pytest)

Install Python dependencies (re-using the wheelhouse if available):

```bash
python -m pip install --upgrade pip
python scripts/ci/pip_install.py --wheelhouse wheelhouse -- \
  PySide6==${PYSIDE6_VERSION:-6.7.0} PySide6_Addons==${PYSIDE6_VERSION:-6.7.0} \
  PySide6_Essentials==${PYSIDE6_VERSION:-6.7.0} shiboken6==${PYSIDE6_VERSION:-6.7.0}
python scripts/ci/pip_install.py --wheelhouse wheelhouse -- .[dev]
```

Run the suite with diagnostics (same flags as CI):

```bash
mkdir -p test-results/qml
pytest -m qml --maxfail=1 --disable-warnings \
  --junitxml=test-results/qml/pytest.xml --durations=15 \
  --log-file=test-results/qml/pytest.log --log-file-level=INFO | tee test-results/qml/pytest-console.log
```

Artifacts of interest:

- `test-results/qml/pytest.xml` – JUnit report
- `test-results/qml/pytest.log` – structured log
- `test-results/qml/pytest-console.log` – console output

## Native UI tests (ctest)

Build and execute the C++ UI tests with the same configuration as CI:

```bash
python -m pip install "aqtinstall==3.2.1"
python -m aqt list-qt linux desktop --arch "${PYSIDE6_VERSION:-6.10.2}"
export QT_LINUX_ARCH="${QT_LINUX_ARCH:-linux_gcc_64}"
python -m aqt list-qt linux desktop --modules "${PYSIDE6_VERSION:-6.10.2}" "$QT_LINUX_ARCH"
python -m aqt install-qt linux desktop "${PYSIDE6_VERSION:-6.10.2}" "$QT_LINUX_ARCH" \
  --outputdir "$HOME/.local/qt" -m qtcharts qtdeclarative qtquickcontrols2 qttools

export QT_ROOT_DIR="$HOME/.local/qt/${PYSIDE6_VERSION:-6.10.2}/$QT_LINUX_ARCH"
export Qt6_DIR="$QT_ROOT_DIR/lib/cmake/Qt6"

cmake -S ui -B ui/build-tests -G Ninja -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH="$QT_ROOT_DIR"
cmake --build ui/build-tests
ctest --test-dir ui/build-tests --output-on-failure --output-log ui/build-tests/Testing/Temporary/ctest.log | tee ui/build-tests/Testing/Temporary/ctest-console.log
```

Useful artifacts:

- `ui/build-tests/Testing/Temporary/ctest.log` – ctest log
- `ui/build-tests/Testing/Temporary/ctest-console.log` – console output
- `ui/build-tests/Testing/Temporary/LastTest.log` – most recent ctest diagnostics
