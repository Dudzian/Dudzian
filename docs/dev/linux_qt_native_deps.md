# Linux QtQuick/GL native dependencies

`python scripts/dev/ensure_pyside6.py --install` and
`python scripts/dev/ensure_ui_runtime_deps.py --install` cover Python packages
only. PySide6 QtQuick still needs native Linux OpenGL/EGL/XCB libraries.
Without them, importing or loading QtQuick can fail before QML errors are
available; for example a minimal import can raise `ImportError: libGL.so.1:
cannot open shared object file`.

For Ubuntu/GitHub-hosted Linux runners, install the minimal native preflight set
used by CI:

```bash
scripts/dev/ensure_linux_qt_native_deps.sh
```

The script installs these apt packages:

```text
libegl1
libgl1
libopengl0
libxkbcommon0
libxkbcommon-x11-0
libxcb-cursor0
libxcb-icccm4
libxcb-image0
libxcb-keysyms1
libxcb-render-util0
libxcb-shape0
libxcb-xinerama0
libxcb-xkb1
xvfb
xauth
```

Recommended local smoke command after Python and native preflight:

```bash
QT_QPA_PLATFORM=offscreen QT_OPENGL=software DUDZIAN_QML_FLUSH_DELETES=0 \
  python -m pytest tests/smoke/test_smoke_app.py -q -rs
```
