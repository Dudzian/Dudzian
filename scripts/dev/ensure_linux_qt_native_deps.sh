#!/usr/bin/env bash
# Install native Linux libraries required by PySide6 QtQuick/GL smoke tests.
# This complements Python-only helpers such as ensure_pyside6.py and
# ensure_ui_runtime_deps.py; it intentionally does not install Python packages.
set -euo pipefail

packages=(
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
)

if [[ "${1:-}" == "--print" ]]; then
  printf '%s\n' "${packages[@]}"
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "ensure_linux_qt_native_deps.sh requires apt-get on Linux." >&2
  exit 1
fi

sudo_cmd=()
if [[ "${EUID}" -ne 0 ]]; then
  sudo_cmd=(sudo)
fi

"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y "${packages[@]}"
