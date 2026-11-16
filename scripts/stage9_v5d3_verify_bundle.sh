#!/usr/bin/env bash
# Uruchamia wyłącznie zaufany interpreter hosta i weryfikator, nigdy kod pakietu.
set -eu
script_directory=${BASH_SOURCE[0]%/*}
if [[ $script_directory == "${BASH_SOURCE[0]}" ]]; then script_directory=.; fi
exec /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
    /usr/bin/python3 -I -S -B "$script_directory/stage9_v5d3_verify_bundle.py" "$@"
