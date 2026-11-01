#!/bin/sh
set -eu

# Sam instalator i interpreter muszą pochodzić z niezależnie zweryfikowanej,
# chronionej kopii. Podpis pakietu nie uwierzytelnia kodu sprawdzającego podpis.
# Diagnostyka nie uruchamia sudo ani operacji przygotowania hosta.
installer_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec /usr/bin/python3 -I -B "$installer_directory/production_local_installer.py" "$@"
