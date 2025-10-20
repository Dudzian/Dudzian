"""Pakiet ``KryptoLowca.managers`` został wycofany."""

from __future__ import annotations

import textwrap

raise RuntimeError(
    textwrap.dedent(
        """
        Pakiet `KryptoLowca.managers` został usunięty w ramach migracji do architektury
        opartej na `bot_core`.  Zamiast niego użyj nowych modułów top-level, np.:

          • KryptoLowca.database_manager
          • KryptoLowca.exchange_manager
          • KryptoLowca.security_manager
          • KryptoLowca.report_manager
          • KryptoLowca.risk_manager

        Jeśli potrzebujesz dawnych struktur DTO, importuj je z `bot_core.exchanges.core`.
        Usuń referencje do `KryptoLowca.managers.*`, aby utrzymać spójność z aktualnym
        runtime.
        """
    ).strip()
)
