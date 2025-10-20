# database_manager.py
# -*- coding: utf-8 -*-
"""
Shim kompatybilności: re-eksportuje publiczny interfejs z KryptoLowca.database_manager.
Zostaw ten plik w katalogu głównym projektu, aby stare importy nadal działały.
"""
from __future__ import annotations
from KryptoLowca.database_manager import *  # noqa: F401,F403
