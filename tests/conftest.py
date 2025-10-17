"""Globalna konfiguracja testów."""
from __future__ import annotations

# Import modułu zapewniającego, że katalog repozytorium znajduje się na sys.path.
# Dzięki temu wszystkie testy mogą importować kod projektu niezależnie od miejsca uruchomienia.
import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import
