"""Artefakty raportowe są częścią zakresu mypy.

Plik zapewnia, że katalog `reports/` jest poprawnie rozpoznawany jako
pakiet na potrzeby statycznej analizy typów, mimo że zawiera głównie
wyniki testów i raporty tekstowe.

`__all__` jest jawnie puste, aby importy w stylu ``from reports import *``
nie wciągały żadnych artefaktów ani dynamicznych obiektów – katalog
pozostaje paczką wyłącznie na potrzeby narzędzi statycznych.
"""

from __future__ import annotations

__all__: list[str] = []
