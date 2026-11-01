"""Identyfikatory dowodów Stage 8 bez zależności od producentów i transportów.

Sonda PostgreSQL oraz ogólny producent dowodów korzystają z tego samego
kontraktu danych. Import identyfikatorów nie włącza producentów Linux do
grafu lokalnych operacji emisji.
"""

WINDOWS_STAGE8_ITEMS = (
    "WINDOWS_POSTGRESQL_SUBSTRATE",
    "WINDOWS_LOCAL_PRINCIPAL_AUTHENTICATION",
)
