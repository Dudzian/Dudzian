"""Production-safe frozen PostgreSQL certificate authentication contract."""

from __future__ import annotations
from typing import Any

MAP_NAME = "stage8_cert"
RUNTIME_ROLE = "freshness_runtime"
VERIFIER_ROLE = "freshness_crypto_verifier"
RUNTIME_IDENTITY = "CryptoHunterBackend"
VERIFIER_IDENTITY = "CryptoHunterFreshnessVerifier"


def final_hba_lines(database: str) -> tuple[str, ...]:
    options = f"cert map={MAP_NAME}"
    return (
        f"hostssl {database} {VERIFIER_ROLE} 127.0.0.1/32 {options}",
        f"hostssl {database} {RUNTIME_ROLE} 127.0.0.1/32 {options}",
        f"host {database} all 127.0.0.1/32 reject",
        "host all all 127.0.0.1/32 reject",
        "host all all 0.0.0.0/0 reject",
        "host all all ::0/0 reject",
    )


def ident_lines() -> tuple[str, ...]:
    return (
        f"{MAP_NAME} {RUNTIME_IDENTITY} {RUNTIME_ROLE}",
        f"{MAP_NAME} {VERIFIER_IDENTITY} {VERIFIER_ROLE}",
    )


def qualify_hba_rows(rows: list[tuple[Any, ...]], database: str) -> None:
    expected = [
        (
            "hostssl",
            database,
            VERIFIER_ROLE,
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            [f"map={MAP_NAME}", "clientcert=verify-full"],
        ),
        (
            "hostssl",
            database,
            RUNTIME_ROLE,
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            [f"map={MAP_NAME}", "clientcert=verify-full"],
        ),
        ("host", database, "all", "127.0.0.1", "255.255.255.255", "reject", None),
        ("host", "all", "all", "127.0.0.1", "255.255.255.255", "reject", None),
        ("host", "all", "all", "0.0.0.0", "0.0.0.0", "reject", None),
        ("host", "all", "all", "::", "::", "reject", None),
    ]
    if len(rows) != len(expected):
        raise ValueError("unexpected effective HBA row count")
    previous = -1
    for row, wanted in zip(rows, expected, strict=True):
        line, kind, databases, users, address, netmask, method, options, error = row
        (
            expected_kind,
            database_name,
            user,
            expected_address,
            expected_netmask,
            auth,
            expected_options,
        ) = wanted
        if (
            error is not None
            or line <= previous
            or (kind, databases, users, address, netmask, method, options)
            != (
                expected_kind,
                [database_name],
                [user],
                expected_address,
                expected_netmask,
                auth,
                expected_options,
            )
        ):
            raise ValueError("exact effective HBA row differs")
        previous = line
