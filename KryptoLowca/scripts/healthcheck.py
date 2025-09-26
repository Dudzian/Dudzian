"""Prosty healthcheck uruchamiany w kontenerze."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from KryptoLowca.logging_utils import setup_app_logging


def _check_required_paths() -> Dict[str, Any]:
    paths = {
        "config": Path(os.getenv("KRYPT_LOWCA_CONFIG", "")),
        "logs": Path(os.getenv("KRYPT_LOWCA_LOG_DIR", Path("/app/logs"))),
    }
    result = {}
    for name, path in paths.items():
        if not path:
            result[name] = {"status": "warning", "detail": "not-configured"}
            continue
        if path.exists():
            result[name] = {"status": "ok", "detail": str(path)}
        else:
            result[name] = {"status": "error", "detail": f"missing:{path}"}
    return result


def main() -> int:
    setup_app_logging()
    status = {
        "paths": _check_required_paths(),
        "env": {
            "mode": os.getenv("BOT_ENV", "demo"),
            "prometheus_port": os.getenv("KRYPT_LOWCA_PROMETHEUS_PORT"),
        },
    }
    if any(item.get("status") == "error" for item in status["paths"].values()):
        print(json.dumps(status))
        return 1
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
