"""Walidacja podsumowania smoke testu paper tradingu."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sprawdza statusy w pliku summary.json wygenerowanym przez smoke test "
            "paper tradingu i zwraca kod wyjścia odpowiedni do dalszej automatyzacji CI."
        )
    )
    parser.add_argument(
        "--summary",
        default="var/paper_smoke_ci/paper_smoke_summary.json",
        help="Ścieżka do pliku podsumowania smoke testu",
    )
    parser.add_argument(
        "--require-environment",
        default=None,
        help="Jeżeli podano, walidacja wymaga zgodności pola 'environment' w summary",
    )
    parser.add_argument(
        "--require-operator",
        default=None,
        help="Jeżeli podano, walidacja wymaga zgodności pola 'operator' w summary",
    )
    parser.add_argument(
        "--require-publish-success",
        action="store_true",
        help="Wymaga aby sekcja publish.status miała wartość 'ok'",
    )
    parser.add_argument(
        "--require-publish-required",
        action="store_true",
        help="Wymaga aby summary oznaczało auto-publikację jako wymaganą (publish.required)",
    )
    parser.add_argument(
        "--require-publish-exit-zero",
        action="store_true",
        help="Wymaga aby publish.exit_code (jeśli obecne) było równe 0",
    )
    parser.add_argument(
        "--require-json-sync-ok",
        action="store_true",
        help="Wymaga aby sekcja publish.json_sync posiadała status 'ok'",
    )
    parser.add_argument(
        "--require-archive-upload-ok",
        action="store_true",
        help="Wymaga aby sekcja publish.archive_upload posiadała status 'ok'",
    )
    return parser.parse_args(argv)


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - ochronny warunek
        raise ValueError("Oczekiwano obiektu JSON na najwyższym poziomie")
    return payload


def _validate_summary(
    payload: dict[str, Any],
    *,
    require_environment: str | None,
    require_operator: str | None,
    require_publish_success: bool,
    require_publish_required: bool,
    require_publish_exit_zero: bool,
    require_json_sync_ok: bool,
    require_archive_upload_ok: bool,
) -> list[str]:
    errors: list[str] = []

    status = payload.get("status")
    if status != "ok":
        errors.append(f"status: oczekiwano 'ok', otrzymano {status!r}")

    severity = payload.get("severity")
    if severity in {"critical", "error"}:
        errors.append(f"severity: {severity!r} wskazuje na błąd")

    if require_environment is not None:
        environment = payload.get("environment")
        if environment != require_environment:
            errors.append(
                "environment: oczekiwano "
                f"{require_environment!r}, otrzymano {environment!r}"
            )

    if require_operator is not None:
        operator = payload.get("operator")
        if operator != require_operator:
            errors.append(
                f"operator: oczekiwano {require_operator!r}, otrzymano {operator!r}"
            )

    publish = payload.get("publish")
    if require_publish_success:
        if not isinstance(publish, dict):
            errors.append("publish: brak sekcji publish pomimo wymogu powodzenia")
        else:
            publish_status = publish.get("status")
            if publish_status != "ok":
                errors.append(
                    "publish.status: oczekiwano 'ok', otrzymano "
                    f"{publish_status!r}"
                )

    if require_publish_required:
        if not isinstance(publish, dict):
            errors.append("publish: brak sekcji publish pomimo wymogu 'required'")
        else:
            if not bool(publish.get("required")):
                errors.append("publish.required: oczekiwano True")

    if require_publish_exit_zero:
        if not isinstance(publish, dict):
            errors.append("publish: brak sekcji publish pomimo wymogu exit_code==0")
        else:
            exit_code = publish.get("exit_code")
            if exit_code is None:
                errors.append("publish.exit_code: brak wartości przy wymogu exit_code==0")
            else:
                try:
                    coerced = int(exit_code)
                except (TypeError, ValueError):
                    errors.append(
                        f"publish.exit_code: oczekiwano liczby całkowitej, otrzymano {exit_code!r}"
                    )
                else:
                    if coerced != 0:
                        errors.append(
                            f"publish.exit_code: oczekiwano 0, otrzymano {coerced}"
                        )

    def _require_step_ok(step_key: str) -> None:
        if not isinstance(publish, dict):
            errors.append(
                f"publish: brak sekcji publish pomimo wymogu {step_key}.status=='ok'"
            )
            return
        step = publish.get(step_key)
        if not isinstance(step, dict):
            errors.append(f"publish.{step_key}: oczekiwano obiektu z polem 'status'")
            return
        status = step.get("status")
        if status != "ok":
            errors.append(
                f"publish.{step_key}.status: oczekiwano 'ok', otrzymano {status!r}"
            )

    if require_json_sync_ok:
        _require_step_ok("json_sync")

    if require_archive_upload_ok:
        _require_step_ok("archive_upload")

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary_path = Path(args.summary)

    result: dict[str, Any] = {
        "summary_path": str(summary_path.resolve()),
        "status": "error",
        "errors": [],
    }

    try:
        payload = _load_summary(summary_path)
    except FileNotFoundError:
        result["errors"].append("summary_not_found")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    except Exception as exc:  # pragma: no cover - ochronny fallback
        result["errors"].append(f"failed_to_load_summary: {exc}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 3

    errors = _validate_summary(
        payload,
        require_environment=args.require_environment,
        require_operator=args.require_operator,
        require_publish_success=args.require_publish_success,
        require_publish_required=args.require_publish_required,
        require_publish_exit_zero=args.require_publish_exit_zero,
        require_json_sync_ok=args.require_json_sync_ok,
        require_archive_upload_ok=args.require_archive_upload_ok,
    )

    result.update({
        "status": "ok" if not errors else "failed",
        "summary": payload,
        "errors": errors,
    })

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover - uruchomienie jako skrypt
    raise SystemExit(main())
