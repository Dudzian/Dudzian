"""Łączy raporty Stage5 i Stage6 w podpisany przegląd hypercare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.runtime.full_hypercare import (
    FullHypercareSummaryBuilder,
    FullHypercareSummaryConfig,
)


def _load_key(value: str | None, path: str | None) -> bytes | None:
    if value and path:
        raise ValueError("Podaj klucz HMAC w formie tekstu lub pliku, nie obu naraz.")
    if value:
        return value.encode("utf-8")
    if path:
        return Path(path).expanduser().read_bytes().strip()
    return None


def _load_metadata(path: str | None) -> Mapping[str, object] | None:
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Plik metadanych powinien zawierać obiekt JSON")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agreguje raporty Stage5 i Stage6 do podpisanego przeglądu hypercare.",
    )

    parser.add_argument("--stage5-summary", required=True, help="Raport Stage5 hypercare JSON")
    parser.add_argument("--stage5-signature", help="Podpis HMAC raportu Stage5")
    parser.add_argument("--stage5-signing-key", help="Klucz HMAC do weryfikacji Stage5")
    parser.add_argument("--stage5-signing-key-file", help="Plik z kluczem HMAC Stage5")
    parser.add_argument(
        "--require-stage5-signature",
        action="store_true",
        help="Wymagaj obecności podpisu HMAC w raporcie Stage5",
    )

    parser.add_argument("--stage6-summary", required=True, help="Raport Stage6 hypercare JSON")
    parser.add_argument("--stage6-signature", help="Podpis HMAC raportu Stage6")
    parser.add_argument("--stage6-signing-key", help="Klucz HMAC do weryfikacji Stage6")
    parser.add_argument("--stage6-signing-key-file", help="Plik z kluczem HMAC Stage6")
    parser.add_argument(
        "--require-stage6-signature",
        action="store_true",
        help="Wymagaj obecności podpisu HMAC w raporcie Stage6",
    )

    parser.add_argument(
        "--output",
        default="var/audit/hypercare/full_hypercare_summary.json",
        help="Ścieżka docelowa raportu zbiorczego",
    )
    parser.add_argument("--signature", help="Opcjonalny podpis raportu zbiorczego")
    parser.add_argument("--signing-key", help="Klucz HMAC do podpisu raportu zbiorczego")
    parser.add_argument("--signing-key-file", help="Plik z kluczem HMAC do podpisu raportu")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza podpisującego raport")
    parser.add_argument(
        "--metadata",
        help="Plik JSON z dodatkowymi metadanymi dołączanymi do raportu zbiorczego",
    )

    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        stage5_key = _load_key(args.stage5_signing_key, args.stage5_signing_key_file)
        stage6_key = _load_key(args.stage6_signing_key, args.stage6_signing_key_file)
        signing_key = _load_key(args.signing_key, args.signing_key_file)
        metadata = _load_metadata(args.metadata)

        config = FullHypercareSummaryConfig(
            stage5_summary_path=Path(args.stage5_summary),
            stage6_summary_path=Path(args.stage6_summary),
            stage5_signature_path=Path(args.stage5_signature) if args.stage5_signature else None,
            stage6_signature_path=Path(args.stage6_signature) if args.stage6_signature else None,
            stage5_signing_key=stage5_key,
            stage6_signing_key=stage6_key,
            stage5_require_signature=bool(args.require_stage5_signature),
            stage6_require_signature=bool(args.require_stage6_signature),
            output_path=Path(args.output),
            signature_path=Path(args.signature) if args.signature else None,
            signing_key=signing_key,
            signing_key_id=args.signing_key_id,
            metadata=metadata,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    result = FullHypercareSummaryBuilder(config).run()
    output = {
        "summary_path": result.output_path.as_posix(),
        "signature_path": result.signature_path.as_posix() if result.signature_path else None,
        "overall_status": result.payload.get("overall_status"),
        "issues": result.payload.get("issues", []),
        "warnings": result.payload.get("warnings", []),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main() -> None:  # pragma: no cover - uruchomienie jako skrypt
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()

