from __future__ import annotations

import json
from pathlib import Path
import string

from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace import (
    generate_exchange_presets,
    reconcile_exchange_presets,
    validate_exchange_presets,
)
from bot_core.marketplace.presets import parse_preset_document


def test_generate_exchange_presets_creates_signed_files(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    exchanges_dir.mkdir()
    output_dir = tmp_path / "presets"

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading
  exchange_manager:
    mode: paper
live:
  description: Production
  exchange_manager:
    mode: margin
""",
        encoding="utf-8",
    )

    private_key = ed25519.Ed25519PrivateKey.generate()

    documents = generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="generator",
        issuer="tests",
        version="1.2.3",
    )

    assert len(documents) == 1

    output_files = list(output_dir.glob("*.json"))
    assert output_files
    payload = json.loads(output_files[0].read_text(encoding="utf-8"))
    parsed = parse_preset_document(json.dumps(payload).encode("utf-8"), source=output_files[0])

    assert parsed.verification.verified is True
    assert parsed.preset_id.startswith("exchange_")
    assert parsed.payload["metadata"]["version"] == "1.2.3"
    assert parsed.payload["metadata"]["required_exchanges"] == ["BINANCE"]


def test_validate_exchange_presets_detects_missing_and_outdated(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    output_dir = tmp_path / "presets"
    exchanges_dir.mkdir()
    output_dir.mkdir()

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading
  exchange_manager:
    mode: paper

""",
        encoding="utf-8",
    )

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version="1.0.0",
    )
    assert len(results) == 1
    missing = results[0]
    assert missing.exists is False
    assert "missing-file" in missing.issues

    private_key = ed25519.Ed25519PrivateKey.generate()
    generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="validator",
        issuer="tests",
        version="1.0.0",
    )

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version="1.0.0",
    )
    assert len(results) == 1
    current = results[0]
    assert current.exists is True
    assert current.verified is True
    assert current.up_to_date is True
    assert not current.issues

    # modyfikacja definicji wymusi różnicę payloadu
    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading zmieniony
  exchange_manager:
    mode: paper

""",
        encoding="utf-8",
    )

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version="1.0.0",
    )
    assert len(results) == 1
    stale = results[0]
    assert stale.exists is True
    assert stale.verified is True
    assert stale.up_to_date is False
    assert "payload-mismatch" in stale.issues

    orphan_path = output_dir / "exchange_orphan.json"
    orphan_path.write_text(
        (output_dir / "exchange_binance.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version="1.0.0",
    )
    assert len(results) == 2
    orphan = next(result for result in results if result.preset_path == orphan_path)
    assert orphan.spec_path is None
    assert orphan.exists is True
    assert orphan.verified is True
    assert orphan.up_to_date is False
    assert "orphan-file" in orphan.issues


def test_generate_exchange_presets_handles_selection(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    exchanges_dir.mkdir()
    output_dir = tmp_path / "presets"

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    (exchanges_dir / "kraken.yaml").write_text(
        """
paper:
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    private_key = ed25519.Ed25519PrivateKey.generate()

    documents = generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="generator",
        issuer="tests",
        version="2.0.0",
        selected_exchanges=["binance"],
    )

    assert len(documents) == 1
    assert documents[0].preset_id == "exchange_binance"
    files = sorted(output_dir.glob("*.json"))
    assert [path.name for path in files] == ["exchange_binance.json"]


def test_reconcile_exchange_presets_regenerates_and_prunes(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    exchanges_dir.mkdir()
    output_dir = tmp_path / "presets"
    output_dir.mkdir()

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    (exchanges_dir / "kraken.yaml").write_text(
        """
paper:
  description: Kraken paper
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    private_key = ed25519.Ed25519PrivateKey.generate()

    generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="initial",
        issuer="tests",
        version="3.1.4",
    )

    # Usuń jeden plik, zmodyfikuj definicję i dodaj osierocony preset
    (output_dir / "exchange_kraken.json").unlink()
    orphan_path = output_dir / "exchange_orphan.json"
    orphan_path.write_text(
        (output_dir / "exchange_binance.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading updated
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    results = reconcile_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="reconcile",
        issuer="tests",
        version="3.1.4",
        remove_orphans=True,
    )

    assert all(result.issues == tuple() for result in results)
    assert all(result.exists for result in results)
    assert all(result.verified for result in results)
    assert all(result.up_to_date for result in results)

    assert not orphan_path.exists()


def test_validate_exchange_presets_selection_filters_results(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    exchanges_dir.mkdir()
    output_dir = tmp_path / "presets"
    output_dir.mkdir()

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    (exchanges_dir / "kraken.yaml").write_text(
        """
paper:
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    private_key = ed25519.Ed25519PrivateKey.generate()

    generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="selection",
        issuer="tests",
    )

    # Osierocony plik dla innej giełdy nie powinien być raportowany.
    (output_dir / "exchange_kraken_backup.json").write_text(
        (output_dir / "exchange_kraken.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        selected_exchanges=["binance"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.exchange_id == "BINANCE"
    assert not result.issues


def test_reconcile_exchange_presets_respects_selection(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    exchanges_dir.mkdir()
    output_dir = tmp_path / "presets"
    output_dir.mkdir()

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    (exchanges_dir / "kraken.yaml").write_text(
        """
paper:
  exchange_manager:
    mode: paper
""",
        encoding="utf-8",
    )

    private_key = ed25519.Ed25519PrivateKey.generate()

    generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="initial",
        issuer="tests",
    )

    # Usuń preset jednej giełdy i zostaw inny w spokoju.
    (output_dir / "exchange_binance.json").unlink()

    results = reconcile_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="repair",
        issuer="tests",
        selected_exchanges=["binance"],
        remove_orphans=True,
    )

    assert len(results) == 1
    binance = results[0]
    assert binance.exchange_id == "BINANCE"
    assert binance.exists is True
    assert binance.verified is True
    assert binance.up_to_date is True

    # Plik drugiej giełdy nie został naruszony.
    assert (output_dir / "exchange_kraken.json").exists()


def test_spec_hash_version_strategy_generates_deterministic_versions(tmp_path: Path) -> None:
    exchanges_dir = tmp_path / "exchanges"
    exchanges_dir.mkdir()
    output_dir = tmp_path / "presets"
    output_dir.mkdir()

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading
  exchange_manager:
    mode: paper
live:
  description: Live trading
  exchange_manager:
    mode: margin
""",
        encoding="utf-8",
    )

    private_key = ed25519.Ed25519PrivateKey.generate()

    documents = generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="hashing",
        issuer="tests",
        version="2.5.0",
        version_strategy="spec-hash",
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.version is not None
    assert document.version.startswith("2.5.0+")
    suffix = document.version.split("+", 1)[1]
    assert len(suffix) == 16
    assert all(ch in string.hexdigits for ch in suffix)

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version_strategy="spec-hash",
    )

    assert len(results) == 1
    result = results[0]
    assert result.issues == tuple()
    assert result.expected_version == document.version

    (exchanges_dir / "binance.yaml").write_text(
        """
paper:
  description: Paper trading updated
  exchange_manager:
    mode: paper
live:
  description: Live trading
  exchange_manager:
    mode: margin
""",
        encoding="utf-8",
    )

    stale_results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version_strategy="spec-hash",
    )

    assert len(stale_results) == 1
    stale = stale_results[0]
    assert "payload-mismatch" in stale.issues
    assert stale.expected_version != document.version

    repaired_results = reconcile_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id="hashing",
        issuer="tests",
        version="2.5.0",
        version_strategy="spec-hash",
    )

    assert len(repaired_results) == 1
    repaired = repaired_results[0]
    assert repaired.issues == tuple()
    assert repaired.expected_version != document.version
    assert repaired.expected_version == repaired.current_version
    assert repaired.current_version is not None
    new_suffix = repaired.current_version.split("+", 1)[1]
    assert len(new_suffix) == 16
    assert new_suffix != suffix
