from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import datetime

from cryptography.fernet import Fernet
import pytest

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.scripts import preset_editor_cli


def test_cli_editor_applies_preset(tmp_path: Path, capsys) -> None:
    marketplace_dir = tmp_path / "marketplace"
    marketplace_dir.mkdir()
    preset_payload = {
        "id": "cli_demo",
        "name": "CLI Demo",
        "description": "Preset do testów CLI",
        "risk_level": "safe",
        "recommended_min_balance": 500,
        "timeframe": "1h",
        "exchanges": ["binance"],
        "tags": ["cli"],
        "version": "0.0.1",
        "last_updated": "2024-06-10T12:00:00+00:00",
        "compatibility": {"app": ">=2.8.0"},
        "compliance": {"required_flags": ["compliance_confirmed"]},
        "config": {
            "strategy": {
                "preset": "CLI",
                "mode": "demo",
            },
            "trade": {"max_open_positions": 3},
        },
    }
    (marketplace_dir / "cli_demo.json").write_text(
        json.dumps(preset_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    key = Fernet.generate_key().decode()

    exit_code = preset_editor_cli.main(
        [
            "--config-path",
            str(config_path),
            "--marketplace-dir",
            str(marketplace_dir),
            "--preset-id",
            "cli_demo",
            "--encryption-key",
            key,
            "--actor",
            "cli@example.com",
            "--set",
            "trade.max_open_positions=5",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "cli_demo" in output

    manager = ConfigManager(config_path, encryption_key=key.encode())
    asyncio.run(manager.load_config())
    assert manager._current_config["trade"]["max_open_positions"] == 5

    history = manager.get_preset_history("cli_demo")
    assert len(history) >= 2


def test_cli_editor_prints_risk_summary(tmp_path: Path, capsys) -> None:
    marketplace_dir = tmp_path / "marketplace"
    marketplace_dir.mkdir()

    payloads = [
        {
            "id": "risk_a",
            "name": "Risk A",
            "description": "Preset z oceną ryzyka",
            "risk_level": "balanced",
            "config": {"strategy": {"preset": "A"}},
            "evaluation": {"rank": 3, "risk_label": "growth", "risk_score": 0.4},
        },
        {
            "id": "risk_b",
            "name": "Risk B",
            "description": "Preset bez oceny",
            "risk_level": "safe",
            "config": {"strategy": {"preset": "B"}},
        },
    ]

    for payload in payloads:
        (marketplace_dir / f"{payload['id']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    config_path = tmp_path / "config.yaml"
    key = Fernet.generate_key().decode()

    exit_code = preset_editor_cli.main(
        [
            "--config-path",
            str(config_path),
            "--marketplace-dir",
            str(marketplace_dir),
            "--preset-id",
            "risk_a",
            "--encryption-key",
            key,
            "--actor",
            "risk@example.com",
            "--print-risk-summary",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Podsumowanie ryzyka marketplace" in output
    assert "growth" in output
    assert "safe" in output
    assert "overall" in output
    assert "- overall: liczba presetów=2 (100.0%), z oceną ryzyka=1 (pokrycie=50.0%), z rankiem=1 (pokrycie=50.0%)," in output
    assert "z oceną ryzyka=1 (pokrycie=100.0%)" in output
    assert "z rankiem=1 (pokrycie=100.0%)" in output
    assert "σ wyniku=0.000" in output
    assert "dolny preset=risk_a" in output
    assert "mediana ranku=3.00" in output
    assert "mediana wyniku=0.40" in output
    assert "najgorszy rank=3" in output
    assert "P10 wyniku=0.40" in output
    assert "Q1 wyniku=0.40" in output
    assert "Q3 wyniku=0.40" in output
    assert "P90 wyniku=0.40" in output
    assert "IQR wyniku=0.00" in output
    assert "wariancja wyniku=0.0000" in output
    assert "MAD wyniku=0.000" in output
    assert "zakres wyniku=0.00" in output
    assert "CV wyniku=0.000" in output
    assert "skośność wyniku=0.000" in output
    assert "kurtoza wyniku=0.000" in output
    assert "JB wyniku=0.000" in output
    assert "P10 ranku=3.00" in output
    assert "Q1 ranku=3.00" in output
    assert "Q3 ranku=3.00" in output
    assert "P90 ranku=3.00" in output
    assert "IQR ranku=0.00" in output
    assert "σ ranku=0.000" in output
    assert "wariancja ranku=0.0000" in output
    assert "MAD ranku=0.000" in output
    assert "zakres ranku=0.00" in output
    assert "CV ranku=0.000" in output
    assert "skośność ranku=0.000" in output
    assert "kurtoza ranku=0.000" in output
    assert "JB ranku=0.000" in output
    assert "pary wynik-rank=1" in output
    assert "kowariancja wynik-rank=brak" in output
    assert "Pearson wynik-rank=brak" in output
    assert "Spearman wynik-rank=brak" in output
    assert "nachylenie regresji wynik→rank=brak" in output
    assert "wyraz wolny regresji=brak" in output
    assert "R^2 regresji=brak" in output
    assert "bias regresji wynik→rank=brak" in output
    assert "MAE regresji=brak" in output
    assert "MSE regresji=brak" in output
    assert "RMSE regresji=brak" in output
    assert "wariancja reszt regresji=brak" in output
    assert "σ reszt regresji=brak" in output
    assert "średni wynik ryzyka=brak danych" in output
    assert "z rankiem=0 (pokrycie=0.0%)" in output
    assert "mediana wyniku=brak" in output
    assert "mediana ranku=brak" in output
    assert "P10 wyniku=brak" in output
    assert "Q1 wyniku=brak" in output
    assert "Q3 wyniku=brak" in output
    assert "P90 wyniku=brak" in output
    assert "IQR wyniku=brak" in output
    assert "wariancja wyniku=brak" in output
    assert "MAD wyniku=brak" in output
    assert "zakres wyniku=brak" in output
    assert "CV wyniku=brak" in output
    assert "skośność wyniku=brak" in output
    assert "kurtoza wyniku=brak" in output
    assert "JB wyniku=brak" in output
    assert "P10 ranku=brak" in output
    assert "Q1 ranku=brak" in output
    assert "Q3 ranku=brak" in output
    assert "P90 ranku=brak" in output
    assert "IQR ranku=brak" in output
    assert "σ ranku=brak" in output
    assert "wariancja ranku=brak" in output
    assert "MAD ranku=brak" in output
    assert "zakres ranku=brak" in output
    assert "CV ranku=brak" in output
    assert "skośność ranku=brak" in output
    assert "kurtoza ranku=brak" in output
    assert "JB ranku=brak" in output
    assert "pary wynik-rank=0" in output
    assert "Pearson wynik-rank=brak" in output
    assert "Spearman wynik-rank=brak" in output
    assert "nachylenie regresji wynik→rank=brak" in output
    assert "wyraz wolny regresji=brak" in output
    assert "bias regresji wynik→rank=brak" in output
    assert "MAE regresji=brak" in output
    assert "MSE regresji=brak" in output
    assert "RMSE regresji=brak" in output
    assert "wariancja reszt regresji=brak" in output
    assert "σ reszt regresji=brak" in output
    assert "R^2 regresji=brak" in output


def test_cli_editor_exports_risk_summary(tmp_path: Path, capsys) -> None:
    marketplace_dir = tmp_path / "marketplace"
    marketplace_dir.mkdir()

    payload = {
        "id": "risk_export",
        "name": "Risk Export",
        "description": "Preset do eksportu podsumowania",
        "risk_level": "balanced",
        "config": {"strategy": {"preset": "EXPORT"}},
        "evaluation": {"rank": 1, "risk_label": "balanced", "risk_score": 0.75},
    }
    (marketplace_dir / "risk_export.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    destination = tmp_path / "summary.json"
    key = Fernet.generate_key().decode()

    exit_code = preset_editor_cli.main(
        [
            "--config-path",
            str(config_path),
            "--marketplace-dir",
            str(marketplace_dir),
            "--preset-id",
            "risk_export",
            "--encryption-key",
            key,
            "--actor",
            "export@example.com",
            "--export-risk-summary",
            str(destination),
        ]
    )

    assert exit_code == 0
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert "summary" in exported
    assert "overall" in exported["summary"]
    assert exported["summary"]["balanced"]["count"] == 1
    assert exported["summary"]["balanced"]["presets_with_score"] == 1
    assert "score_rank_regression_slope" in exported["summary"]["balanced"]
    assert exported["summary"]["balanced"]["score_rank_regression_slope"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_intercept"] is None
    assert exported["summary"]["balanced"]["score_rank_r_squared"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_bias"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_mae"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_mse"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_rmse"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_residual_variance"] is None
    assert exported["summary"]["balanced"]["score_rank_regression_residual_std_error"] is None
    assert exported["summary"]["balanced"]["presets_with_rank"] == 1
    assert exported["summary"]["balanced"]["min_risk_score"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["max_risk_score"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["risk_score_cv"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_cv"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_p10"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["risk_score_p90"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["rank_p10"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["rank_p90"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["avg_rank"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["risk_score_stddev"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_p25"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["risk_score_p75"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["risk_score_iqr"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_variance"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_mad"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_range"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_skewness"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_kurtosis"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["count_share"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["bottom_preset"] == "risk_export"
    assert exported["summary"]["balanced"]["top_preset"] == "risk_export"
    assert exported["summary"]["balanced"]["risk_score_median"] == pytest.approx(0.75)
    assert exported["summary"]["balanced"]["rank_median"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["score_coverage"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["rank_coverage"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["rank_p25"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["rank_p75"] == pytest.approx(1.0)
    assert exported["summary"]["balanced"]["rank_iqr"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_stddev"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_variance"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_mad"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_range"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_skewness"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_kurtosis"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["risk_score_jarque_bera"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["rank_jarque_bera"] == pytest.approx(0.0)
    assert exported["summary"]["balanced"]["score_rank_count"] == 1
    assert exported["summary"]["balanced"]["score_rank_covariance"] is None
    assert exported["summary"]["balanced"]["score_rank_pearson"] is None
    assert exported["summary"]["balanced"]["score_rank_spearman"] is None
    assert "generated_at" in exported
    # data ISO8601 z sufiksem strefy czasowej
    datetime.fromisoformat(exported["generated_at"])
    assert exported["summary"]["overall"]["count"] == 1
    assert exported["summary"]["overall"]["presets_with_score"] == 1
    assert exported["summary"]["overall"]["presets_with_rank"] == 1
    assert exported["summary"]["overall"]["count_share"] == pytest.approx(1.0)
    assert exported["summary"]["overall"]["score_coverage"] == pytest.approx(1.0)
    assert exported["summary"]["overall"]["rank_coverage"] == pytest.approx(1.0)
    assert exported["summary"]["overall"]["top_preset"] == "risk_export"
    assert exported["summary"]["overall"]["bottom_preset"] == "risk_export"
    assert exported["summary"]["overall"]["score_rank_count"] == 1
