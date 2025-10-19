"""Prosty edytor CLI presetów marketplace.

Pozwala zastosować preset, wprowadzić modyfikacje i zapisać konfigurację
z wykorzystaniem szyfrowania sekcji API. Moduł stanowi stub możliwy do
rozszerzenia o GUI w przyszłości.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import json
import yaml
from KryptoLowca.config_manager import ConfigManager, ConfigError, ValidationError


def _load_key(args: argparse.Namespace) -> bytes:
    if args.encryption_key and args.encryption_key_file:
        raise SystemExit("Podaj klucz w formie tekstu lub pliku, nie obu jednocześnie.")
    if args.encryption_key:
        return args.encryption_key.encode()
    if args.encryption_key_file:
        path = Path(args.encryption_key_file)
        try:
            return path.read_text(encoding="utf-8").strip().encode()
        except OSError as exc:  # pragma: no cover - informacja o błędzie IO
            raise SystemExit(f"Nie można odczytać pliku z kluczem: {exc}")
    raise SystemExit("Wymagany jest klucz szyfrujący (podaj --encryption-key lub --encryption-key-file).")


def _parse_overrides(items: Iterable[str]) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Nieprawidłowy format nadpisania: '{item}'. Użyj section.key=value.")
        path, raw_value = item.split("=", 1)
        if "." not in path:
            raise SystemExit(f"Nieprawidłowy format ścieżki: '{path}'. Użyj section.key.")
        section, key = path.split(".", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        section = section.strip()
        key = key.strip()
        overrides.setdefault(section, {})[key] = value
    return overrides


def _summarise_overrides(overrides: Dict[str, Dict[str, object]]) -> str:
    flattened: Iterable[Tuple[str, str, object]] = (
        (section, key, value)
        for section, values in overrides.items()
        for key, value in values.items()
    )
    return ", ".join(f"{section}.{key}={value}" for section, key, value in flattened)


def _format_risk_summary(summary: Dict[str, Dict[str, object]]) -> str:
    if not summary:
        return "Brak presetów w marketplace do podsumowania ryzyka."

    lines = ["Podsumowanie ryzyka marketplace:"]
    ordered_labels = sorted(summary)
    if "overall" in summary:
        ordered_labels = ["overall"] + [label for label in ordered_labels if label != "overall"]
    for label in ordered_labels:
        data = summary[label]
        avg_risk = data.get("avg_risk_score")
        avg_display = f"{avg_risk:.2f}" if isinstance(avg_risk, (int, float)) else "brak danych"
        min_risk = data.get("min_risk_score")
        min_display = f"{min_risk:.2f}" if isinstance(min_risk, (int, float)) else "brak"
        max_risk = data.get("max_risk_score")
        max_display = f"{max_risk:.2f}" if isinstance(max_risk, (int, float)) else "brak"
        best_rank = data.get("best_rank")
        rank_display = best_rank if best_rank is not None else "brak"
        worst_rank = data.get("worst_rank")
        worst_display = worst_rank if worst_rank is not None else "brak"
        top_preset = data.get("top_preset") or "brak"
        bottom_preset = data.get("bottom_preset") or "brak"
        scored = data.get("presets_with_score", 0)
        ranked = data.get("presets_with_rank", 0)
        avg_rank = data.get("avg_rank")
        avg_rank_display = f"{avg_rank:.2f}" if isinstance(avg_rank, (int, float)) else "brak"
        median_risk = data.get("risk_score_median")
        median_risk_display = (
            f"{median_risk:.2f}" if isinstance(median_risk, (int, float)) else "brak"
        )
        median_rank = data.get("rank_median")
        median_rank_display = (
            f"{median_rank:.2f}" if isinstance(median_rank, (int, float)) else "brak"
        )
        std_dev = data.get("risk_score_stddev")
        std_display = f"{std_dev:.3f}" if isinstance(std_dev, (int, float)) else "brak"
        score_p10 = data.get("risk_score_p10")
        score_p10_display = (
            f"{score_p10:.2f}" if isinstance(score_p10, (int, float)) else "brak"
        )
        score_p25 = data.get("risk_score_p25")
        score_p25_display = (
            f"{score_p25:.2f}" if isinstance(score_p25, (int, float)) else "brak"
        )
        score_p75 = data.get("risk_score_p75")
        score_p75_display = (
            f"{score_p75:.2f}" if isinstance(score_p75, (int, float)) else "brak"
        )
        score_p90 = data.get("risk_score_p90")
        score_p90_display = (
            f"{score_p90:.2f}" if isinstance(score_p90, (int, float)) else "brak"
        )
        score_iqr = data.get("risk_score_iqr")
        score_iqr_display = (
            f"{score_iqr:.2f}" if isinstance(score_iqr, (int, float)) else "brak"
        )
        score_variance = data.get("risk_score_variance")
        score_variance_display = (
            f"{score_variance:.4f}" if isinstance(score_variance, (int, float)) else "brak"
        )
        score_mad = data.get("risk_score_mad")
        score_mad_display = (
            f"{score_mad:.3f}" if isinstance(score_mad, (int, float)) else "brak"
        )
        score_range = data.get("risk_score_range")
        score_range_display = (
            f"{score_range:.2f}" if isinstance(score_range, (int, float)) else "brak"
        )
        score_cv = data.get("risk_score_cv")
        score_cv_display = (
            f"{score_cv:.3f}" if isinstance(score_cv, (int, float)) else "brak"
        )
        score_skewness = data.get("risk_score_skewness")
        score_skew_display = (
            f"{score_skewness:.3f}" if isinstance(score_skewness, (int, float)) else "brak"
        )
        score_kurtosis = data.get("risk_score_kurtosis")
        score_kurtosis_display = (
            f"{score_kurtosis:.3f}" if isinstance(score_kurtosis, (int, float)) else "brak"
        )
        score_jb = data.get("risk_score_jarque_bera")
        score_jb_display = (
            f"{score_jb:.3f}" if isinstance(score_jb, (int, float)) else "brak"
        )
        share = data.get("count_share", 0.0)
        share_display = f"{share * 100:.1f}%" if isinstance(share, (int, float)) else "brak"
        score_cov = data.get("score_coverage", 0.0)
        score_cov_display = (
            f"{score_cov * 100:.1f}%" if isinstance(score_cov, (int, float)) else "brak"
        )
        rank_cov = data.get("rank_coverage", 0.0)
        rank_cov_display = (
            f"{rank_cov * 100:.1f}%" if isinstance(rank_cov, (int, float)) else "brak"
        )
        rank_p10 = data.get("rank_p10")
        rank_p10_display = (
            f"{rank_p10:.2f}" if isinstance(rank_p10, (int, float)) else "brak"
        )
        rank_p25 = data.get("rank_p25")
        rank_p25_display = (
            f"{rank_p25:.2f}" if isinstance(rank_p25, (int, float)) else "brak"
        )
        rank_p75 = data.get("rank_p75")
        rank_p75_display = (
            f"{rank_p75:.2f}" if isinstance(rank_p75, (int, float)) else "brak"
        )
        rank_p90 = data.get("rank_p90")
        rank_p90_display = (
            f"{rank_p90:.2f}" if isinstance(rank_p90, (int, float)) else "brak"
        )
        rank_iqr = data.get("rank_iqr")
        rank_iqr_display = (
            f"{rank_iqr:.2f}" if isinstance(rank_iqr, (int, float)) else "brak"
        )
        rank_std = data.get("rank_stddev")
        rank_std_display = (
            f"{rank_std:.3f}" if isinstance(rank_std, (int, float)) else "brak"
        )
        rank_variance = data.get("rank_variance")
        rank_variance_display = (
            f"{rank_variance:.4f}" if isinstance(rank_variance, (int, float)) else "brak"
        )
        rank_mad = data.get("rank_mad")
        rank_mad_display = (
            f"{rank_mad:.3f}" if isinstance(rank_mad, (int, float)) else "brak"
        )
        rank_range = data.get("rank_range")
        rank_range_display = (
            f"{rank_range:.2f}" if isinstance(rank_range, (int, float)) else "brak"
        )
        rank_cv = data.get("rank_cv")
        rank_cv_display = (
            f"{rank_cv:.3f}" if isinstance(rank_cv, (int, float)) else "brak"
        )
        rank_skewness = data.get("rank_skewness")
        rank_skew_display = (
            f"{rank_skewness:.3f}" if isinstance(rank_skewness, (int, float)) else "brak"
        )
        rank_kurtosis = data.get("rank_kurtosis")
        rank_kurtosis_display = (
            f"{rank_kurtosis:.3f}" if isinstance(rank_kurtosis, (int, float)) else "brak"
        )
        rank_jb = data.get("rank_jarque_bera")
        rank_jb_display = (
            f"{rank_jb:.3f}" if isinstance(rank_jb, (int, float)) else "brak"
        )
        pair_count = data.get("score_rank_count", 0)
        pair_cov = data.get("score_rank_covariance")
        pair_cov_display = (
            f"{pair_cov:.4f}" if isinstance(pair_cov, (int, float)) else "brak"
        )
        pair_pearson = data.get("score_rank_pearson")
        pair_pearson_display = (
            f"{pair_pearson:.3f}" if isinstance(pair_pearson, (int, float)) else "brak"
        )
        pair_spearman = data.get("score_rank_spearman")
        pair_spearman_display = (
            f"{pair_spearman:.3f}" if isinstance(pair_spearman, (int, float)) else "brak"
        )
        pair_slope = data.get("score_rank_regression_slope")
        pair_slope_display = (
            f"{pair_slope:.3f}" if isinstance(pair_slope, (int, float)) else "brak"
        )
        pair_intercept = data.get("score_rank_regression_intercept")
        pair_intercept_display = (
            f"{pair_intercept:.3f}" if isinstance(pair_intercept, (int, float)) else "brak"
        )
        pair_r_squared = data.get("score_rank_r_squared")
        pair_r_squared_display = (
            f"{pair_r_squared:.3f}" if isinstance(pair_r_squared, (int, float)) else "brak"
        )
        pair_bias = data.get("score_rank_regression_bias")
        pair_bias_display = (
            f"{pair_bias:.3f}" if isinstance(pair_bias, (int, float)) else "brak"
        )
        pair_mae = data.get("score_rank_regression_mae")
        pair_mae_display = (
            f"{pair_mae:.3f}" if isinstance(pair_mae, (int, float)) else "brak"
        )
        pair_mse = data.get("score_rank_regression_mse")
        pair_mse_display = (
            f"{pair_mse:.4f}" if isinstance(pair_mse, (int, float)) else "brak"
        )
        pair_rmse = data.get("score_rank_regression_rmse")
        pair_rmse_display = (
            f"{pair_rmse:.3f}" if isinstance(pair_rmse, (int, float)) else "brak"
        )
        pair_residual_variance = data.get("score_rank_regression_residual_variance")
        pair_residual_variance_display = (
            f"{pair_residual_variance:.4f}"
            if isinstance(pair_residual_variance, (int, float))
            else "brak"
        )
        pair_residual_std_error = data.get("score_rank_regression_residual_std_error")
        pair_residual_std_error_display = (
            f"{pair_residual_std_error:.3f}"
            if isinstance(pair_residual_std_error, (int, float))
            else "brak"
        )
        lines.append(
            (
                "- {label}: liczba presetów={count} ({share}), z oceną ryzyka={scored} "
                "(pokrycie={score_cov}), z rankiem={ranked} (pokrycie={rank_cov}), "
                "średni wynik ryzyka={avg}, mediana wyniku={median_risk}, σ wyniku={std_dev}, CV wyniku={score_cv}, "
                "skośność wyniku={score_skew}, kurtoza wyniku={score_kurtosis}, JB wyniku={score_jb}, "
                "min={min_risk}, max={max_risk}, P10 wyniku={score_p10}, Q1 wyniku={score_p25}, Q3 wyniku={score_p75}, "
                "P90 wyniku={score_p90}, IQR wyniku={score_iqr}, wariancja wyniku={score_variance}, "
                "MAD wyniku={score_mad}, zakres wyniku={score_range}, średni rank={avg_rank}, mediana ranku={median_rank}, "
                "P10 ranku={rank_p10}, Q1 ranku={rank_p25}, Q3 ranku={rank_p75}, P90 ranku={rank_p90}, "
                "IQR ranku={rank_iqr}, σ ranku={rank_std}, CV ranku={rank_cv}, skośność ranku={rank_skew}, kurtoza ranku={rank_kurtosis}, JB ranku={rank_jb}, "
                "wariancja ranku={rank_variance}, MAD ranku={rank_mad}, "
                "zakres ranku={rank_range}, najlepszy rank={rank}, najgorszy rank={worst}, "
                "top preset={top}, dolny preset={bottom}, "
                "pary wynik-rank={pair_count}, kowariancja wynik-rank={pair_cov}, "
                "Pearson wynik-rank={pair_pearson}, Spearman wynik-rank={pair_spearman}, "
                "nachylenie regresji wynik→rank={pair_slope}, "
                "wyraz wolny regresji={pair_intercept}, R^2 regresji={pair_r_squared}, "
                "bias regresji wynik→rank={pair_bias}, MAE regresji={pair_mae}, "
                "MSE regresji={pair_mse}, RMSE regresji={pair_rmse}, "
                "wariancja reszt regresji={pair_residual_variance}, "
                "σ reszt regresji={pair_residual_std_error}"
            ).format(
                label=label,
                count=data.get("count", 0),
                scored=scored,
                ranked=ranked,
                avg=avg_display,
                median_risk=median_risk_display,
                std_dev=std_display,
                min_risk=min_display,
                max_risk=max_display,
                rank=rank_display,
                worst=worst_display,
                top=top_preset,
                bottom=bottom_preset,
                share=share_display,
                avg_rank=avg_rank_display,
                score_cov=score_cov_display,
                rank_cov=rank_cov_display,
                median_rank=median_rank_display,
                score_p25=score_p25_display,
                score_p75=score_p75_display,
                score_p10=score_p10_display,
                score_p90=score_p90_display,
                score_iqr=score_iqr_display,
                score_variance=score_variance_display,
                score_mad=score_mad_display,
                score_range=score_range_display,
                score_cv=score_cv_display,
                score_skew=score_skew_display,
                score_kurtosis=score_kurtosis_display,
                score_jb=score_jb_display,
                rank_p25=rank_p25_display,
                rank_p75=rank_p75_display,
                rank_p10=rank_p10_display,
                rank_p90=rank_p90_display,
                rank_iqr=rank_iqr_display,
                rank_std=rank_std_display,
                rank_variance=rank_variance_display,
                rank_mad=rank_mad_display,
                rank_range=rank_range_display,
                rank_cv=rank_cv_display,
                rank_skew=rank_skew_display,
                rank_kurtosis=rank_kurtosis_display,
                rank_jb=rank_jb_display,
                pair_count=pair_count,
                pair_cov=pair_cov_display,
                pair_pearson=pair_pearson_display,
                pair_spearman=pair_spearman_display,
                pair_slope=pair_slope_display,
                pair_intercept=pair_intercept_display,
                pair_r_squared=pair_r_squared_display,
                pair_bias=pair_bias_display,
                pair_mae=pair_mae_display,
                pair_mse=pair_mse_display,
                pair_rmse=pair_rmse_display,
                pair_residual_variance=pair_residual_variance_display,
                pair_residual_std_error=pair_residual_std_error_display,
            )
        )
    return "\n".join(lines)


def _export_risk_summary(summary: Dict[str, Dict[str, object]], destination: str | Path) -> Path:
    path = Path(destination)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - błędy systemu plików
        raise SystemExit(f"Nie można zapisać pliku podsumowania ryzyka: {exc}")
    return path


async def _run_async(args: argparse.Namespace) -> int:
    key = _load_key(args)
    manager = ConfigManager(Path(args.config_path), encryption_key=key)
    if args.marketplace_dir:
        manager.set_marketplace_directory(Path(args.marketplace_dir))

    await manager.load_config()
    try:
        config = manager.apply_marketplace_preset(
            args.preset_id,
            actor=args.actor,
            user_confirmed=args.confirm_live,
            note=args.note,
        )
    except (ConfigError, ValidationError) as exc:
        print(f"Błąd zastosowania presetu: {exc}")
        return 1

    summary: Dict[str, Dict[str, object]] | None = None
    if args.print_risk_summary or args.export_risk_summary:
        summary = manager.get_marketplace_risk_summary()

    if args.print_risk_summary and summary is not None:
        print(_format_risk_summary(summary))

    if args.export_risk_summary and summary is not None:
        _export_risk_summary(summary, args.export_risk_summary)

    overrides = _parse_overrides(args.set or [])
    for section, values in overrides.items():
        section_payload = config.get(section)
        if not isinstance(section_payload, dict):
            section_payload = {}
            config[section] = section_payload
        section_payload.update(values)

    override_note = _summarise_overrides(overrides) if overrides else None
    combined_note = args.note
    if override_note:
        combined_note = f"{args.note}; overrides: {override_note}" if args.note else f"overrides: {override_note}"

    await manager.save_config(
        config,
        actor=args.actor,
        preset_id=args.preset_id,
        note=combined_note,
        source="editor",
    )

    print(
        "Zapisano preset '{preset}' (wersja: {version}) do pliku {path}".format(
            preset=args.preset_id,
            version=config.get("strategy", {}).get("preset", "custom"),
            path=manager.config_path,
        )
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edytor presetów marketplace (CLI)")
    parser.add_argument("--config-path", required=True, help="Ścieżka do pliku konfiguracji YAML")
    parser.add_argument("--preset-id", required=True, help="Identyfikator presetu z marketplace")
    parser.add_argument("--marketplace-dir", help="Katalog z plikami presetów marketplace (opcjonalnie)")
    parser.add_argument("--encryption-key", help="Klucz Fernet w formacie base64")
    parser.add_argument("--encryption-key-file", help="Plik zawierający klucz Fernet")
    parser.add_argument("--set", action="append", help="Nadpisania sekcji, np. trade.max_open_positions=3")
    parser.add_argument("--actor", required=True, help="Adres e-mail lub identyfikator użytkownika wykonującego zmianę")
    parser.add_argument("--note", help="Dodatkowa notatka do zapisu audytowego")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Potwierdza świadomą aktywację trybu LIVE (jeśli preset go wymaga)",
    )
    parser.add_argument(
        "--print-risk-summary",
        action="store_true",
        help="Wyświetla agregację ryzyka presetów marketplace przed zapisaniem konfiguracji",
    )
    parser.add_argument(
        "--export-risk-summary",
        help="Zapisuje agregację ryzyka presetów marketplace do wskazanego pliku JSON",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:  # pragma: no cover - obsługa przerwania
        return 130


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
