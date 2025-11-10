"""Migrator presetów GUI do konfiguracji Stage6 Core."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from shutil import SameFileError, copy2
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import json
import os
import yaml
from importlib import metadata
from bot_core.runtime.paths import build_desktop_app_paths_from_root
from bot_core.runtime.preset_service import (
    PresetConfigService,
    flatten_secret_payload,
    load_preset,
)
from bot_core.security.file_storage import EncryptedFileSecretStorage


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


_STAGE6_SENSITIVE_FLAGS = {
    "--secret-passphrase",
    "--legacy-security-passphrase",
}


_LEGACY_SECURITY_REMOVAL_MESSAGE = (
    "Obsługa zaszyfrowanych plików SecurityManager została przeniesiona do pakietu "
    "'dudzian-migrate'. Uruchom narzędzie migracyjne z pakietu pomocniczego lub skorzystaj "
    "z instrukcji w docs/migrations/2024-legacy-storage-removal.md."
)


def _sanitise_stage6_invocation(argv: Sequence[str]) -> Dict[str, object]:
    """Zwraca zanonimizowaną reprezentację wywołania CLI migratora Stage6."""

    sanitised: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue

        matched = False
        for flag in _STAGE6_SENSITIVE_FLAGS:
            prefix = f"{flag}="
            if token.startswith(prefix):
                sanitised.append(f"{prefix}***REDACTED***")
                matched = True
                break

        if matched:
            continue

        if token in _STAGE6_SENSITIVE_FLAGS:
            sanitised.append(token)
            sanitised.append("***REDACTED***")
            skip_next = True
            continue

        sanitised.append(token)

    command = " ".join(shlex.quote(item) for item in sanitised)
    return {"argv": sanitised, "command": command}


def _ensure_legacy_security_not_requested(args: argparse.Namespace) -> None:
    if any(
        (
            args.legacy_security_file,
            args.legacy_security_salt,
            args.legacy_security_passphrase,
            args.legacy_security_passphrase_file,
            args.legacy_security_passphrase_env,
        )
    ):
        raise SystemExit(_LEGACY_SECURITY_REMOVAL_MESSAGE)


def _collect_stage6_tool_metadata() -> tuple[Dict[str, object], list[str]]:
    """Zbiera dane audytowe o środowisku uruchomieniowym migratora Stage6."""

    warnings: list[str] = []
    payload: Dict[str, object] = {
        "package": "dudzian-bot",
        "version": None,
        "package_available": False,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "module": __name__,
        "git_commit": None,
        "git_available": False,
        "git_commit_error": None,
    }

    try:
        payload["version"] = metadata.version("dudzian-bot")
        payload["package_available"] = True
    except metadata.PackageNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        if commit:
            payload["git_commit"] = commit
            payload["git_available"] = True
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        payload["git_commit_error"] = str(exc)

    return payload, warnings


def _print_core_diff(destination: Path, original: str | None, updated: str) -> None:
    original_exists = original is not None
    before_label = (
        f"{destination} (przed migracją)"
        if original_exists
        else f"{destination} (nowy plik)"
    )
    after_label = f"{destination} (po migracji)"
    before_lines = (original or "").splitlines(keepends=True)
    after_lines = updated.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before_label,
            tofile=after_label,
        )
    )
    if diff_lines:
        print(f"Podgląd zmian {destination}:")
        print("".join(diff_lines), end="")
    else:
        print(f"Podgląd zmian {destination}: brak różnic.")


def _compute_text_checksum(payload: str) -> str:
    digest = hashlib.sha256()
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _compute_file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file_checksum(path: Path) -> tuple[str | None, str | None]:
    try:
        return _compute_file_checksum(path), None
    except OSError as exc:
        return None, f"Ostrzeżenie: nie udało się obliczyć sumy SHA-256 dla {path}: {exc}"


def _describe_passphrase_args(
    *,
    inline: str | None,
    file: str | None,
    env: str | None,
) -> dict[str, object | None]:
    """Zwraca metadane o pochodzeniu hasła bez ujawniania jego wartości."""

    info: dict[str, object | None] = {
        "provided": bool(inline or file or env),
        "source": None,
        "identifier": None,
    }

    if inline:
        info["source"] = "inline"
    elif file:
        info["source"] = "file"
        info["identifier"] = str(Path(file).expanduser())
    elif env:
        info["source"] = "env"
        info["identifier"] = env

    return info


def _resolve_backup_path(target: Path, candidate: str | None) -> Path:
    if candidate:
        return Path(candidate).expanduser()
    return target.with_name(f"{target.name}.bak")


def _create_backup(source: Path, destination: Path) -> None:
    if not source.exists():
        raise OSError(f"plik źródłowy {source} nie istnieje")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)


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




def _load_secret_payload(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - błąd IO
        raise SystemExit(f"Nie można odczytać pliku sekretów: {exc}")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SystemExit("Plik sekretów ma niepoprawny format JSON/YAML") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit("Plik sekretów musi zawierać mapowanie klucz→wartość")
    return payload


def _resolve_passphrase(args: argparse.Namespace) -> str:
    provided = [
        bool(args.secret_passphrase),
        bool(args.secret_passphrase_env),
        bool(args.secret_passphrase_file),
    ]
    if sum(provided) > 1:
        raise SystemExit(
            "Hasło magazynu sekretów może pochodzić tylko z jednego źródła (parametr, plik lub zmienna środowiskowa)."
        )

    if args.secret_passphrase:
        return args.secret_passphrase
    if args.secret_passphrase_file:
        path = Path(args.secret_passphrase_file).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Nie można odczytać pliku z hasłem: {exc}")
    if args.secret_passphrase_env:
        value = os.environ.get(args.secret_passphrase_env)
        if value:
            return value
        raise SystemExit(
            f"Zmienna środowiskowa {args.secret_passphrase_env} nie została ustawiona lub jest pusta."
        )
    raise SystemExit(
        "Brak hasła do magazynu sekretów. Użyj --secret-passphrase, --secret-passphrase-file lub --secret-passphrase-env."
    )


def _resolve_rotation_passphrase(args: argparse.Namespace, current: str) -> str:
    provided = [
        bool(args.secrets_rotate_passphrase),
        bool(args.secrets_rotate_passphrase_env),
        bool(args.secrets_rotate_passphrase_file),
    ]
    if sum(provided) > 1:
        raise SystemExit(
            "Nowe hasło magazynu sekretów może pochodzić tylko z jednego źródła (parametr, plik lub zmienna środowiskowa)."
        )

    if args.secrets_rotate_passphrase:
        return args.secrets_rotate_passphrase
    if args.secrets_rotate_passphrase_file:
        path = Path(args.secrets_rotate_passphrase_file).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Nie można odczytać pliku z nowym hasłem magazynu sekretów: {exc}") from exc
    if args.secrets_rotate_passphrase_env:
        value = os.environ.get(args.secrets_rotate_passphrase_env)
        if value:
            return value
        raise SystemExit(
            f"Zmienna środowiskowa {args.secrets_rotate_passphrase_env} nie została ustawiona lub jest pusta."
        )
    return current


def _apply_secret_filters(
    entries: Mapping[str, str],
    *,
    include: Sequence[str],
    exclude: Sequence[str],
) -> tuple[dict[str, str], list[str], list[str], list[str]]:
    """Zwraca wpisy po filtrach oraz listę pominiętych i brakujących kluczy/wzorów."""

    include_patterns = [item for item in include if item]
    exclude_patterns = [item for item in exclude if item]

    include_hits: dict[str, bool] = {pattern: False for pattern in include_patterns}

    filtered: dict[str, str] = {}
    skipped_by_include: list[str] = []
    skipped_by_exclude: list[str] = []

    for key, value in entries.items():
        if exclude_patterns and any(fnmatchcase(key, pattern) for pattern in exclude_patterns):
            skipped_by_exclude.append(key)
            continue

        if include_patterns:
            matched = False
            for pattern in include_patterns:
                if fnmatchcase(key, pattern):
                    include_hits[pattern] = True
                    matched = True
            if not matched:
                skipped_by_include.append(key)
                continue

        filtered[key] = value

    missing_includes = sorted(pattern for pattern, matched in include_hits.items() if not matched)
    skipped_by_include.sort()
    skipped_by_exclude.sort()

    return filtered, skipped_by_include, skipped_by_exclude, missing_includes


def _configure_migration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migracja presetów GUI do Stage6 core.yaml"
    )
    parser.add_argument("--core-config", required=True, help="Ścieżka do docelowego pliku core.yaml")
    parser.add_argument("--preset", required=True, help="Preset GUI (JSON/YAML) do zaimportowania")
    parser.add_argument("--profile-name", help="Nazwa profilu ryzyka utworzonego na bazie presetu")
    parser.add_argument("--template-profile", help="Profil bazowy użyty do uzupełnienia brakujących pól")
    parser.add_argument(
        "--runtime-entrypoint",
        default="trading_gui",
        help="Entrypoint runtime, który ma korzystać z nowego profilu",
    )
    parser.add_argument("--output", help="Alternatywna ścieżka zapisu YAML (domyślnie nadpisuje core.yaml)")
    parser.add_argument(
        "--core-backup",
        nargs="?",
        const="",
        help=(
            "Utwórz kopię zapasową pliku konfiguracji przed zapisem. "
            "Można opcjonalnie podać ścieżkę docelową; domyślnie tworzy <plik>.bak"
        ),
    )
    parser.add_argument(
        "--core-diff",
        action="store_true",
        help="Po migracji wypisz diff zmian w core.yaml względem poprzedniej zawartości.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Wyświetl wynik YAML bez zapisywania")
    parser.add_argument("--secrets-input", help="Plik z legacy sekretami (JSON/YAML)")
    parser.add_argument(
        "--secrets-output",
        help="Docelowy zaszyfrowany magazyn sekretów (plik EncryptedFileSecretStorage)",
    )
    parser.add_argument(
        "--secrets-include",
        action="append",
        default=[],
        help=(
            "Ogranicz migrację tylko do wskazanych kluczy lub wzorców glob sekretów "
            "(np. api_*). Opcję można podać wielokrotnie"
        ),
    )
    parser.add_argument(
        "--secrets-exclude",
        action="append",
        default=[],
        help=(
            "Pomiń wybrane klucze lub wzorce glob podczas migracji sekretów (np. *_token). "
            "Opcję można podać wielokrotnie"
        ),
    )
    parser.add_argument(
        "--secrets-preview",
        action="store_true",
        help="Wypisz listę kluczy sekretów zakwalifikowanych do migracji (po filtrach).",
    )
    parser.add_argument("--secret-passphrase", help="Hasło do zaszyfrowanego magazynu sekretów")
    parser.add_argument(
        "--secret-passphrase-env",
        help="Nazwa zmiennej środowiskowej zawierającej hasło magazynu sekretów",
    )
    parser.add_argument(
        "--secret-passphrase-file",
        help="Plik zawierający hasło magazynu sekretów (tekst w UTF-8)",
    )
    parser.add_argument("--secrets-backup", help="Ścieżka zapisu kopii zapasowej magazynu sekretów")
    parser.add_argument(
        "--secrets-backup-stdout",
        action="store_true",
        help="Wypisz kopię zapasową magazynu sekretów na stdout",
    )
    parser.add_argument(
        "--secrets-recover-from",
        help="Plik z kopią zapasową magazynu sekretów do odtworzenia",
    )
    parser.add_argument(
        "--secrets-rotate-passphrase",
        help="Nowe hasło magazynu sekretów po migracji",
    )
    parser.add_argument(
        "--secrets-rotate-passphrase-file",
        help="Plik zawierający nowe hasło magazynu sekretów",
    )
    parser.add_argument(
        "--secrets-rotate-passphrase-env",
        help="Zmienna środowiskowa z nowym hasłem magazynu sekretów",
    )
    parser.add_argument(
        "--secrets-rotate-iterations",
        type=int,
        help="Nowa liczba iteracji PBKDF2 używana do zaszyfrowania magazynu sekretów",
    )
    parser.add_argument(
        "--legacy-security-file",
        help=(
            "(wyłączone) Obsługa plików SecurityManager została przeniesiona do pakietu "
            "'dudzian-migrate' – użycie flagi zakończy się błędem"
        ),
    )
    parser.add_argument(
        "--legacy-security-salt",
        help="(wyłączone) patrz docs/migrations/2024-legacy-storage-removal.md",
    )
    parser.add_argument(
        "--legacy-security-passphrase",
        help="(wyłączone) patrz docs/migrations/2024-legacy-storage-removal.md",
    )
    parser.add_argument(
        "--legacy-security-passphrase-file",
        help="(wyłączone) patrz docs/migrations/2024-legacy-storage-removal.md",
    )
    parser.add_argument(
        "--legacy-security-passphrase-env",
        help="(wyłączone) patrz docs/migrations/2024-legacy-storage-removal.md",
    )
    parser.add_argument(
        "--desktop-root",
        help=(
            "Katalog aplikacji desktopowej. Jeśli podany, domyślnie zapisze sekrety w api_keys.vault"
        ),
    )
    parser.add_argument(
        "--summary-json",
        help="Zapisz podsumowanie migracji do pliku JSON (UTF-8)",
    )
    return parser


def _run_stage6_migration(argv: Sequence[str]) -> int:
    parser = _configure_migration_parser()
    provided_args = list(argv)
    args = parser.parse_args(provided_args)

    _ensure_legacy_security_not_requested(args)

    core_path = Path(args.core_config)
    if not core_path.exists():
        raise SystemExit(f"Plik core.yaml nie istnieje: {core_path}")

    sanitised_invocation = _sanitise_stage6_invocation(provided_args)

    summary_path = Path(args.summary_json).expanduser() if args.summary_json else None
    desktop_paths = None
    if args.desktop_root:
        desktop_paths = build_desktop_app_paths_from_root(args.desktop_root)

    preset = load_preset(args.preset)
    service = PresetConfigService(core_path)
    profile = service.import_gui_preset(
        preset,
        profile_name=args.profile_name,
        template_profile=args.template_profile,
        runtime_entrypoint=args.runtime_entrypoint,
    )

    destination = Path(args.output).expanduser() if args.output else core_path

    original_text: str | None = None
    try:
        if destination.exists():
            original_text = destination.read_text(encoding="utf-8")
    except OSError:
        original_text = None

    backup_request = args.core_backup
    created_backup_path: Path | None = None
    if backup_request is not None:
        if args.dry_run:
            print("Tryb dry-run: pominięto utworzenie kopii zapasowej (--core-backup).")
        else:
            backup_source = destination if destination.exists() else core_path
            backup_path = _resolve_backup_path(
                backup_source,
                None if backup_request == "" else backup_request,
            )
            try:
                _create_backup(backup_source, backup_path)
            except SameFileError as exc:
                raise SystemExit(
                    "Ścieżka kopii zapasowej nie może wskazywać na ten sam plik co konfiguracja."
                ) from exc
            except OSError as exc:
                raise SystemExit(
                    f"Nie udało się utworzyć kopii zapasowej {backup_source}: {exc}"
                ) from exc
            else:
                created_backup_path = backup_path
                print(
                    "Utworzono kopię zapasową {source} → {dest}".format(
                        source=backup_source,
                        dest=backup_path,
                    )
                )

    rendered = service.save(destination=destination, dry_run=args.dry_run)

    rendered_checksum = _compute_text_checksum(rendered)

    if args.dry_run:
        print(rendered)
    else:
        print(
            "Zapisano profil '{name}' w {path}".format(
                name=profile.name,
                path=destination,
            )
        )

    if args.core_diff:
        _print_core_diff(destination, original_text, rendered)

    original_checksum: str | None = None
    if original_text is not None:
        original_checksum = _compute_text_checksum(original_text)

    secrets_input_path = Path(args.secrets_input).expanduser() if args.secrets_input else None
    secrets_output_path: Path | None = None
    used_default_vault = False
    if args.secrets_output:
        secrets_output_path = Path(args.secrets_output).expanduser()
    elif secrets_input_path and desktop_paths is not None:
        secrets_output_path = desktop_paths.secret_vault_file
        used_default_vault = True
        if not args.dry_run:
            print(f"Użyto domyślnego magazynu sekretów: {secrets_output_path}")

    rotation_requested = any(
        (
            args.secrets_rotate_passphrase,
            args.secrets_rotate_passphrase_file,
            args.secrets_rotate_passphrase_env,
            args.secrets_rotate_iterations is not None,
        )
    )
    backup_requested = bool(args.secrets_backup or args.secrets_backup_stdout)
    recover_requested = bool(args.secrets_recover_from)

    secrets_payload: Mapping[str, Any] | None = None
    secrets_source_label: str | None = None
    secrets_source_path: Path | None = None
    include_filters: list[str] = []
    exclude_filters: list[str] = []
    if secrets_input_path:
        secrets_payload = _load_secret_payload(secrets_input_path)
        secrets_source_label = f"plik {secrets_input_path}"
        secrets_source_path = secrets_input_path

    if (
        secrets_output_path is None
        and desktop_paths is not None
        and (secrets_payload is not None or rotation_requested or backup_requested or recover_requested)
    ):
        secrets_output_path = desktop_paths.secret_vault_file
        if not used_default_vault and not args.dry_run:
            print(f"Użyto domyślnego magazynu sekretów: {secrets_output_path}")
        used_default_vault = True

    secret_entries: dict[str, str] | None = None
    skipped_by_include: list[str] = []
    skipped_by_exclude: list[str] = []
    missing_includes: list[str] = []
    if secrets_payload is not None:
        secret_entries = flatten_secret_payload(secrets_payload)
        include_filters = [item.strip() for item in args.secrets_include if item and item.strip()]
        exclude_filters = [item.strip() for item in args.secrets_exclude if item and item.strip()]
        if include_filters or exclude_filters:
            (
                secret_entries,
                skipped_by_include,
                skipped_by_exclude,
                missing_includes,
            ) = _apply_secret_filters(
                secret_entries,
                include=include_filters,
                exclude=exclude_filters,
            )
        if missing_includes:
            print(
                "Nie znaleziono sekretów wymaganych przez --secrets-include: {keys}".format(
                    keys=", ".join(sorted(missing_includes))
                )
            )
        if skipped_by_include:
            print(
                "Pominięto sekrety spoza listy --secrets-include: {keys}".format(
                    keys=", ".join(skipped_by_include)
                )
            )
        if skipped_by_exclude:
            print(
                "Pominięto sekrety oznaczone --secrets-exclude: {keys}".format(
                    keys=", ".join(skipped_by_exclude)
                )
            )

    if (
        secret_entries is not None
        and secrets_output_path is None
        and not args.secrets_preview
        and not args.dry_run
    ):
        parser.error(
            (
                "Do migracji sekretów wymagane są oba parametry: "
                "źródło (--secrets-input) oraz --secrets-output"
            )
        )

    if (rotation_requested or backup_requested or recover_requested) and secrets_output_path is None:
        parser.error(
            "Operacje rotacji/backup/odzyskiwania wymagają wskazania --secrets-output lub --desktop-root."
        )

    if args.secrets_preview:
        if secret_entries is None:
            print("Podgląd sekretów: brak źródła sekretów do migracji.")
        elif secret_entries:
            preview_keys = ", ".join(sorted(secret_entries))
            message = "Podgląd sekretów ({count}): {keys}".format(
                count=len(secret_entries),
                keys=preview_keys,
            )
            if secrets_source_label:
                message += f" (źródło: {secrets_source_label})"
            print(message)
        else:
            message = "Podgląd sekretów: brak wpisów do migracji po filtrach."
            if secrets_source_label:
                message += f" (źródło: {secrets_source_label})"
            print(message)

    entries_count = len(secret_entries) if secret_entries is not None else 0

    storage: EncryptedFileSecretStorage | None = None
    resolved_passphrase: str | None = None
    recovered_from_backup = False
    rotation_performed = False
    backup_path_written: Path | None = None
    backup_stdout_snapshot: str | None = None

    def _current_passphrase() -> str:
        nonlocal resolved_passphrase
        if resolved_passphrase is None:
            resolved_passphrase = _resolve_passphrase(args)
        return resolved_passphrase

    def _ensure_storage() -> EncryptedFileSecretStorage:
        nonlocal storage
        if secrets_output_path is None:
            raise SystemExit("Docelowy magazyn sekretów nie został określony.")
        if storage is None:
            storage = EncryptedFileSecretStorage(secrets_output_path, _current_passphrase())
        return storage

    if args.dry_run:
        if secret_entries is not None:
            if entries_count:
                message = "Tryb dry-run: pominięto zapis {count} sekretów".format(count=entries_count)
                if secrets_output_path is not None:
                    message += " do magazynu {path}".format(path=secrets_output_path)
                else:
                    message += " (nie wskazano --secrets-output)"
                if secrets_source_label:
                    message += f" (źródło: {secrets_source_label})"
            else:
                message = "Tryb dry-run: brak sekretów do zapisania po zastosowaniu filtrów"
                if secrets_output_path is not None:
                    message += " (docelowy magazyn: {path})".format(path=secrets_output_path)
                if secrets_source_label:
                    message += f" (źródło: {secrets_source_label})"
            print(message)
        elif secrets_output_path is not None:
            print(f"Tryb dry-run: pominięto utworzenie magazynu sekretów {secrets_output_path}")
        elif used_default_vault and desktop_paths is not None:
            print(
                "Tryb dry-run: pominięto utworzenie domyślnego magazynu sekretów {path}".format(
                    path=desktop_paths.secret_vault_file
                )
            )
        if recover_requested:
            print("Tryb dry-run: pominięto odtworzenie magazynu sekretów z kopii zapasowej.")
        if rotation_requested:
            print("Tryb dry-run: pominięto rotację hasła magazynu sekretów.")
        if backup_requested:
            print("Tryb dry-run: pominięto zapis kopii zapasowej magazynu sekretów.")
    else:
        if recover_requested and secrets_output_path is not None:
            backup_source_path = Path(args.secrets_recover_from).expanduser()
            try:
                backup_text = backup_source_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise SystemExit(f"Nie udało się odczytać kopii zapasowej magazynu sekretów: {exc}") from exc
            storage = EncryptedFileSecretStorage.recover_from_backup(
                secrets_output_path,
                _current_passphrase(),
                backup_text,
            )
            resolved_passphrase = _current_passphrase()
            recovered_from_backup = True
            print(
                "Odtworzono magazyn sekretów z kopii zapasowej {src} → {dest}".format(
                    src=backup_source_path,
                    dest=secrets_output_path,
                )
            )

        if secret_entries is not None and secrets_output_path is not None:
            if entries_count:
                storage = _ensure_storage()
                for key, value in secret_entries.items():
                    storage.set_secret(key, value)
                print(
                    "Zapisano {count} sekretów do magazynu {path}".format(
                        count=entries_count,
                        path=secrets_output_path,
                    )
                )
                if secrets_source_label:
                    print(f"Źródło sekretów: {secrets_source_label}")
            else:
                print("Pominięto zapis sekretów: brak dopasowanych wpisów (po filtrach).")
                if secrets_source_label:
                    print(f"Źródło sekretów: {secrets_source_label}")
        elif secret_entries is not None and secrets_output_path is None:
            if entries_count == 0:
                print("Pominięto zapis sekretów: brak dopasowanych wpisów (po filtrach).")
                if secrets_source_label:
                    print(f"Źródło sekretów: {secrets_source_label}")
            elif args.secrets_preview:
                message = (
                    "Pominięto zapis {count} sekretów: nie wskazano --secrets-output (tryb podglądu)."
                ).format(count=entries_count)
                if secrets_source_label:
                    message += f" Źródło: {secrets_source_label}."
                print(message)
            else:
                parser.error(
                    (
                        "Do migracji sekretów wymagane są oba parametry: "
                        "źródło (--secrets-input) oraz --secrets-output"
                    )
                )
        elif secret_entries is None and secrets_output_path is not None and not recover_requested:
            # Tworzenie pustego magazynu, jeśli wskazano tylko --secrets-output
            storage = _ensure_storage()

        if rotation_requested and secrets_output_path is not None:
            storage = _ensure_storage()
            current_pass = _current_passphrase()
            new_pass = _resolve_rotation_passphrase(args, current=current_pass)
            storage.rotate_passphrase(new_pass, iterations=args.secrets_rotate_iterations)
            resolved_passphrase = new_pass
            rotation_performed = True
            print("Zrotowano hasło magazynu sekretów.")

        if backup_requested and secrets_output_path is not None:
            storage = _ensure_storage()
            snapshot = storage.export_backup()
            if args.secrets_backup:
                backup_path = Path(args.secrets_backup).expanduser()
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(snapshot, encoding="utf-8")
                backup_path_written = backup_path
                print(f"Zapisano kopię zapasową magazynu sekretów do {backup_path}")
            if args.secrets_backup_stdout:
                print(snapshot)
                backup_stdout_snapshot = snapshot

    secrets_written = 0
    if (
        secret_entries is not None
        and secrets_output_path is not None
        and not args.dry_run
        and entries_count
    ):
        secrets_written = entries_count

    backup_checksum: str | None = None
    warnings: list[str] = []
    if created_backup_path is not None and created_backup_path.exists():
        backup_checksum, warning = _safe_file_checksum(created_backup_path)
        if warning:
            warnings.append(warning)

    secrets_output_checksum: str | None = None
    if (
        secrets_output_path is not None
        and not args.dry_run
        and secrets_output_path.exists()
    ):
        secrets_output_checksum, warning = _safe_file_checksum(secrets_output_path)
        if warning:
            warnings.append(warning)

    secrets_backup_file_checksum: str | None = None
    if backup_path_written is not None and backup_path_written.exists():
        secrets_backup_file_checksum, warning = _safe_file_checksum(backup_path_written)
        if warning:
            warnings.append(warning)

    secrets_backup_inline_checksum: str | None = None
    if backup_stdout_snapshot is not None:
        secrets_backup_inline_checksum = _compute_text_checksum(backup_stdout_snapshot)

    secrets_source_checksum: str | None = None
    if secrets_source_path is not None and secrets_source_path.exists():
        secrets_source_checksum, warning = _safe_file_checksum(secrets_source_path)
        if warning:
            warnings.append(warning)

    output_passphrase_info = _describe_passphrase_args(
        inline=getattr(args, "secret_passphrase", None),
        file=getattr(args, "secret_passphrase_file", None),
        env=getattr(args, "secret_passphrase_env", None),
    )
    output_passphrase_info["used"] = bool(secrets_written or rotation_performed or recovered_from_backup)
    output_passphrase_info["rotated"] = bool(rotation_performed)

    tool_metadata, metadata_warnings = _collect_stage6_tool_metadata()
    warnings.extend(metadata_warnings)

    if summary_path is not None:
        summary_payload = {
            "profile_name": profile.name,
            "runtime_entrypoint": args.runtime_entrypoint,
            "core_config_destination": str(destination),
            "core_backup_requested": backup_request is not None,
            "core_backup_path": str(created_backup_path) if created_backup_path else None,
            "core_backup_checksum": backup_checksum,
            "core_diff_requested": bool(args.core_diff),
            "dry_run": bool(args.dry_run),
            "desktop_root": args.desktop_root or None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "core_original_checksum": original_checksum,
            "core_rendered_checksum": rendered_checksum,
            "cli_invocation": sanitised_invocation,
            "warnings": warnings,
            "tool": tool_metadata,
            "secrets": {
                "source_label": secrets_source_label,
                "source_path": str(secrets_source_path) if secrets_source_path else None,
                "source_checksum": secrets_source_checksum,
                "output_path": str(secrets_output_path) if secrets_output_path else None,
                "planned": entries_count if secret_entries is not None else 0,
                "written": secrets_written,
                "rotation_performed": bool(rotation_performed),
                "rotation_iterations": args.secrets_rotate_iterations,
                "used_default_vault": bool(used_default_vault and secrets_output_path is not None),
                "filters": {
                    "include": include_filters,
                    "exclude": exclude_filters,
                },
                "skipped_by_include": skipped_by_include,
                "skipped_by_exclude": skipped_by_exclude,
                "missing_includes": missing_includes,
                "preview": bool(args.secrets_preview),
                "dry_run_skipped": bool(
                    args.dry_run and secret_entries is not None and entries_count > 0
                ),
                "output_checksum": secrets_output_checksum,
                "backup_file_path": str(backup_path_written) if backup_path_written else None,
                "backup_file_checksum": secrets_backup_file_checksum,
                "backup_stdout_checksum": secrets_backup_inline_checksum,
                "output_passphrase": output_passphrase_info,
                "recovered_from_backup": bool(recovered_from_backup),
            },
        }
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise SystemExit(
                f"Nie udało się zapisać podsumowania migracji: {exc}"
            ) from exc
        else:
            print(f"Zapisano podsumowanie migracji do {summary_path}")

    for warning in warnings:
        print(warning)

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    provided = list(argv) if argv is not None else None
    if provided is None:
        import sys

        provided = sys.argv[1:]

    trigger_flags = {"--core-config", "--preset", "--secrets-input", "--secrets-output"}
    if any(flag in provided for flag in trigger_flags):
        return _run_stage6_migration(provided)
    raise SystemExit(
        "Stage6 migrator wymaga flag --core-config oraz --preset. "
        "Funkcje marketplace zostały usunięte z tej komendy."
    )


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
