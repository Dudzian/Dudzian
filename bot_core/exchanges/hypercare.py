"""Narzędzia do publikacji checklist HyperCare dla adapterów giełdowych."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import MutableMapping


@dataclass(slots=True)
class HypercareChecklistExporter:
    """Publikuje checklistę HyperCare wraz z odwołaniem do snapshotu jakości sygnałów."""

    exchange: str
    checklist_id: str
    signed_by: str = "compliance"

    def export(
        self,
        *,
        report_dir: str | Path,
        signal_quality_snapshot: str | Path | None = None,
        daily_csv_dir: str | Path | None = None,
    ) -> tuple[Path, Path | None]:
        report_root = Path(report_dir)
        report_root.mkdir(parents=True, exist_ok=True)

        payload: MutableMapping[str, object] = {
            "exchange": self.exchange,
            "checklist_id": self.checklist_id,
            "signed": True,
            "signed_by": self.signed_by,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if signal_quality_snapshot:
            payload["signal_quality_snapshot"] = str(signal_quality_snapshot)

        json_path = report_root / f"{self.exchange}_hypercare.json"
        tmp_json = json_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_json.replace(json_path)

        csv_path: Path | None = None
        if daily_csv_dir is not None:
            csv_root = Path(daily_csv_dir)
            csv_root.mkdir(parents=True, exist_ok=True)
            csv_path = csv_root / f"{datetime.now(timezone.utc).date()}_hypercare.csv"
            header = "exchange,checklist_id,signed,signed_by,signal_quality_snapshot,generated_at\n"
            row = f"{self.exchange},{self.checklist_id},True,{self.signed_by},{payload.get('signal_quality_snapshot','')},{payload['generated_at']}\n"
            tmp_csv = csv_path.with_suffix(".csv.tmp")
            tmp_csv.write_text(header + row, encoding="utf-8")
            tmp_csv.replace(csv_path)

        return json_path, csv_path


__all__ = ["HypercareChecklistExporter"]
