"""Reporting helpers for unified backtests."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from KryptoLowca.logging_utils import get_logger

from .simulation import BacktestReport, PerformanceMetrics

logger = get_logger(__name__)


def _as_rows(metrics: PerformanceMetrics) -> List[tuple[str, str]]:
    return [
        ("Total return", f"{metrics.total_return_pct:.2f}%"),
        ("CAGR", f"{metrics.cagr_pct:.2f}%"),
        ("Max drawdown", f"{metrics.max_drawdown_pct:.2f}%"),
        ("Sharpe", f"{metrics.sharpe_ratio:.2f}"),
        ("Hit ratio", f"{metrics.hit_ratio_pct:.2f}%"),
        ("Risk of ruin", f"{metrics.risk_of_ruin_pct:.2f}%"),
        ("Fees paid", f"{metrics.fees_paid:.4f}"),
        ("Slippage cost", f"{metrics.slippage_cost:.4f}"),
    ]


def render_html_report(report: BacktestReport, *, title: str = "Backtest report") -> str:
    metrics = report.metrics or PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0)
    metrics_rows = "".join(
        f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in _as_rows(metrics)
    )
    trades_rows = "".join(
        (
            "<tr>"
            f"<td>{trade.direction}</td>"
            f"<td>{trade.entry_time}</td>"
            f"<td>{trade.exit_time}</td>"
            f"<td>{trade.entry_price:.4f}</td>"
            f"<td>{trade.exit_price:.4f}</td>"
            f"<td>{trade.quantity:.4f}</td>"
            f"<td>{trade.pnl:.4f}</td>"
            f"<td>{trade.pnl_pct:.2f}%</td>"
            "</tr>"
        )
        for trade in report.trades
    ) or "<tr><td colspan='8'>No completed trades</td></tr>"
    warnings = "".join(f"<li>{w}</li>" for w in report.warnings)
    params_rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in sorted(report.parameters.items())
    )
    html = f"""
    <html>
      <head>
        <meta charset='utf-8' />
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
          th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
          th {{ background-color: #f5f5f5; }}
          h1 {{ margin-bottom: 0.5rem; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <section>
          <h2>Parameters</h2>
          <table>{params_rows}</table>
        </section>
        <section>
          <h2>Performance</h2>
          <table>{metrics_rows}</table>
        </section>
        <section>
          <h2>Trades</h2>
          <table>
            <tr><th>Direction</th><th>Entry</th><th>Exit</th><th>Entry price</th><th>Exit price</th><th>Quantity</th><th>PNL</th><th>Return</th></tr>
            {trades_rows}
          </table>
        </section>
        <section>
          <h2>Warnings</h2>
          <ul>{warnings or '<li>None</li>'}</ul>
        </section>
      </body>
    </html>
    """
    return html


def export_report(
    report: BacktestReport,
    output_dir: Path,
    *,
    title: str = "Backtest report",
    html_name: str = "report.html",
    pdf_name: str = "report.pdf",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / html_name
    html_content = render_html_report(report, title=title)
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path: Path | None = None
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        pdf_path = output_dir / pdf_name
        canv = canvas.Canvas(str(pdf_path), pagesize=A4)
        text = canv.beginText(40, A4[1] - 50)
        text.textLine(title)
        text.textLine("Performance metrics:")
        for name, value in _as_rows(report.metrics or PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0)):
            text.textLine(f"  {name}: {value}")
        text.textLine("")
        text.textLine(f"Trades: {len(report.trades)}")
        for trade in report.trades[:25]:
            text.textLine(
                f"  {trade.direction} {trade.entry_time:%Y-%m-%d %H:%M} -> {trade.exit_time:%Y-%m-%d %H:%M}"
            )
        canv.drawText(text)
        canv.showPage()
        canv.save()
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("PDF export unavailable; install reportlab for PDF output")

    result: Dict[str, Path] = {"html": html_path}
    if pdf_path is not None:
        result["pdf"] = pdf_path
    return result


__all__ = ["render_html_report", "export_report"]
