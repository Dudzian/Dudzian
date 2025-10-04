# setup_mypy.ps1 — uruchamiaj z katalogu repo w aktywnym .venv

# 1) mypy.ini (UTF-8 bez BOM), z exclude w jednej linii
$ini = @'
[mypy]
python_version = 3.10
pretty = True
show_error_codes = True
warn_return_any = True
warn_unused_ignores = True
implicit_reexport = True
namespace_packages = True
mypy_path = .
exclude = (?x)^(?:\.venv/|build/|dist/|KryptoLowca/tests/|KryptoLowca/telemetry_pb\.py)$

[mypy-KryptoLowca.tests.*]
ignore_errors = True

[mypy-KryptoLowca.ai_models]
ignore_errors = True
[mypy-KryptoLowca.trading_strategies.engine]
ignore_errors = True
[mypy-KryptoLowca.data_preprocessor]
ignore_errors = True
[mypy-KryptoLowca.backtest.report]
ignore_errors = True
[mypy-KryptoLowca.reporting]
ignore_errors = True
[mypy-KryptoLowca.dashboard.desktop]
ignore_errors = True

[mypy-pandas.*]
ignore_missing_imports = True
[mypy-matplotlib.*]
ignore_missing_imports = True
[mypy-seaborn.*]
ignore_missing_imports = True
[mypy-plotly.*]
ignore_missing_imports = True
[mypy-reportlab.*]
ignore_missing_imports = True
[mypy-tkinterweb.*]
ignore_missing_imports = True
[mypy-prometheus_client.*]
ignore_missing_imports = True
[mypy-fastapi.*]
ignore_missing_imports = True
[mypy-httpx.*]
ignore_missing_imports = True
[mypy-websockets.*]
ignore_missing_imports = True
[mypy-torch.*]
ignore_missing_imports = True
[mypy-sklearn.*]
ignore_missing_imports = True
[mypy-joblib.*]
ignore_missing_imports = True
[mypy-lightgbm.*]
ignore_missing_imports = True
[mypy-xgboost.*]
ignore_missing_imports = True
[mypy-vectorbt.*]
ignore_missing_imports = True
[mypy-google.protobuf.*]
ignore_missing_imports = True
[mypy-pytest.*]
ignore_missing_imports = True
'@

$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $PWD "mypy.ini"), $ini, $enc)
Write-Host "Saved mypy.ini (UTF-8 no BOM)."

# 2) stuby (idempotent)
pip install -U mypy pandas-stubs types-protobuf | Out-Host
mypy --install-types --non-interactive | Out-Host

# 3) py.typed
$pyTyped = Join-Path $PWD "KryptoLowca\py.typed"
if (-not (Test-Path $pyTyped)) { New-Item -ItemType File -Path $pyTyped | Out-Null }
Write-Host "Ensured KryptoLowca\py.typed."

# 4) opcjonalne skrypty repo (jesli sa)
$patches = @("04_refactor_imports.py","05_apply_hotfixes.py","06_add_types_common.py","08_more_fixes.py")
foreach ($s in $patches) {
  if (Test-Path $s) { Write-Host "Running $s ..."; python $s | Out-Host }
  else { Write-Host "Skipping $s (not found)." }
}

# 5) od-uciekanie wstawionych cudzyslowow w calym pakiecie
$py = @'
import re
from pathlib import Path

ROOT = Path("KryptoLowca")
if not ROOT.exists():
    print("[skip] no KryptoLowca dir")
    raise SystemExit(0)

patts = [
    (re.compile(r"\(\\\""), '("'),   # .get(\"foo\", ...) -> .get("foo", ...)
    (re.compile(r"\(\\\'"), "('"),   # .get(\'foo\', ...) -> .get('foo', ...)
    (re.compile(r"\[\\\""), '["'),   # ["foo\"] -> ["foo"]
    (re.compile(r"\\\"\]"), '"]'),   # ["foo\"] -> ["foo"]
]

changed_total = 0
for py in ROOT.rglob("*.py"):
    if ".venv" in py.parts: 
        continue
    txt = py.read_text(encoding="utf-8")
    new = txt
    for rx, repl in patts:
        new = rx.sub(repl, new)
    if new != txt:
        bak = py.with_suffix(py.suffix + ".bak")
        bak.write_text(txt, encoding="utf-8")
        py.write_text(new, encoding="utf-8")
        print(f"[fix] {py}")
        changed_total += 1

print(f"[fix] files changed: {changed_total}")
'@

$fixFile = Join-Path $PWD "09_fix_escaped_quotes.py"
[System.IO.File]::WriteAllText($fixFile, $py, $enc)
python $fixFile | Out-Host

# 6) mypy
mypy KryptoLowca --python-version 3.10 --pretty --show-error-codes
