[CmdletBinding()]
param(
    [switch]$IncludeTypes
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$qualityPython = Join-Path $repoRoot ".venv-quality\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $qualityPython) { $qualityPython } else { "python" }

Push-Location $repoRoot
try {
    & $python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $changedFiles = @(
        git -c "safe.directory=$($repoRoot.Replace('\', '/'))" diff --name-only --diff-filter=ACMR HEAD --
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
    if ($changedFiles.Count -gt 0) {
        & $python scripts/quality/betterleaks_hook.py @changedFiles
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($IncludeTypes) {
        & $python -m mypy
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
}
