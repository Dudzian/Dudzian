param(
  [string]$QtQpaPlatform = "offscreen",
  [string]$QtOpenGL = "software",
  [string]$CrashDumpDir = "var/crashdumps",
  [string]$ResultsDir = "test-results/qml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:QT_QPA_PLATFORM = $QtQpaPlatform
$env:QT_OPENGL = $QtOpenGL

New-Item -ItemType Directory -Force $CrashDumpDir | Out-Null
New-Item -ItemType Directory -Force $ResultsDir | Out-Null

$consoleLog = Join-Path $ResultsDir "pytest-console.log"
$pytestArgs = @(
  "-m", "qml",
  "--maxfail=1",
  "--disable-warnings",
  "--junitxml=$ResultsDir/pytest.xml",
  "--durations=15",
  "--log-file=$ResultsDir/pytest.log",
  "--log-file-level=INFO"
)

$supportsBoxed = $false
$supportsForked = $false
& python -c "import xdist" 2>$null
if ($LASTEXITCODE -eq 0) {
  $supportsBoxed = $true
} else {
  & python -c "import pytest_forked" 2>$null
  if ($LASTEXITCODE -eq 0) {
    $supportsForked = $true
  }
}
if ($supportsBoxed) {
  $pytestArgs += "--boxed"
} elseif ($supportsForked) {
  $pytestArgs += "--forked"
} else {
  Write-Error "pytest isolation unavailable: install pytest-xdist (--boxed) or pytest-forked (--forked)."
  exit 1
}

$procdump = Get-Command procdump.exe -ErrorAction SilentlyContinue
if ($procdump) {
  Write-Host "Running QML tests with ProcDump ($($procdump.Source))."
  $procdumpVersion = (Get-Item $procdump.Source).VersionInfo.FileVersion
  if ($procdumpVersion) {
    Write-Host "ProcDump version: $procdumpVersion"
  }
  $procdumpArgs = @(
    "-accepteula",
    "-ma",
    "-e",
    "-n", "1",
    "-x", $CrashDumpDir,
    "--",
    "python",
    "-m",
    "pytest"
  ) + $pytestArgs
  & $procdump.Source @procdumpArgs 2>&1 | Tee-Object -FilePath $consoleLog
  exit $LASTEXITCODE
}

Write-Warning "ProcDump not found in PATH. Install from https://learn.microsoft.com/sysinternals/downloads/procdump"
Write-Host "Running QML tests without crash dump capture."
& python -m pytest @pytestArgs 2>&1 | Tee-Object -FilePath $consoleLog
exit $LASTEXITCODE
