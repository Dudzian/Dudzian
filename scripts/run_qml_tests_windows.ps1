param(
  [string]$QtQpaPlatform = "offscreen",
  [string]$QtOpenGL = "software",
  [string]$QtQuickBackend = "software",
  [string]$CrashDumpDir = "var/crashdumps",
  [string]$ResultsDir = "test-results/qml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:QT_QPA_PLATFORM = $QtQpaPlatform
$env:QT_OPENGL = $QtOpenGL
$env:QT_QUICK_BACKEND = $QtQuickBackend
$prevQtCharts = $env:DUDZIAN_DISABLE_QTCHARTS
if ($QtQpaPlatform -eq "offscreen") {
  $env:DUDZIAN_DISABLE_QTCHARTS = "1"
} else {
  Remove-Item Env:DUDZIAN_DISABLE_QTCHARTS -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force $CrashDumpDir | Out-Null
New-Item -ItemType Directory -Force $ResultsDir | Out-Null

$prevPluginAutoload = $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "0"

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

try {
  $versionOutput = & python -m pytest -q --version 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Error "pytest version check failed: $versionOutput"
    exit $LASTEXITCODE
  }

  $helpOutput = & python -m pytest --help 2>&1
  $supportsBoxedHelp = $false
  $supportsForkedHelp = $false
  if ($LASTEXITCODE -eq 0) {
    $supportsBoxedHelp = [bool]($helpOutput | Select-String -SimpleMatch "--boxed")
    $supportsForkedHelp = [bool]($helpOutput | Select-String -SimpleMatch "--forked")
  } else {
    Write-Warning "Unable to inspect pytest --help output; falling back to installed plugin detection."
  }

  $supportsBoxed = $false
  $supportsForked = $false
  & python -c "import importlib.metadata as md; md.version('pytest-xdist')" 2>$null
  if ($LASTEXITCODE -eq 0) {
    $supportsBoxed = $true
  } else {
    & python -c "import importlib.metadata as md; md.version('pytest-forked')" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $supportsForked = $true
    }
  }

  if ($supportsBoxedHelp -or $supportsForkedHelp) {
    if ($supportsBoxedHelp) {
      $pytestArgs += @("--boxed")
    } elseif ($supportsForkedHelp) {
      $pytestArgs += @("--forked")
    }
  } elseif ($supportsBoxed) {
    $pytestArgs += @("--boxed")
  } elseif ($supportsForked) {
    $pytestArgs += @("--forked")
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
} finally {
  if ($null -ne $prevPluginAutoload) {
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = $prevPluginAutoload
  } else {
    Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
  }
  if ($null -ne $prevQtCharts) {
    $env:DUDZIAN_DISABLE_QTCHARTS = $prevQtCharts
  } else {
    Remove-Item Env:DUDZIAN_DISABLE_QTCHARTS -ErrorAction SilentlyContinue
  }
}
