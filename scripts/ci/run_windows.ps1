Param(
    [Parameter(Mandatory=$true)][string]$Job,
    [string]$Wheelhouse
)

$ErrorActionPreference = "Stop"
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$artifactRoot = "artifacts/local-ci/$Job/$timestamp"
$logFile = Join-Path $artifactRoot "run.log"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

function Write-Log {
    param([string]$Message)
    $Message | Tee-Object -FilePath $logFile -Append
}

function Run {
    param([string]$Command)
    Write-Log "[run] $Command"
    powershell -Command $Command 2>&1 | Tee-Object -FilePath $logFile -Append
}

function Activate-Venv {
    param([string]$VenvPath)
    if (-Not (Test-Path $VenvPath)) {
        Run "py -3.11 -m venv $VenvPath"
    }
    $activate = Join-Path $VenvPath "Scripts/Activate.ps1"
    . $activate
    Run "python -m pip install --upgrade pip"
}

function Pip-Install {
    param([string]$Args)
    Activate-Venv -VenvPath (".venv-" + $Job)
    $wheelArg = ""
    if ($Wheelhouse -and (Test-Path $Wheelhouse)) {
        $resolved = (Resolve-Path $Wheelhouse).Path
        $wheelArg = "--wheelhouse `"$resolved`""
    }
    Run "python scripts/ci/pip_install.py $wheelArg -- $Args"
}

function Install-DevDeps {
    Pip-Install ".[dev]"
}

function Ui-Packaging-Windows {
    Install-DevDeps
    $qtPrefix = $env:Qt6_DIR
    if (-not $qtPrefix) { $qtPrefix = $env:QT_ROOT_DIR }
    if (-not $qtPrefix) { Write-Log "Qt prefix not set; set Qt6_DIR or QT_ROOT_DIR" }
    $extraQt = ""
    if ($qtPrefix) { $extraQt = "--qt-prefix $qtPrefix" }
    if (-not $env:QT_WINDOWS_MODULES) {
        $env:QT_WINDOWS_MODULES = "qtcharts qtdeclarative qtquickcontrols2 qtshadertools qtimageformats"
    }
    Run "python scripts/packaging/qt_bundle.py --platform windows --build-dir ui/build-windows --install-dir ui/install-windows --artifact-dir $artifactRoot $extraQt"
    Run "Get-ChildItem -Force $artifactRoot"
}

function Prepare-Wheelhouse {
    Activate-Venv -VenvPath (".venv-" + $Job)
    $target = $Wheelhouse
    if (-not $target) { $target = "wheelhouse" }
    Run "python scripts/ci/build_wheelhouse.py --wheelhouse `"$target`" --pyside6-version `"$env:PYSIDE6_VERSION`" --only-binary :all:"
    Run "Get-ChildItem -Force $target"
}

switch ($Job) {
    "ui-packaging-windows" { Ui-Packaging-Windows }
    "prepare-wheelhouse" { Prepare-Wheelhouse }
    default { Write-Log "Unknown job: $Job"; exit 1 }
}

Write-Log "Logs saved to $logFile"
