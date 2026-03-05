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
    $wrapped = "`$ErrorActionPreference='Stop'; $Command"
    powershell -NoProfile -Command $wrapped 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Command" }
}

function Activate-Venv {
    param([string]$VenvPath)
    if (-Not (Test-Path $VenvPath)) {
        Run "python -m venv $VenvPath"
    }
    $activate = Join-Path $VenvPath "Scripts/Activate.ps1"
    . $activate
    Run "python -m pip install --upgrade pip"
}

function Pip-Install {
    param([string]$PipArgs)
    Activate-Venv -VenvPath (".venv-" + $Job)
    $wheelArg = ""
    $wheelhousePath = $Wheelhouse
    if (-not $wheelhousePath -and $env:WHEELHOUSE_DIR) { $wheelhousePath = $env:WHEELHOUSE_DIR }
    if ($wheelhousePath -and (Test-Path $wheelhousePath)) {
        $resolved = (Resolve-Path $wheelhousePath).Path
        $wheelArg = "--wheelhouse `"$resolved`""
    }
    Run "python scripts/ci/pip_install.py $wheelArg -- $PipArgs"
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

function Lint-And-Test {
    Install-DevDeps
    Pip-Install "pytest pytest-cov pre-commit"
    Run "python scripts/lint_paths.py"
    Run "pre-commit run --all-files --show-diff-on-failure"
    Run "pytest --cov=bot_core.strategies --cov=bot_core.runtime.multi_strategy_scheduler --cov=bot_core.runtime.journal --cov-config=.coveragerc --cov-report=xml --cov-report=term --cov-fail-under=75 tests/test_pipeline_paper.py tests/test_risk_profiles.py tests/test_mean_reversion_strategy.py tests/test_volatility_target_strategy.py tests/test_cross_exchange_arbitrage_strategy.py tests/test_multi_strategy_scheduler.py tests/test_backtest_dataset_library.py tests/test_telemetry_risk_profiles.py tests/test_trading_decision_journal.py tests/test_smoke_demo_strategies_cli.py tests/runtime/test_stage6_hypercare_cycle_runtime.py tests/runtime/test_multi_strategy_scheduler_async.py tests/test_journal_analysis.py tests/strategies/test_regime_workflow.py tests/strategies/test_day_trading_strategy.py tests/test_futures_spread_strategy.py tests/test_cross_exchange_hedge_strategy.py"
}

function Bot-Core-Fast-Tests {
    Install-DevDeps
    Run "python scripts/lint_paths.py"
    Run "python scripts/generate_trading_stubs.py --skip-cpp"
    $env:PYTEST_FAST = "1"
    Run "mkdir -Force test-results | Out-Null"
    Run "pytest --fast --maxfail=1 --durations=10 --junitxml=test-results/pytest.xml"
}

function Release-Quality-Gates {
    Install-DevDeps
    Run "python scripts/lint_paths.py"
    Pip-Install "mypy"
    Run "mypy"
    Run "pytest -m e2e_demo_paper --maxfail=1 --disable-warnings"
    Run "pytest tests/test_paper_execution.py"
    Run "pytest tests/integration/test_execution_router_failover.py"
}

function Qml-Collect-Only {
    Install-DevDeps
    $output = & python -m pytest --collect-only -q tests/ui/qml/test_risk_panels.py 2>&1
    $status = $LASTEXITCODE
    $output | Tee-Object -FilePath $logFile -Append | Out-Null
    if ($output -match "found no collectors") {
        throw "QML collect-only failed: found no collectors"
    }
    if ($status -ne 0 -and $status -ne 5) {
        throw "QML collect-only failed with status $status"
    }
}

function Ui-Native-Tests {
    Install-DevDeps
    $qtPrefix = $env:Qt6_DIR
    if (-not $qtPrefix) { $qtPrefix = $env:QT_ROOT_DIR }
    if (-not $qtPrefix) { throw "Qt prefix not set; set Qt6_DIR or QT_ROOT_DIR" }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue) -or -not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        Pip-Install "cmake ninja"
    }
    $cmakePrefixArg = "-DCMAKE_PREFIX_PATH=`"$qtPrefix`""
    Run "cmake -S ui -B ui/build-tests -G Ninja -DBUILD_TESTING=ON $cmakePrefixArg"
    Run "cmake --build ui/build-tests"
    Run "ctest --test-dir ui/build-tests --output-on-failure"
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
    "ui-native-tests" { Ui-Native-Tests }
    "lint-and-test" { Lint-And-Test }
    "bot-core-fast-tests" { Bot-Core-Fast-Tests }
    "release-quality-gates" { Release-Quality-Gates }
    "qml-collect-only" { Qml-Collect-Only }
    "prepare-wheelhouse" { Prepare-Wheelhouse }
    default { Write-Log "Unknown job: $Job"; exit 1 }
}

Write-Log "Logs saved to $logFile"
