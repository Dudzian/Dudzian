@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Dudzian Bot - Windows preview EXE builder DEBUG v2
REM Run from repository root, next to pyproject.toml.
REM Keeps CMD open, logs every command, captures real exit codes.
REM ============================================================

set "REPO_ROOT=%~dp0"
cd /d "%REPO_ROOT%"

set "LOG_DIR=var\tmp\local_windows_preview"
set "LOG_FILE=%LOG_DIR%\build_preview_exe_windows_debug_v2.log"
set "DIST_DIR=dist\preview\windows"
set "WORK_DIR=var\build\preview\pyinstaller\windows"
set "EXE_DIR=%DIST_DIR%\dudzian-bot-preview"
set "EXE_PATH=%EXE_DIR%\dudzian-bot-preview.exe"
set "RESOURCE_FILE=bot_core\ai\_defaults\risk_thresholds.yaml"
set "RESOURCE_DEST=bot_core\ai\_defaults"
set "LAST_OUT=%LOG_DIR%\last_command.out"
set "SMOKE_STDOUT=%LOG_DIR%\smoke_stdout.json"
set "SMOKE_STDERR=%LOG_DIR%\smoke_stderr.txt"
set "IMPORT_PROBE=%LOG_DIR%\import_probe.py"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul

> "%LOG_FILE%" echo === START %DATE% %TIME% ===
>> "%LOG_FILE%" echo Repo root: %REPO_ROOT%
>> "%LOG_FILE%" echo Log file : %LOG_FILE%

echo ============================================================
echo Dudzian Bot - Windows preview EXE DEBUG v2
echo Repo root: %REPO_ROOT%
echo Log file : %LOG_FILE%
echo ============================================================
echo.

echo [0/10] Checking repository root...
if not exist "pyproject.toml" goto :fail_root_pyproject
if not exist "scripts\run_local_bot.py" goto :fail_root_script
if not exist "%RESOURCE_FILE%" goto :fail_root_resource
call :log "Repository root checks OK."

echo [1/10] Selecting Python...
set "PY_CMD="
py -3.11 --version >nul 2>nul
if "%ERRORLEVEL%"=="0" set "PY_CMD=py -3.11"
if not defined PY_CMD (
    py -3 --version >nul 2>nul
    if "%ERRORLEVEL%"=="0" set "PY_CMD=py -3"
)
if not defined PY_CMD (
    python --version >nul 2>nul
    if "%ERRORLEVEL%"=="0" set "PY_CMD=python"
)
if not defined PY_CMD goto :fail_no_python
call :run %PY_CMD% --version
if errorlevel 1 goto :fail_python_version

echo [2/10] Creating / checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    call :run %PY_CMD% -m venv .venv
    if errorlevel 1 goto :fail_venv_create
)
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :fail_venv_missing
call :run "%PY%" --version
if errorlevel 1 goto :fail_venv_python

echo [3/10] Installing / updating build dependencies...
call :run "%PY%" -m pip install -U pip setuptools wheel
if errorlevel 1 goto :fail_pip_bootstrap

call :run "%PY%" -m pip install -e .[dev]
if errorlevel 1 goto :fail_project_install

call :run "%PY%" -m pip install pyinstaller
if errorlevel 1 goto :fail_pyinstaller_install

echo [4/10] Running import sanity check...
> "%IMPORT_PROBE%" echo import importlib
>> "%IMPORT_PROBE%" echo mods = ["numpy", "pandas", "yaml", "cryptography", "grpc", "google.protobuf", "pydantic"]
>> "%IMPORT_PROBE%" echo missing = []
>> "%IMPORT_PROBE%" echo for m in mods:
>> "%IMPORT_PROBE%" echo     try:
>> "%IMPORT_PROBE%" echo         importlib.import_module(m)
>> "%IMPORT_PROBE%" echo     except Exception as exc:
>> "%IMPORT_PROBE%" echo         missing.append((m, repr(exc)))
>> "%IMPORT_PROBE%" echo print("missing:", missing)
>> "%IMPORT_PROBE%" echo raise SystemExit(1 if missing else 0)

call :run "%PY%" "%IMPORT_PROBE%"
if errorlevel 1 goto :fail_import_sanity

echo [5/10] Running safety/readiness contracts...
call :run "%PY%" scripts\safe_exe_preview_readiness.py --json
if errorlevel 1 goto :fail_readiness

call :run "%PY%" scripts\safe_exe_preview_build_plan.py --json
if errorlevel 1 goto :fail_build_plan

call :run "%PY%" scripts\safe_exe_preview_profile_validator.py --json
if errorlevel 1 goto :fail_profile_validator

call :run "%PY%" scripts\safe_exe_preview_command_renderer.py --json
if errorlevel 1 goto :fail_command_renderer

REM security_packaging_readiness may return warning semantics depending on release readiness.
call :run_allowfail "%PY%" scripts\security_packaging_readiness.py --config config\e2e\demo_paper.yml --json

echo [6/10] Cleaning previous local Windows preview build...
if exist "%DIST_DIR%" (
    call :log "Removing %DIST_DIR%"
    rmdir /s /q "%DIST_DIR%" >> "%LOG_FILE%" 2>&1
)
if exist "%WORK_DIR%" (
    call :log "Removing %WORK_DIR%"
    rmdir /s /q "%WORK_DIR%" >> "%LOG_FILE%" 2>&1
)
if exist "dudzian-bot-preview.spec" (
    call :log "Removing old dudzian-bot-preview.spec"
    del /f /q "dudzian-bot-preview.spec" >> "%LOG_FILE%" 2>&1
)
if not exist "%DIST_DIR%" mkdir "%DIST_DIR%" >nul 2>nul
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%" >nul 2>nul

echo [7/10] Building Windows preview EXE with PyInstaller...
echo This can take several minutes.
call :run "%PY%" -m PyInstaller --clean --noconfirm --name dudzian-bot-preview --distpath "%DIST_DIR%" --workpath "%WORK_DIR%" --hidden-import bot_core.runtime.bootstrap --hidden-import bot_core.runtime.pipeline --hidden-import bot_core.runtime.config --hidden-import bot_core.ai._defaults --collect-submodules bot_core.ai._defaults --add-data "%RESOURCE_FILE%;%RESOURCE_DEST%" scripts\run_local_bot.py
if errorlevel 1 goto :fail_pyinstaller_build

echo [8/10] Verifying packaged files...
if not exist "%EXE_PATH%" goto :fail_exe_missing
call :log "EXE exists: %EXE_PATH%"

if not exist "%EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml" (
    call :log "Resource not found in _internal after build. Copying it manually into artifact layout."
    mkdir "%EXE_DIR%\_internal\bot_core\ai\_defaults" >nul 2>nul
    copy /y "%RESOURCE_FILE%" "%EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml" >> "%LOG_FILE%" 2>&1
)

echo.
echo --- Packaged default resource check ---
if exist "%EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml" (
    echo OK: %EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml
    >> "%LOG_FILE%" echo OK: %EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml
) else (
    echo MISSING: %EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml
    >> "%LOG_FILE%" echo MISSING: %EXE_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml
)

echo [9/10] Running preview-plan smoke test...
if exist "%SMOKE_STDOUT%" del /f /q "%SMOKE_STDOUT%" >nul 2>nul
if exist "%SMOKE_STDERR%" del /f /q "%SMOKE_STDERR%" >nul 2>nul

call :log "Running smoke: %EXE_PATH% --mode demo --preview-plan"
"%EXE_PATH%" --mode demo --preview-plan > "%SMOKE_STDOUT%" 2> "%SMOKE_STDERR%"
set "SMOKE_CODE=%ERRORLEVEL%"

echo.
echo ---------------- SMOKE STDOUT ----------------
if exist "%SMOKE_STDOUT%" type "%SMOKE_STDOUT%"
echo.
echo ---------------- SMOKE STDERR ----------------
if exist "%SMOKE_STDERR%" type "%SMOKE_STDERR%"
echo ------------------------------------------------
echo.

>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ---------------- SMOKE STDOUT ----------------
if exist "%SMOKE_STDOUT%" type "%SMOKE_STDOUT%" >> "%LOG_FILE%"
>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ---------------- SMOKE STDERR ----------------
if exist "%SMOKE_STDERR%" type "%SMOKE_STDERR%" >> "%LOG_FILE%"
>> "%LOG_FILE%" echo ------------------------------------------------
call :log "Smoke exit code: %SMOKE_CODE%"

if not "%SMOKE_CODE%"=="0" goto :fail_smoke_exit

findstr /i /c:"Default risk thresholds resource missing" "%SMOKE_STDERR%" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto :fail_smoke_thresholds

findstr /i /c:"bot_core.ai._defaults" "%SMOKE_STDERR%" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto :fail_smoke_defaults

findstr /i /c:"Traceback" "%SMOKE_STDERR%" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto :fail_smoke_traceback

echo [10/10] DONE.
echo.
echo EXE path:
echo %EXE_PATH%
echo.
echo Safe preview command:
echo "%EXE_PATH%" --mode demo --preview-plan
echo.
call :log "SUCCESS. EXE path: %EXE_PATH%"
goto :success_end

REM ------------------------------------------------------------
REM Subroutines
REM ------------------------------------------------------------

:run
echo.
echo ^> %*
>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ^> %*
%* > "%LAST_OUT%" 2>&1
set "RC=%ERRORLEVEL%"
if exist "%LAST_OUT%" type "%LAST_OUT%"
if exist "%LAST_OUT%" type "%LAST_OUT%" >> "%LOG_FILE%"
echo Command exit code: %RC%
>> "%LOG_FILE%" echo Command exit code: %RC%
exit /b %RC%

:run_allowfail
echo.
echo ^> %*
>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ^> %*
%* > "%LAST_OUT%" 2>&1
set "RC=%ERRORLEVEL%"
if exist "%LAST_OUT%" type "%LAST_OUT%"
if exist "%LAST_OUT%" type "%LAST_OUT%" >> "%LOG_FILE%"
echo Non-fatal command exit code: %RC%
>> "%LOG_FILE%" echo Non-fatal command exit code: %RC%
exit /b 0

:log
echo %~1
>> "%LOG_FILE%" echo %~1
exit /b 0

REM ------------------------------------------------------------
REM Failure reasons
REM ------------------------------------------------------------

:fail_root_pyproject
set "FAIL_REASON=pyproject.toml not found. Put this .bat in the repository root."
goto :failure_end

:fail_root_script
set "FAIL_REASON=scripts\run_local_bot.py not found. Wrong directory or incomplete repo."
goto :failure_end

:fail_root_resource
set "FAIL_REASON=Required resource missing in repo: %RESOURCE_FILE%"
goto :failure_end

:fail_no_python
set "FAIL_REASON=Python not found. Install Python 3.11+ and make sure it is on PATH."
goto :failure_end

:fail_python_version
set "FAIL_REASON=Python version check failed."
goto :failure_end

:fail_venv_create
set "FAIL_REASON=Failed to create .venv."
goto :failure_end

:fail_venv_missing
set "FAIL_REASON=Virtualenv Python not found: %PY%"
goto :failure_end

:fail_venv_python
set "FAIL_REASON=Virtualenv Python failed."
goto :failure_end

:fail_pip_bootstrap
set "FAIL_REASON=pip bootstrap failed."
goto :failure_end

:fail_project_install
set "FAIL_REASON=Project dev install failed."
goto :failure_end

:fail_pyinstaller_install
set "FAIL_REASON=PyInstaller install failed."
goto :failure_end

:fail_import_sanity
set "FAIL_REASON=Import sanity check failed."
goto :failure_end

:fail_readiness
set "FAIL_REASON=safe_exe_preview_readiness failed."
goto :failure_end

:fail_build_plan
set "FAIL_REASON=safe_exe_preview_build_plan failed."
goto :failure_end

:fail_profile_validator
set "FAIL_REASON=safe_exe_preview_profile_validator failed."
goto :failure_end

:fail_command_renderer
set "FAIL_REASON=safe_exe_preview_command_renderer failed."
goto :failure_end

:fail_pyinstaller_build
set "FAIL_REASON=PyInstaller build failed. See log above."
goto :failure_end

:fail_exe_missing
set "FAIL_REASON=Build finished, but EXE was not found: %EXE_PATH%"
goto :failure_end

:fail_smoke_exit
set "FAIL_REASON=Smoke test failed with exit code %SMOKE_CODE%."
goto :failure_end

:fail_smoke_thresholds
set "FAIL_REASON=Smoke produced default risk thresholds warning. Resource was not packaged correctly."
goto :failure_end

:fail_smoke_defaults
set "FAIL_REASON=Smoke produced bot_core.ai._defaults warning. Resource was not packaged correctly."
goto :failure_end

:fail_smoke_traceback
set "FAIL_REASON=Smoke produced a Python traceback."
goto :failure_end

:failure_end
echo.
echo ============================================================
echo BUILD FAILED
echo Reason: %FAIL_REASON%
echo Full log: %LOG_FILE%
echo ============================================================
echo.
>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo BUILD FAILED: %FAIL_REASON%
goto :pause_and_exit_failure

:success_end
echo.
echo ============================================================
echo BUILD SUCCESS
echo Full log: %LOG_FILE%
echo ============================================================
goto :pause_and_exit_success

:pause_and_exit_failure
echo.
echo Press any key to close this window.
echo Send me this file:
echo %LOG_FILE%
pause >nul
exit /b 1

:pause_and_exit_success
echo.
echo Press any key to close this window.
echo Send me this file if the EXE still does not open normally:
echo %LOG_FILE%
pause >nul
exit /b 0
