@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Dudzian Bot - Windows preview EXE debug builder
REM Run this file from the ROOT of the repository, next to pyproject.toml.
REM This script keeps the CMD window open and writes a full log file.
REM ============================================================

set "REPO_ROOT=%~dp0"
cd /d "%REPO_ROOT%"

set "LOG_DIR=var\tmp\local_windows_preview"
set "LOG_FILE=%LOG_DIR%\build_preview_exe_windows_debug.log"
set "DIST_DIR=dist\preview\windows"
set "WORK_DIR=var\build\preview\pyinstaller\windows"
set "EXE_DIR=%DIST_DIR%\dudzian-bot-preview"
set "EXE_PATH=%EXE_DIR%\dudzian-bot-preview.exe"
set "RESOURCE_FILE=bot_core\ai\_defaults\risk_thresholds.yaml"
set "RESOURCE_DEST=bot_core\ai\_defaults"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul

echo ============================================================
echo Dudzian Bot - Windows preview EXE DEBUG build
echo Repo root: %REPO_ROOT%
echo Log file : %LOG_FILE%
echo ============================================================
echo.

call :log "=== START %DATE% %TIME% ==="
call :log "Repo root: %REPO_ROOT%"
call :log "Log file : %LOG_FILE%"

echo [0/9] Checking repository root...
if not exist "pyproject.toml" (
    call :fail "pyproject.toml not found. Put this .bat in the repository root and run it again."
)
if not exist "scripts\run_local_bot.py" (
    call :fail "scripts\run_local_bot.py not found. Wrong directory or incomplete repo."
)
if not exist "%RESOURCE_FILE%" (
    call :fail "Required resource missing in repo: %RESOURCE_FILE%"
)
call :log "Repository root checks OK."

echo [1/9] Checking Python...
set "PY_CMD="
where py >nul 2>nul
if %ERRORLEVEL%==0 set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 set "PY_CMD=python"
)
if not defined PY_CMD (
    call :fail "Python not found. Install Python 3.11+ and make sure it is on PATH."
)
call :run "%PY_CMD% --version"
if errorlevel 1 call :fail "Python version check failed."

echo [2/9] Creating / checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    call :run "%PY_CMD% -m venv .venv"
    if errorlevel 1 call :fail "Failed to create .venv."
)
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" call :fail "Virtualenv Python not found: %PY%"
call :run ""%PY%" --version"
if errorlevel 1 call :fail "Virtualenv Python failed."

echo [3/9] Installing / updating build dependencies...
call :run ""%PY%" -m pip install -U pip setuptools wheel"
if errorlevel 1 call :fail "pip bootstrap failed."
call :run ""%PY%" -m pip install -e .[dev]"
if errorlevel 1 call :fail "Project dev install failed."
call :run ""%PY%" -m pip install pyinstaller"
if errorlevel 1 call :fail "PyInstaller install failed."

echo [4/9] Running import sanity check...
call :run ""%PY%" -c ""import importlib; mods=['numpy','pandas','yaml','cryptography','grpc','google.protobuf','pydantic']; missing=[]; [missing.append(m) for m in mods if importlib.util.find_spec(m) is None]; print('missing:', missing); raise SystemExit(1 if missing else 0)"""
if errorlevel 1 call :fail "Import sanity check failed. See log above."

echo [5/9] Running safety/readiness contracts...
call :run ""%PY%" scripts\safe_exe_preview_readiness.py --json"
if errorlevel 1 call :fail "safe_exe_preview_readiness failed."
call :run ""%PY%" scripts\safe_exe_preview_build_plan.py --json"
if errorlevel 1 call :fail "safe_exe_preview_build_plan failed."
call :run ""%PY%" scripts\safe_exe_preview_profile_validator.py --json"
if errorlevel 1 call :fail "safe_exe_preview_profile_validator failed."
call :run ""%PY%" scripts\safe_exe_preview_command_renderer.py --json"
if errorlevel 1 call :fail "safe_exe_preview_command_renderer failed."

REM security_packaging_readiness may return warning semantics depending on release readiness.
REM Run it, log it, but do not hard-stop unless Python itself crashes.
call :run_no_fail ""%PY%" scripts\security_packaging_readiness.py --config config\e2e\demo_paper.yml --json"

echo [6/9] Cleaning previous local Windows preview build...
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

echo [7/9] Building Windows preview EXE with PyInstaller...
echo        This may take a while.
call :run ""%PY%" -m PyInstaller --noconfirm --name dudzian-bot-preview --distpath "%DIST_DIR%" --workpath "%WORK_DIR%" --add-data "%RESOURCE_FILE%;%RESOURCE_DEST%" scripts\run_local_bot.py"
if errorlevel 1 call :fail "PyInstaller build failed. See log above."

if not exist "%EXE_PATH%" (
    call :fail "Build finished, but EXE was not found: %EXE_PATH%"
)
call :log "EXE exists: %EXE_PATH%"

echo [8/9] Running preview-plan smoke test...
set "SMOKE_STDOUT=%LOG_DIR%\smoke_stdout.json"
set "SMOKE_STDERR=%LOG_DIR%\smoke_stderr.txt"
if exist "%SMOKE_STDOUT%" del /f /q "%SMOKE_STDOUT%" >nul 2>nul
if exist "%SMOKE_STDERR%" del /f /q "%SMOKE_STDERR%" >nul 2>nul

call :log "Running smoke: %EXE_PATH% --mode demo --preview-plan"
"%EXE_PATH%" --mode demo --preview-plan > "%SMOKE_STDOUT%" 2> "%SMOKE_STDERR%"
set "SMOKE_CODE=%ERRORLEVEL%"

echo.
echo ---------------- SMOKE STDOUT ----------------
type "%SMOKE_STDOUT%"
echo.
echo ---------------- SMOKE STDERR ----------------
type "%SMOKE_STDERR%"
echo ------------------------------------------------
echo.

type "%SMOKE_STDOUT%" >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo --- SMOKE STDERR --- >> "%LOG_FILE%"
type "%SMOKE_STDERR%" >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
call :log "Smoke exit code: %SMOKE_CODE%"

if not "%SMOKE_CODE%"=="0" (
    call :fail "Smoke test failed with exit code %SMOKE_CODE%. Copy the console output or send %LOG_FILE%."
)

findstr /i /c:"Default risk thresholds resource missing" "%SMOKE_STDERR%" >nul 2>nul
if %ERRORLEVEL%==0 (
    call :fail "Smoke produced default risk thresholds warning. Resource was not packaged correctly. Copy the log."
)
findstr /i /c:"bot_core.ai._defaults" "%SMOKE_STDERR%" >nul 2>nul
if %ERRORLEVEL%==0 (
    call :fail "Smoke produced bot_core.ai._defaults warning. Resource was not packaged correctly. Copy the log."
)
findstr /i /c:"Traceback" "%SMOKE_STDERR%" >nul 2>nul
if %ERRORLEVEL%==0 (
    call :fail "Smoke produced a Python traceback. Copy the log."
)

echo [9/9] DONE.
echo.
echo EXE path:
echo %EXE_PATH%
echo.
echo Log file:
echo %LOG_FILE%
echo.
echo Safe preview command:
echo "%EXE_PATH%" --mode demo --preview-plan
echo.
call :log "SUCCESS. EXE path: %EXE_PATH%"
call :end_pause 0

:run
echo.
echo ^> %~1
echo ^> %~1 >> "%LOG_FILE%"
cmd /c %~1 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
exit /b %ERRORLEVEL%

:run_no_fail
echo.
echo ^> %~1
echo ^> %~1 >> "%LOG_FILE%"
cmd /c %~1 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
set "RC=%ERRORLEVEL%"
echo Non-fatal command exit code: %RC%
echo Non-fatal command exit code: %RC% >> "%LOG_FILE%"
exit /b 0

:log
echo %~1 >> "%LOG_FILE%"
exit /b 0

:fail
echo.
echo ============================================================
echo BUILD FAILED
echo Reason: %~1
echo Full log: %LOG_FILE%
echo ============================================================
echo.
echo BUILD FAILED: %~1 >> "%LOG_FILE%"
call :end_pause 1

:end_pause
echo.
echo ============================================================
echo Press any key to close this window.
echo Copy the visible output above, or send this file:
echo %LOG_FILE%
echo ============================================================
pause >nul
exit /b %~1
