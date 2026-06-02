@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Dudzian UI Preview

rem Keep output readable and keep the window open at the end.
chcp 65001 >nul 2>nul

echo ============================================================
echo  Dudzian - visible PySide6/QML UI preview
echo ============================================================
echo.

set "DEFAULT_REPO=C:\Users\kamil\Documents\GitHub\Dudzian"
set "REPO="

call :try_repo "%DEFAULT_REPO%"
if defined REPO goto repo_found

call :try_repo "%~dp0"
if defined REPO goto repo_found

call :try_repo "%~dp0.."
if defined REPO goto repo_found

call :try_repo "%~dp0..\.."
if defined REPO goto repo_found

echo [UI preview] ERROR: Repo root not found.
echo [UI preview] Put this BAT inside the repo root or edit DEFAULT_REPO:
echo [UI preview] DEFAULT_REPO=%DEFAULT_REPO%
goto fail

:repo_found
cd /d "%REPO%"
if errorlevel 1 goto fail

echo [UI preview] Repo: %CD%

if not exist "pyproject.toml" (
    echo [UI preview] ERROR: pyproject.toml not found in repo root.
    goto fail
)

if not exist "ui\pyside_app\__main__.py" (
    echo [UI preview] ERROR: ui\pyside_app\__main__.py not found.
    goto fail
)

if not exist "ui\config\preview_local.yaml" (
    echo [UI preview] ERROR: ui\config\preview_local.yaml not found.
    echo [UI preview] Pull latest repo changes first.
    goto fail
)

set "PY=%CD%\.venv\Scripts\python.exe"
set "NEED_BOOTSTRAP=0"

if not exist "%PY%" (
    echo [UI preview] Local .venv not found. Creating .venv...
    call :create_venv
    if errorlevel 1 (
        echo [UI preview] ERROR: Could not create .venv.
        echo [UI preview] Install Python 3.12 or 3.11, then run this file again.
        goto fail
    )
    set "NEED_BOOTSTRAP=1"
)

if not exist "%PY%" (
    echo [UI preview] ERROR: .venv\Scripts\python.exe still not found.
    goto fail
)

echo [UI preview] Python executable:
"%PY%" -c "import sys; print(sys.executable)"
if errorlevel 1 goto fail

echo [UI preview] Python version:
"%PY%" --version
if errorlevel 1 goto fail

"%PY%" -c "import PySide6; print('PySide6 OK')" >nul 2>nul
if errorlevel 1 (
    echo [UI preview] PySide6 missing in .venv. Bootstrapping dependencies...
    set "NEED_BOOTSTRAP=1"
)

if "%NEED_BOOTSTRAP%"=="1" (
    echo [UI preview] Installing/updating pip, setuptools, wheel...
    "%PY%" -m pip install -U pip setuptools wheel
    if errorlevel 1 goto fail

    echo [UI preview] Installing project dev dependencies...
    "%PY%" -m pip install -e ".[dev]"
    if errorlevel 1 goto fail

    "%PY%" -c "import PySide6; print('PySide6 OK')" >nul 2>nul
    if errorlevel 1 (
        echo [UI preview] PySide6 still missing after project install. Trying direct PySide6 install...
        "%PY%" -m pip install PySide6
        if errorlevel 1 goto fail
    )
)

"%PY%" -c "import PySide6; print('PySide6 OK')"
if errorlevel 1 (
    echo [UI preview] ERROR: PySide6 import failed after bootstrap.
    goto fail
)

echo.
echo [UI preview] Command:
echo "%PY%" -m ui.pyside_app --config ui/config/preview_local.yaml
echo.
echo [UI preview] Running visible PySide6/QML UI...
echo [UI preview] Close the UI window to return here.
echo.

"%PY%" -m ui.pyside_app --config ui/config/preview_local.yaml
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [UI preview] Exit code: %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%

:create_venv
if exist ".venv\Scripts\python.exe" exit /b 0

where py >nul 2>nul
if not errorlevel 1 (
    echo [UI preview] Trying: py -3.12 -m venv .venv
    py -3.12 -m venv .venv >nul 2>nul
    if exist ".venv\Scripts\python.exe" exit /b 0

    echo [UI preview] Trying: py -3.11 -m venv .venv
    py -3.11 -m venv .venv >nul 2>nul
    if exist ".venv\Scripts\python.exe" exit /b 0

    echo [UI preview] Trying: py -3.13 -m venv .venv
    py -3.13 -m venv .venv >nul 2>nul
    if exist ".venv\Scripts\python.exe" exit /b 0

    echo [UI preview] Trying: py -3.14 -m venv .venv
    py -3.14 -m venv .venv >nul 2>nul
    if exist ".venv\Scripts\python.exe" exit /b 0
)

where python >nul 2>nul
if not errorlevel 1 (
    echo [UI preview] Trying: python -m venv .venv
    python -m venv .venv >nul 2>nul
    if exist ".venv\Scripts\python.exe" exit /b 0
)

exit /b 1

:try_repo
set "CAND=%~f1"
if exist "%CAND%\pyproject.toml" (
    if exist "%CAND%\ui\pyside_app\__main__.py" (
        set "REPO=%CAND%"
    )
)
exit /b 0

:fail
set "EXIT_CODE=1"
echo.
echo [UI preview] FAILED.
echo [UI preview] Exit code: %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%
