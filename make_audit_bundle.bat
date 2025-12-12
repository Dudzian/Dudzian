@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==============================
REM make_audit_bundle.bat
REM Safe: copies selected paths into a separate folder and zips it.
REM No deletes. No modifications to your repo.
REM ==============================

REM ---- Config: what to include ----
set "INCLUDE_DIRS=bot_core ui tests scripts config deploy update .github"
set "INCLUDE_FILES=pyproject.toml poetry.lock requirements.txt requirements-dev.txt requirements-dev.in requirements.in setup.cfg setup.py pytest.ini tox.ini ruff.toml mypy.ini .pre-commit-config.yaml README.md README.txt README.rst"

REM ---- Config: what to exclude (patterns) ----
REM These are applied during ZIP creation and during copy filtering where possible.
set "EXCLUDE_PATTERNS=\.git\; \.venv\; \venv\; __pycache__\; \.pytest_cache\; \.mypy_cache\; \.ruff_cache\; \.idea\; \.vscode\; \dist\; \build\; \node_modules\; \logs\; \log\; \data\; \datasets\; \cache\; \caches\; \artifacts\; \models\; \checkpoints\; \weights\; \output\; \outputs\; \tmp\; \temp\; \downloads\"

REM ---- Mode selection ----
set "MODE=RUN"
if /I "%~1"=="--dry-run" set "MODE=DRY"
if /I "%~1"=="--dryrun" set "MODE=DRY"

REM ---- Determine source repo path ----
set "SRC=%~2"
if "%SRC%"=="" set "SRC=%CD%"

REM If user passed only one arg as a path (without --dry-run), accept it.
if "%~1" NEQ "" (
  if /I "%~1" NEQ "--dry-run" if /I "%~1" NEQ "--dryrun" (
    set "SRC=%~1"
  )
)

REM Normalize
for %%I in ("%SRC%") do set "SRC=%%~fI"

REM Basic check
if not exist "%SRC%\" (
  echo [ERROR] Source path does not exist: "%SRC%"
  echo Usage:
  echo   make_audit_bundle.bat
  echo   make_audit_bundle.bat --dry-run
  echo   make_audit_bundle.bat "C:\path\to\repo"
  echo   make_audit_bundle.bat --dry-run "C:\path\to\repo"
  pause
  exit /b 1
)

REM Check it looks like a repo (optional)
if not exist "%SRC%\.git\" (
  echo [WARN] "%SRC%" does not contain .git\. Continuing anyway...
)

REM ---- Create output folder next to repo ----
REM Put bundles into: <repo_parent>\audit_bundles\<repo_name>_audit_YYYYMMDD_HHMMSS
for %%I in ("%SRC%") do (
  set "REPO_NAME=%%~nI"
  set "REPO_PARENT=%%~dpI"
)
set "REPO_PARENT=%REPO_PARENT:~0,-1%"

REM Timestamp via PowerShell
for /f %%T in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyyMMdd_HHmmss\")"') do set "TS=%%T"

set "OUT_ROOT=%REPO_PARENT%\audit_bundles"
set "BUNDLE_DIR=%OUT_ROOT%\%REPO_NAME%_audit_%TS%"
set "ZIP_PATH=%BUNDLE_DIR%.zip"

echo =========================================================
echo  Trading Bot Audit Bundle Builder
echo  Mode: %MODE%
echo  Source: "%SRC%"
echo  Output folder: "%BUNDLE_DIR%"
echo  Zip: "%ZIP_PATH%"
echo =========================================================
echo.

if /I "%MODE%"=="DRY" (
  echo [DRY-RUN] No files will be copied. Showing what WOULD be included.
  echo.
)

REM ---- Ensure output root ----
if /I "%MODE%"=="RUN" (
  if not exist "%OUT_ROOT%\" mkdir "%OUT_ROOT%" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to create output root: "%OUT_ROOT%"
    pause
    exit /b 1
  )
  if not exist "%BUNDLE_DIR%\" mkdir "%BUNDLE_DIR%" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to create bundle dir: "%BUNDLE_DIR%"
    pause
    exit /b 1
  )
)

REM ---- Function-like label: print exclude list ----
echo Exclude patterns (heuristic):
echo   %EXCLUDE_PATTERNS%
echo.

REM ---- Copy included directories if exist ----
for %%D in (%INCLUDE_DIRS%) do (
  if exist "%SRC%\%%D\" (
    if /I "%MODE%"=="DRY" (
      echo [WOULD COPY DIR] %%D\
    ) else (
      echo [COPY DIR] %%D\
      REM Use robocopy for robust copy; exclude common heavy dirs by name
      REM robocopy exit codes: 0-7 are OK.
      robocopy "%SRC%\%%D" "%BUNDLE_DIR%\%%D" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NC /NS ^
        /XD ".git" ".venv" "venv" "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" ".idea" ".vscode" "dist" "build" "node_modules" "logs" "log" "data" "datasets" "cache" "caches" "artifacts" "models" "checkpoints" "weights" "output" "outputs" "tmp" "temp" "downloads" >nul
      set "RC=!errorlevel!"
      if !RC! GEQ 8 (
        echo [ERROR] robocopy failed for dir %%D with code !RC!
        echo         You can still proceed, but the bundle may be incomplete.
      )
    )
  ) else (
    if /I "%MODE%"=="DRY" (
      REM silence missing dirs in dry run
    ) else (
      REM silence missing dirs in run
    )
  )
)

echo.

REM ---- Copy include files if exist ----
for %%F in (%INCLUDE_FILES%) do (
  if exist "%SRC%\%%F" (
    if /I "%MODE%"=="DRY" (
      echo [WOULD COPY FILE] %%F
    ) else (
      echo [COPY FILE] %%F
      copy /Y "%SRC%\%%F" "%BUNDLE_DIR%\%%F" >nul
    )
  )
)

echo.

REM ---- Create ZIP ----
if /I "%MODE%"=="RUN" (
  echo [ZIP] Creating archive...
  REM PowerShell Compress-Archive doesn't support exclusions well; since we copied a filtered set, it's fine.
  powershell -NoProfile -Command ^
    "if (Test-Path -LiteralPath '%ZIP_PATH%') { Remove-Item -LiteralPath '%ZIP_PATH%' -Force } ; Compress-Archive -Path '%BUNDLE_DIR%\*' -DestinationPath '%ZIP_PATH%' -Force"
  if errorlevel 1 (
    echo [ERROR] Failed to create ZIP via PowerShell.
    echo        Bundle folder still exists: "%BUNDLE_DIR%"
    pause
    exit /b 1
  )
  echo.
  echo ✅ Done.
  echo Bundle folder: "%BUNDLE_DIR%"
  echo ZIP file:      "%ZIP_PATH%"
  echo.
  echo Next: upload the ZIP here (and make sure it contains no secrets).
) else (
  echo.
  echo [DRY-RUN] Done. No changes made.
  echo Tip: run without --dry-run to generate the bundle.
)

echo.
pause
endlocal
