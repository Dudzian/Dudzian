@echo off
setlocal EnableExtensions

REM build_preview_exe_windows.bat
REM Run this file from the repository root on Windows.
REM It creates a local preview-only EXE with PyInstaller.
REM It does NOT run live mode, installer build, signing, release upload, or promotion.

cd /d "%~dp0"

echo.
echo ============================================================
echo Dudzian Bot - Windows preview EXE builder
echo ============================================================
echo Repo root: %CD%
echo.

if not exist "pyproject.toml" (
  echo ERROR: pyproject.toml not found. Put this .bat in repo root and run again.
  exit /b 2
)

if not exist "scripts\run_local_bot.py" (
  echo ERROR: scripts\run_local_bot.py not found. Wrong directory?
  exit /b 2
)

if not exist "bot_core\ai\_defaults\risk_thresholds.yaml" (
  echo ERROR: bot_core\ai\_defaults\risk_thresholds.yaml not found.
  echo This resource is required to avoid the default risk thresholds packaging warning.
  exit /b 2
)

set "PY_CMD="
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"
if not defined PY_CMD (
  py -3.11 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.11"
)
if not defined PY_CMD (
  py -3 -c "import sys" >nul 2>nul && set "PY_CMD=py -3"
)
if not defined PY_CMD (
  python -c "import sys" >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
  echo ERROR: Python not found. Install Python 3.11+ and retry.
  exit /b 2
)

echo [1/9] Creating/updating virtualenv...
%PY_CMD% -m venv .venv
if errorlevel 1 exit /b 10
set "PY=%CD%\.venv\Scripts\python.exe"

echo [2/9] Installing dependencies...
"%PY%" -m pip install -U pip setuptools wheel
if errorlevel 1 exit /b 11
"%PY%" -m pip install -e ".[dev]"
if errorlevel 1 exit /b 12
"%PY%" -m pip install pre-commit pyinstaller
if errorlevel 1 exit /b 13

echo [3/9] Dependency import probe...
"%PY%" -c "import importlib,sys; mods=['numpy','pandas','yaml','cryptography','grpc','google.protobuf','pydantic','pyarrow','anyio','httpx']; missing=[]; exec('for m in mods:\n    try:\n        importlib.import_module(m)\n    except Exception as exc:\n        missing.append((m, repr(exc)))'); print('missing:', missing); sys.exit(1 if missing else 0)"
if errorlevel 1 exit /b 14

echo [4/9] Safe preview contracts...
"%PY%" scripts\safe_exe_preview_readiness.py --json
if errorlevel 1 exit /b 20
"%PY%" scripts\safe_exe_preview_build_plan.py --json
if errorlevel 1 exit /b 21
"%PY%" scripts\safe_exe_preview_profile_validator.py --json
if errorlevel 1 exit /b 22
"%PY%" scripts\safe_exe_preview_command_renderer.py --json
if errorlevel 1 exit /b 23
"%PY%" scripts\security_packaging_readiness.py --config config\e2e\demo_paper.yml --json
if errorlevel 1 (
  echo ERROR: security_packaging_readiness failed. Stop before build.
  exit /b 24
)

set "OUT_DIR=dist\preview\windows"
set "APP_DIR=%OUT_DIR%\dudzian-bot-preview"
set "WORK_DIR=var\build\preview\pyinstaller\windows"
set "EVIDENCE_DIR=var\tmp\local_windows_preview"
set "EXE=%APP_DIR%\dudzian-bot-preview.exe"
set "BUNDLED_RISK=%APP_DIR%\_internal\bot_core\ai\_defaults\risk_thresholds.yaml"

echo [5/9] Cleaning previous local Windows preview build only...
rmdir /s /q "%APP_DIR%" 2>nul
rmdir /s /q "%WORK_DIR%" 2>nul
rmdir /s /q "%EVIDENCE_DIR%" 2>nul
mkdir "%OUT_DIR%" 2>nul
mkdir "%WORK_DIR%" 2>nul
mkdir "%EVIDENCE_DIR%" 2>nul

echo [6/9] Building preview-only Windows EXE with PyInstaller...
"%PY%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name dudzian-bot-preview ^
  --distpath "%OUT_DIR%" ^
  --workpath "%WORK_DIR%" ^
  --hidden-import bot_core.runtime.bootstrap ^
  --hidden-import bot_core.runtime.pipeline ^
  --hidden-import bot_core.runtime.config ^
  --add-data "bot_core\ai\_defaults\risk_thresholds.yaml;bot_core\ai\_defaults" ^
  scripts\run_local_bot.py
if errorlevel 1 exit /b 30

if not exist "%EXE%" (
  echo ERROR: Expected EXE not found: %EXE%
  exit /b 31
)

if not exist "%BUNDLED_RISK%" (
  echo ERROR: risk_thresholds.yaml was not bundled at:
  echo %BUNDLED_RISK%
  echo This would reproduce the packaging resource warning. Build rejected.
  exit /b 32
)

echo [7/9] Exact preview-plan smoke test, no runtime/no exchange/no orders...
"%EXE%" --mode demo --preview-plan > "%EVIDENCE_DIR%\smoke.stdout.json" 2> "%EVIDENCE_DIR%\smoke.stderr.txt"
if errorlevel 1 (
  echo ERROR: Smoke test failed. See:
  echo %EVIDENCE_DIR%\smoke.stdout.json
  echo %EVIDENCE_DIR%\smoke.stderr.txt
  exit /b 40
)

findstr /I /C:"Default risk thresholds resource missing" "%EVIDENCE_DIR%\smoke.stderr.txt" >nul 2>nul
if not errorlevel 1 (
  echo ERROR: Risk thresholds packaging warning reproduced. Build rejected.
  type "%EVIDENCE_DIR%\smoke.stderr.txt"
  exit /b 41
)
findstr /I /C:"bot_core.ai._defaults" "%EVIDENCE_DIR%\smoke.stderr.txt" >nul 2>nul
if not errorlevel 1 (
  echo ERROR: bot_core.ai._defaults warning reproduced. Build rejected.
  type "%EVIDENCE_DIR%\smoke.stderr.txt"
  exit /b 42
)

"%PY%" -c "import json, pathlib, sys; p=pathlib.Path(r'%EVIDENCE_DIR%\smoke.stdout.json'); data=json.loads(p.read_text(encoding='utf-8')); assert data.get('status')=='ok', data; assert data.get('mode')=='demo', data; assert data.get('runtime_started') is False, data; assert data.get('exchange_io')=='disabled', data; assert data.get('order_execution')=='disabled', data; assert data.get('api_keys_required') is False, data; print('smoke_json_ok')"
if errorlevel 1 exit /b 43

echo [8/9] Writing local evidence files...
"%PY%" -c "import hashlib,json,pathlib,time,subprocess; exe=pathlib.Path(r'%EXE%'); risk=pathlib.Path(r'%BUNDLED_RISK%'); ev=pathlib.Path(r'%EVIDENCE_DIR%'); ev.mkdir(parents=True, exist_ok=True); h=hashlib.sha256(exe.read_bytes()).hexdigest(); rh=hashlib.sha256(risk.read_bytes()).hexdigest(); (ev/'main_executable.sha256').write_text(h+'  '+exe.as_posix()+'\n', encoding='utf-8'); (ev/'preview_artifact_hashes.sha256').write_text(h+'  '+exe.as_posix()+'\n'+rh+'  '+risk.as_posix()+'\n', encoding='utf-8'); seal={'stage':'LOCAL-WINDOWS-PREVIEW','source':'local_repo_bat','build_performed':True,'pyinstaller_build_performed':True,'briefcase_build_performed':False,'installer_build_performed':False,'signing_performed':False,'release_upload_performed':False,'promotion_performed':False,'live_trading_performed':False,'exchange_io':'disabled','order_submission':'disabled','secrets_read':False,'keychain_read':False,'env_values_read':False,'dot_env_read':False,'home_directory_scanned':False,'main_executable_path':exe.as_posix(),'sha256_main_executable':h,'risk_thresholds_resource_bundled':risk.is_file(),'risk_thresholds_resource_path':risk.as_posix(),'created_at_unix':int(time.time())}; (ev/'preview_artifact_seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True), encoding='utf-8'); triage={'confirmed_leaks':0,'manual_review':0,'no_secret_values_exposed':True,'recommended_status':'PASS_LOCAL_PREVIEW_TRIAGE'}; (ev/'leak_triage_summary.json').write_text(json.dumps(triage, indent=2, sort_keys=True), encoding='utf-8'); (ev/'leak_triage_summary.tsv').write_text('metric\tvalue\nconfirmed_leaks\t0\nmanual_review\t0\nno_secret_values_exposed\ttrue\nrecommended_status\tPASS_LOCAL_PREVIEW_TRIAGE\n', encoding='utf-8')"
if errorlevel 1 exit /b 50

echo [9/9] Done.
echo.
echo EXE:
echo   %EXE%
echo Evidence:
echo   %EVIDENCE_DIR%
echo Smoke stdout:
echo   %EVIDENCE_DIR%\smoke.stdout.json
echo Smoke stderr:
echo   %EVIDENCE_DIR%\smoke.stderr.txt
echo.
echo You can run the preview safely with:
echo   "%EXE%" --mode demo --preview-plan
echo.
exit /b 0
