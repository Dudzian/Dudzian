@echo off
setlocal EnableExtensions
cd /d "%~dp0\..\.."
echo [UI preview] Repo: %CD%
python --version
echo [UI preview] Command: python -m ui.pyside_app --config ui/config/preview_local.yaml
echo [UI preview] Running visible PySide6/QML UI...
python -m ui.pyside_app --config ui/config/preview_local.yaml
set EXIT_CODE=%ERRORLEVEL%
echo [UI preview] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
