@echo off
rem Tubby Qlib Console: the workbench in its own window (pywebview), or in the browser as a fallback.
setlocal
cd /d "%~dp0"
set "PY=%~dp0env\qlib\Scripts\python.exe"
set "PYW=%~dp0env\qlib\Scripts\pythonw.exe"
if not exist "%PYW%" (
  powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('The Qlib Python environment is missing. Run scripts\windows\setup.ps1 (see docs/QLIB_SETUP.md), then launch again.', 'Tubby Qlib Console')"
  exit /b 1
)
"%PY%" -c "import webview" >nul 2>&1
if errorlevel 1 goto browser
start "" "%PYW%" -m qlib_yahoo.webapp %*
exit /b 0

:browser
title Tubby Qlib Console - close this window to stop it
echo The app window needs pywebview, so the workbench opens in your browser instead.
echo For the app window run once:  env\qlib\Scripts\python.exe -m pip install -r requirements-qlib-windows.txt
echo Keep this window open while you use the workbench; close it to stop the workbench.
"%PY%" -m qlib_yahoo.webapp --browser %*
