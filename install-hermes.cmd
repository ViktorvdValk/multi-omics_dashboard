@echo off
rem Double-click to install Hermes Agent + Desktop for the current Windows user.
rem All settings live at the top of install-hermes.ps1.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-hermes.ps1" %*
echo.
pause
