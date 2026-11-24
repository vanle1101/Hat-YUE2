@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0start_studio.ps1"
