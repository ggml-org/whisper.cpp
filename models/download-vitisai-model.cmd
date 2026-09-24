@echo off
setlocal

set "script=%~dp0download-vitisai-model.ps1"

if "%~1"=="" (
  PowerShell -NoProfile -ExecutionPolicy Bypass -File "%script%"
  exit /b %ERRORLEVEL%
)

if /I "%~1"=="--list" (
  PowerShell -NoProfile -ExecutionPolicy Bypass -File "%script%" -List
  exit /b %ERRORLEVEL%
)

if /I "%~1"=="-l" (
  PowerShell -NoProfile -ExecutionPolicy Bypass -File "%script%" -List
  exit /b %ERRORLEVEL%
)

if /I "%~1"=="list" (
  PowerShell -NoProfile -ExecutionPolicy Bypass -File "%script%" -List
  exit /b %ERRORLEVEL%
)

if "%~2"=="" (
  PowerShell -NoProfile -ExecutionPolicy Bypass -File "%script%" -Model "%~1"
) else (
  PowerShell -NoProfile -ExecutionPolicy Bypass -File "%script%" -Model "%~1" -ModelsPath "%~2"
)

exit /b %ERRORLEVEL%
