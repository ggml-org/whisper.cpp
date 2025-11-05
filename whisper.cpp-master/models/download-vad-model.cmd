@echo off

rem Save the original working directory
set "orig_dir=%CD%"

rem Get the script directory
set "script_dir=%~dp0"

rem Check if the script directory contains "\bin\" (case-insensitive)
echo %script_dir% | findstr /i "\\bin\\" >nul
if %ERRORLEVEL%==0 (
  rem If script is in a \bin\ directory, use the original working directory as default download path
  set "default_download_path=%orig_dir%"
) else (
  rem Otherwise, use script directory
  pushd %~dp0
  set "default_download_path=%CD%"
  popd
)

rem Set the root path to be the parent directory of the script
for %%d in (%~dp0..) do set "root_path=%%~fd"

rem Count number of arguments passed to script
set argc=0
for %%x in (%*) do set /A argc+=1

set models=silero-v5.1.2

rem If argc is not equal to 1 or 2, print usage information and exit
if %argc% NEQ 1 (
  if %argc% NEQ 2 (
    echo.
    echo Usage: download-vad-model.cmd model [models_path]
    CALL :list_models
    goto :eof
  )
)

if %argc% EQU 2 (
  set models_path=%2
) else (
  set models_path=%default_download_path%
)

set model=%1

for %%b in (%models%) do (
  if "%%b"=="%model%" (
    CALL :download_model
    goto :eof
  )
)

echo Invalid model: %model%
CALL :list_models
goto :eof

:download_model
echo Downloading vad model %model%...

if exist "%models_path%\\ggml-%model%.bin" (
  echo Model %model% already exists. Skipping download.
  goto :eof
)


PowerShell -NoProfile -ExecutionPolicy Bypass -Command "Start-BitsTransfer -Source https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-%model%.bin -Destination \"%models_path%\\ggml-%model%.bin\""

if %ERRORLEVEL% neq 0 (
  echo Failed to download ggml model %model%
  echo Please try again later or download the original Whisper model files and convert them yourself.
  goto :eof
)

rem Check if 'whisper-cli' is available in the system PATH
where whisper-cli >nul 2>&1
if %ERRORLEVEL%==0 (
  rem If found, suggest 'whisper-cli' (relying on PATH resolution)
  set "whisper_cmd=whisper-cli"
) else (
  rem If not found, suggest the local build version
  set "whisper_cmd=%root_path%\build\bin\Release\whisper-cli.exe"
)

echo Done! Model %model% saved in %models_path%\ggml-%model%.bin
echo You can now use it like this:
echo %whisper_cmd% -vm %models_path%\ggml-%model%.bin --vad -m models/ggml-base.en.bin -f samples\jfk.wav

goto :eof

:list_models
  echo.
  echo Available models:
  (for %%a in (%models%) do (
    echo %%a
  ))
  echo.
  exit /b
