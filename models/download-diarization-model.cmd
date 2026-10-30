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

set models=q8_0

rem If argc is not equal to 1 or 2, print usage information and exit
if %argc% NEQ 1 (
  if %argc% NEQ 2 (
    echo.
    echo Usage: download-diarization-model.cmd model [models_path]
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
echo Downloading diarization model %model%...

if exist "%models_path%\\Nemotron-3-Diarization.%model%.gguf" (
  echo Model %model% already exists. Skipping download.
  goto :eof
)


PowerShell -NoProfile -ExecutionPolicy Bypass -Command "Start-BitsTransfer -Source https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/Nemotron-3-Diarization.%model%.gguf -Destination \"%models_path%\\Nemotron-3-Diarization.%model%.gguf\""

if %ERRORLEVEL% neq 0 (
  echo Failed to download diarization model %model%
  echo Please try again later or download the model file manually.
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

echo Done! Model %model% saved in %models_path%\Nemotron-3-Diarization.%model%.gguf
echo You can now use it like this:
echo %whisper_cmd% --diarization --diarization-model %models_path%\Nemotron-3-Diarization.%model%.gguf -m models/ggml-base.en.bin -f samples\jfk.wav

goto :eof

:list_models
  echo.
  echo Available models:
  (for %%a in (%models%) do (
    echo %%a
  ))
  echo.
  exit /b
