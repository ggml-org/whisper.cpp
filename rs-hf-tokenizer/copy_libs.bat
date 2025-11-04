@echo off
echo Copying native libraries to Android project...

set ANDROID_PROJECT=..\whisper.cpp-master\examples\whisper.android\app\src\main\jniLibs

REM Copy arm64-v8a (aarch64)
if exist target\aarch64-linux-android\release\libhftokenizer.so (
    copy target\aarch64-linux-android\release\libhftokenizer.so %ANDROID_PROJECT%\arm64-v8a\libhftokenizer.so
    echo ✓ Copied arm64-v8a library
) else (
    echo ✗ arm64-v8a library not found
)

REM Copy armeabi-v7a (armv7)
if exist target\armv7-linux-androideabi\release\libhftokenizer.so (
    copy target\armv7-linux-androideabi\release\libhftokenizer.so %ANDROID_PROJECT%\armeabi-v7a\libhftokenizer.so
    echo ✓ Copied armeabi-v7a library
) else (
    echo ✗ armeabi-v7a library not found
)

REM Copy x86
if exist target\i686-linux-android\release\libhftokenizer.so (
    copy target\i686-linux-android\release\libhftokenizer.so %ANDROID_PROJECT%\x86\libhftokenizer.so
    echo ✓ Copied x86 library
) else (
    echo ✗ x86 library not found
)

REM Copy x86_64
if exist target\x86_64-linux-android\release\libhftokenizer.so (
    copy target\x86_64-linux-android\release\libhftokenizer.so %ANDROID_PROJECT%\x86_64\libhftokenizer.so
    echo ✓ Copied x86_64 library
) else (
    echo ✗ x86_64 library not found
)

echo.
echo Native library integration complete!
echo.
echo Libraries copied to:
dir %ANDROID_PROJECT%\*\*.so /s

pause