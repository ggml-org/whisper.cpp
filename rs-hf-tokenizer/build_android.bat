@echo off
echo Building Rust tokenizer for Android...

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul

REM Install Android targets if not already installed
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi 
rustup target add i686-linux-android
rustup target add x86_64-linux-android

REM Build for each architecture
echo Building for arm64-v8a...
cargo build --release --target aarch64-linux-android

echo Building for armeabi-v7a...
cargo build --release --target armv7-linux-androideabi

echo Building for x86...
cargo build --release --target i686-linux-android

echo Building for x86_64...
cargo build --release --target x86_64-linux-android

echo.
echo Build complete! Generated libraries:
echo - target/aarch64-linux-android/release/libhftokenizer.so (arm64-v8a)
echo - target/armv7-linux-androideabi/release/libhftokenizer.so (armeabi-v7a)
echo - target/i686-linux-android/release/libhftokenizer.so (x86)
echo - target/x86_64-linux-android/release/libhftokenizer.so (x86_64)
echo.
echo Next steps:
echo 1. Copy these .so files to your Android project's jniLibs folders
echo 2. Update the Android build.gradle to include the native libraries
echo 3. Test the improved tokenization accuracy!

pause