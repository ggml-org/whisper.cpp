<#
.SYNOPSIS
    Local build script for whisper-cpp-amd. Mirrors the GitHub Actions build.yml jobs for Windows.

.DESCRIPTION
    Builds one or more AMD backends locally, producing the same zip artifacts that CI publishes.

.PARAMETER Backend
    Which backend to build: cpu, vulkan, rocm, npu, all. Default: cpu

.PARAMETER GfxTarget
    ROCm GPU target. Default: gfx1151
    Common: gfx1151, gfx1150, gfx1100, gfx1200

.PARAMETER RocmVersion
    ROCm version to download. Default: 7.12.0

.PARAMETER OutputDir
    Directory for final zip artifacts. Default: .\dist

.PARAMETER BuildDir
    CMake build directory prefix. Default: .\build-local

.PARAMETER Version
    Version string used in artifact filenames. Default: local

.EXAMPLE
    .\scripts\local-build.ps1 -Backend cpu
    .\scripts\local-build.ps1 -Backend vulkan
    .\scripts\local-build.ps1 -Backend rocm -GfxTarget gfx1151
    .\scripts\local-build.ps1 -Backend npu
    .\scripts\local-build.ps1 -Backend all -Version 1.8.4
#>

param(
    [ValidateSet("cpu","vulkan","rocm","npu","all")]
    [string]$Backend      = "cpu",
    [string]$GfxTarget    = "gfx1151",
    [string]$RocmVersion  = "7.12.0",
    [string]$OutputDir    = ".\dist",
    [string]$BuildDir     = ".\build-local",
    [string]$Version      = "local"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
}

function Write-Ok([string]$msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green  }
function Write-Info([string]$msg) { Write-Host "  -->  $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "  [X]  $msg" -ForegroundColor Red    }

function Require-Command([string]$cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Fail "$cmd not found in PATH"
        throw "Missing requirement: $cmd"
    }
    Write-Ok "$cmd found"
}

function Download-SDL2 {
    param([string]$Ver = "2.28.5")
    $sdlDir = "SDL2-$Ver"
    if (Test-Path $sdlDir) {
        Write-Info "SDL2 already extracted at $sdlDir"
    } else {
        Write-Info "Downloading SDL2 $Ver ..."
        $url = "https://github.com/libsdl-org/SDL/releases/download/release-$Ver/SDL2-devel-$Ver-VC.zip"
        Invoke-WebRequest -Uri $url -OutFile "sdl2.zip"
        7z x sdl2.zip -y | Out-Null
        Remove-Item sdl2.zip

        # Patch SDL_endian.h (needed for AMD clang compatibility)
        $hdr = Get-ChildItem -Recurse -Filter "SDL_endian.h" | Select-Object -First 1
        if ($hdr) {
            $content = Get-Content $hdr.FullName -Raw
            if ($content -match 'extern void _m_prefetch') {
                $patched = $content -replace 'extern void _m_prefetch\(void \*__P\);', '// extern void _m_prefetch(void *__P);'
                Set-Content -Path $hdr.FullName -Value $patched -NoNewline
                Write-Ok "Patched SDL_endian.h"
            }
        }
    }
    $cmake = Get-ChildItem -Recurse -Filter "sdl2-config.cmake" | Select-Object -First 1
    if (-not $cmake) { throw "sdl2-config.cmake not found after SDL2 extraction" }
    return $cmake.DirectoryName
}

function Package-Build {
    param([string]$Name, [string]$BinPath)
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $zip = Join-Path $OutputDir "$Name.zip"
    Write-Info "Creating $zip ..."
    Compress-Archive -Path "$BinPath\*" -DestinationPath $zip -Force
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 2)
    Write-Ok "Created $zip ($mb MB)"
    return $zip
}

function Run-MSBuild {
    param([string]$Dir, [string[]]$ConfigArgs, [string]$Config = "Release", [string]$Arch = "x64")
    Write-Info "CMake configure ..."
    & cmake -S . -B $Dir @ConfigArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }
    Write-Info "MSBuild $Config ..."
    & cmake --build $Dir --config $Config -j $env:NUMBER_OF_PROCESSORS
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }
}

# ── Preflight ─────────────────────────────────────────────────────────────────

if (-not (Test-Path "CMakeLists.txt") -or -not (Test-Path "src\whisper.cpp")) {
    Write-Fail "Run this script from the whisper-cpp-amd repo root."
    exit 1
}

Require-Command cmake
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# ── Build functions ───────────────────────────────────────────────────────────

function Build-CPU {
    Write-Step "CPU - Windows x64"
    Require-Command msbuild

    $SDL2_DIR = Download-SDL2
    $dir = "$BuildDir-cpu"

    Run-MSBuild $dir @(
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DWHISPER_SDL2=ON",
        "-DSDL2_DIR=$SDL2_DIR"
    )

    $sdl2dll = Get-ChildItem -Path "SDL2-*\lib\x64\SDL2.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sdl2dll) { Copy-Item $sdl2dll.FullName "$dir\bin\Release\" -Force }

    $zip = Package-Build "whisper-$Version-windows-cpu-x64" "$dir\bin\Release"
    Write-Ok "CPU build done. Artifact: $zip"
}

function Build-Vulkan {
    Write-Step "Vulkan - Windows x64"
    Require-Command msbuild

    # Locate Vulkan SDK
    $VULKAN_SDK = $env:VULKAN_SDK
    if (-not $VULKAN_SDK) {
        $sdkDir = Get-ChildItem "C:\VulkanSDK" -ErrorAction SilentlyContinue |
                  Sort-Object Name -Descending | Select-Object -First 1
        if (-not $sdkDir) {
            Write-Fail "Vulkan SDK not found. Install from https://vulkan.lunarg.com/sdk/home"
            throw "Missing Vulkan SDK"
        }
        $VULKAN_SDK = $sdkDir.FullName
    }
    Write-Ok "Vulkan SDK: $VULKAN_SDK"

    $SDL2_DIR = Download-SDL2
    $dir = "$BuildDir-vulkan"

    Run-MSBuild $dir @(
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_VULKAN=ON",
        "-DWHISPER_SDL2=ON",
        "-DSDL2_DIR=$SDL2_DIR",
        "-DVULKAN_SDK=$VULKAN_SDK"
    )

    $sdl2dll = Get-ChildItem -Path "SDL2-*\lib\x64\SDL2.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sdl2dll) { Copy-Item $sdl2dll.FullName "$dir\bin\Release\" -Force }

    $zip = Package-Build "whisper-$Version-windows-vulkan-x64" "$dir\bin\Release"
    Write-Ok "Vulkan build done. Artifact: $zip"
}

function Build-ROCm {
    Write-Step "ROCm - Windows x64 (target: $GfxTarget)"
    Require-Command ninja

    # ── Download ROCm tarball ──────────────────────────────────────────────
    $rocmRoot = "C:\opt\rocm"
    if (-not (Test-Path "$rocmRoot\bin\amdclang.exe")) {
        Write-Info "Downloading ROCm $RocmVersion for $GfxTarget (2-4 GB, takes a few minutes) ..."

        # Replicate resolve-rocm-version.sh: group targets use gfx1151 as the base tarball
        $baseTarget = $GfxTarget
        if ($GfxTarget -in @("gfx110X","gfx120X","gfx1150","gfx1100")) {
            $baseTarget = "gfx1151"
        }
        $tarballUrl = "https://repo.amd.com/rocm/tarball/therock-dist-windows-${baseTarget}-${RocmVersion}.tar.gz"
        Write-Info "URL: $tarballUrl"

        Invoke-WebRequest -Uri $tarballUrl -OutFile rocm.tar.gz
        New-Item -ItemType Directory -Force -Path $rocmRoot | Out-Null
        & tar -xzf rocm.tar.gz -C $rocmRoot --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "ROCm extraction failed" }
        Remove-Item rocm.tar.gz
        Write-Ok "ROCm extracted to $rocmRoot"
    } else {
        Write-Ok "ROCm already present at $rocmRoot"
    }

    # ── Map GFX target (mirrors map-gpu-target.sh) ─────────────────────────
    $mappedTarget = switch ($GfxTarget) {
        "gfx110X" { "gfx1100;gfx1101;gfx1102" }
        "gfx120X" { "gfx1200;gfx1201" }
        default   { $GfxTarget }
    }
    Write-Info "GPU target: $GfxTarget -> $mappedTarget"

    $SDL2_DIR = Download-SDL2

    # ── Set ROCm env ──────────────────────────────────────────────────────
    $env:HIP_PATH     = $rocmRoot
    $env:HIP_PLATFORM = "amd"
    $env:PATH         = "$rocmRoot\bin;$rocmRoot\lib\llvm\bin;$env:PATH"

    # ── Configure ─────────────────────────────────────────────────────────
    $dir = "$BuildDir-rocm-$GfxTarget"
    Write-Info "CMake configure (Ninja Multi-Config) ..."
    & cmake -S . -B $dir `
        -G "Ninja Multi-Config" `
        "-DGPU_TARGETS=$mappedTarget" `
        -DGGML_HIP=ON `
        "-DCMAKE_C_COMPILER=$rocmRoot/lib/llvm/bin/amdclang.exe" `
        "-DCMAKE_CXX_COMPILER=$rocmRoot/lib/llvm/bin/amdclang++.exe" `
        "-DCMAKE_HIP_COMPILER=$rocmRoot/lib/llvm/bin/amdclang++.exe" `
        "-DCMAKE_C_FLAGS=-D__PRFCHWINTRIN_H" `
        "-DCMAKE_CXX_FLAGS=-D__PRFCHWINTRIN_H" `
        "-DCMAKE_HIP_FLAGS=--rocm-path=$rocmRoot" `
        "-DCMAKE_PREFIX_PATH=$rocmRoot" `
        -DCMAKE_BUILD_TYPE=Release `
        -DBUILD_SHARED_LIBS=ON `
        -DWHISPER_SDL2=ON `
        "-DSDL2_DIR=$SDL2_DIR"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

    Write-Info "Building ..."
    & cmake --build $dir --config Release -j $env:NUMBER_OF_PROCESSORS
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }

    # ── Copy ROCm DLLs ────────────────────────────────────────────────────
    $binOut = "$dir\bin\Release"
    $rocBin = "$rocmRoot\bin"
    Write-Info "Copying ROCm DLLs ..."
    @("amdhip64_*.dll","amd_comgr*.dll","libhipblas.dll","rocblas.dll",
      "rocsolver.dll","hipblaslt.dll","libhipblaslt.dll","hipblas.dll") | ForEach-Object {
        Get-ChildItem $rocBin -Name $_ -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item (Join-Path $rocBin $_) (Join-Path $binOut $_) -Force }
    }
    $rocblasLib = Join-Path $rocBin "rocblas\library"
    if (Test-Path $rocblasLib) {
        Copy-Item $rocblasLib -Destination (Join-Path $binOut "rocblas\library") -Recurse -Force
    }
    $hipblasltLib = Join-Path $rocBin "hipblaslt\library"
    if (Test-Path $hipblasltLib) {
        Copy-Item $hipblasltLib -Destination (Join-Path $binOut "hipblaslt\library") -Recurse -Force
    }

    $sdl2dll = Get-ChildItem -Path "SDL2-*\lib\x64\SDL2.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sdl2dll) { Copy-Item $sdl2dll.FullName $binOut -Force }

    $zip = Package-Build "whisper-$Version-windows-rocm-$GfxTarget" $binOut
    Write-Ok "ROCm build done. Artifact: $zip"
}

function Build-NPU {
    Write-Step "NPU (VitisAI / RyzenAI) - Windows x64"
    Require-Command msbuild

    # ── FlexML Runtime ────────────────────────────────────────────────────
    $flexmlDir = Get-ChildItem -Directory | Where-Object { $_.Name -like "flexmlrt*" } | Select-Object -First 1
    if (-not $flexmlDir) {
        Write-Info "Downloading FlexML Runtime ..."
        $url = "https://github.com/lemonade-sdk/whisper.cpp/releases/download/deps/flexmlrt1.7.0-win.zip"
        Invoke-WebRequest -Uri $url -OutFile flexmlrt.zip
        if (-not (Test-Path "flexmlrt.zip") -or (Get-Item "flexmlrt.zip").Length -eq 0) {
            throw "flexmlrt.zip download failed or is empty"
        }
        $mb = [math]::Round((Get-Item "flexmlrt.zip").Length / 1MB, 2)
        Write-Ok "Downloaded FlexML: $mb MB"

        & tar xvf flexmlrt.zip
        if ($LASTEXITCODE -ne 0) { throw "FlexML extraction failed" }
        Remove-Item flexmlrt.zip

        $flexmlDir = Get-ChildItem -Directory | Where-Object { $_.Name -like "flexmlrt*" } | Select-Object -First 1
        if (-not $flexmlDir) { throw "No flexmlrt directory found after extraction" }
    }
    Write-Ok "FlexML Runtime: $($flexmlDir.FullName)"

    # ── Run setup.bat via a temporary cmd script ───────────────────────────
    # cmd /c with && is not reliable from PowerShell; use a temp .bat file instead
    $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
    $setupPath = Join-Path $flexmlDir.FullName "setup.bat"
    Set-Content -Path $tempBat -Value "@echo off`r`ncall `"$setupPath`"`r`nif errorlevel 1 exit /b 1`r`necho FLEXML_OK"
    Write-Info "Running FlexML setup.bat ..."
    $setupOut = & cmd /c $tempBat 2>&1
    Remove-Item $tempBat -ErrorAction SilentlyContinue

    if ($LASTEXITCODE -ne 0 -or ($setupOut -notmatch "FLEXML_OK")) {
        Write-Fail "FlexML setup.bat failed. Output:"
        $setupOut | ForEach-Object { Write-Host "    $_" }
        throw "FlexML setup failed. Ensure NPU drivers (>= .280) are installed."
    }
    Write-Ok "FlexML environment configured"

    # ── CMake configure + build ───────────────────────────────────────────
    $dir = "$BuildDir-npu"
    Write-Info "CMake configure with -DWHISPER_VITISAI=ON ..."
    & cmake -B $dir -A x64 -DCMAKE_BUILD_TYPE=Release -DWHISPER_VITISAI=ON
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

    Write-Info "Building ..."
    & cmake --build $dir --config Release -j $env:NUMBER_OF_PROCESSORS
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }

    # ── List output ───────────────────────────────────────────────────────
    $binOut = "$dir\bin\Release"
    if (Test-Path $binOut) {
        Write-Info "Build output:"
        Get-ChildItem $binOut | Format-Table Name, Length -AutoSize
    } else {
        throw "Expected output directory $binOut not found"
    }

    # ── Copy FlexML DLLs ─────────────────────────────────────────────────
    Write-Info "Copying FlexML DLLs ..."
    $copied = 0
    foreach ($sub in @("bin", "lib")) {
        $subPath = Join-Path $flexmlDir.FullName $sub
        if (Test-Path $subPath) {
            $dlls = Get-ChildItem "$subPath\*.dll" -ErrorAction SilentlyContinue
            if ($dlls) {
                Copy-Item $dlls.FullName $binOut -Force
                $copied += $dlls.Count
            }
        }
    }
    Write-Ok "Copied $copied FlexML DLLs"

    $zip = Package-Build "whisper-$Version-windows-npu-x64" $binOut
    Write-Ok "NPU build done. Artifact: $zip"
    Write-Info "To run: place the .rai encoder model next to your ggml-*.bin and run whisper-cli.exe normally."
}

# ── Main dispatch ─────────────────────────────────────────────────────────────

$targets = if ($Backend -eq "all") { @("cpu","vulkan","rocm","npu") } else { @($Backend) }
$results = [ordered]@{}

foreach ($t in $targets) {
    try {
        switch ($t) {
            "cpu"    { Build-CPU    }
            "vulkan" { Build-Vulkan }
            "rocm"   { Build-ROCm   }
            "npu"    { Build-NPU    }
        }
        $results[$t] = "[OK]    PASSED"
    } catch {
        Write-Fail "[$t] failed: $_"
        $results[$t] = "[FAIL]  $_"
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Step "Build Summary"
foreach ($t in $targets) {
    $color = if ($results[$t].StartsWith("[OK]")) { "Green" } else { "Red" }
    Write-Host "  $t : $($results[$t])" -ForegroundColor $color
}

Write-Host ""
Write-Host "Artifacts in: $(Resolve-Path $OutputDir)" -ForegroundColor Cyan
if (Test-Path $OutputDir) {
    Get-ChildItem $OutputDir -Filter "*.zip" | ForEach-Object {
        $mb = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $($_.Name) ($mb MB)"
    }
}
