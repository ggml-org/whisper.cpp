<#
.SYNOPSIS
    Stage and install everything needed to build/run the native-Windows
    whisper-gui (with neural diarization) on an AIR-GAPPED machine.

.DESCRIPTION
    Two-stage workflow:

      1) On a CONNECTED Windows machine, from the repo root:
             .\examples\whisper.gui\whisper-gui-airgap.ps1 -Mode Stage
         Downloads CMake, Python, the Python wheels (sherpa-onnx, numpy),
         ffmpeg, the whisper + diarization models, and a copy of this repo
         into a self-contained bundle folder.

      2) Copy the bundle folder to the OFFLINE target, then from inside it:
             .\whisper-gui-airgap.ps1 -Mode Install
         Installs Python + the wheels offline, places the models, and builds
         whisper-gui.exe with MSVC. No network access required.

    REQUIREMENT ON THE TARGET: Visual Studio 2022 Build Tools with the
    "Desktop development with C++" workload (provides the MSVC compiler that
    CMake uses). Staging the C++ compiler offline is a multi-GB special case;
    see -VsBootstrapper below. Everything else is handled by this script.

.NOTES
    This script has NOT been verified on Windows by its author (it was written
    on Linux). Treat the first Stage/Install run as a shakedown; the build is
    the part most likely to surface issues (it is the first MSVC compile of
    this code). Errors are printed verbatim so you can iterate.
#>

[CmdletBinding()]
param(
    [ValidateSet('Stage', 'Install')]
    [string] $Mode = 'Stage',

    # Bundle directory. Stage writes here; Install reads from here (defaults to
    # the folder this script lives in when run from inside a staged bundle).
    [string] $BundleDir,

    # Python version to stage/install (must have prebuilt sherpa-onnx wheels).
    [string] $PythonVersion = '3.11.9',

    # Tool versions / sources (override if you need a different pin).
    [string] $CMakeVersion = '3.30.5',

    # Skip the (large) ffmpeg download if you don't need video->wav conversion.
    [switch] $SkipFfmpeg,

    # Optional: path to vs_BuildTools.exe. If given, Stage runs an offline
    # --layout into the bundle so the target needs no internet for MSVC either.
    [string] $VsBootstrapper
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------------------------------------------------------------- helpers -----
function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "!!  $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host "XX  $m" -ForegroundColor Red; exit 1 }

function Get-File($Url, $Dest) {
    if (Test-Path $Dest) { Say "have $(Split-Path $Dest -Leaf) (skip)"; return }
    Say "download $Url"
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest
}

function Find-Exe($name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { $c.Source } else { $null }
}

# repo root = two levels up from examples/whisper.gui
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

$GH  = 'https://github.com/k2-fsa/sherpa-onnx/releases/download'
$EMB = "$GH/speaker-recongition-models/nemo_en_titanet_small.onnx"
$SEG = "$GH/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
$WHISPER_MODEL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin'
$CMAKE_URL = "https://github.com/Kitware/CMake/releases/download/v$CMakeVersion/cmake-$CMakeVersion-windows-x86_64.zip"
$PY_URL    = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$FFMPEG_URL = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'

# =============================================================== STAGE =========
function Invoke-Stage {
    if (-not $BundleDir) { $BundleDir = Join-Path (Get-Location) 'whisper-gui-bundle' }
    $BundleDir = [IO.Path]::GetFullPath($BundleDir)
    Say "staging bundle -> $BundleDir"
    foreach ($d in 'tools', 'wheels', 'models', 'repo') {
        New-Item -ItemType Directory -Force -Path (Join-Path $BundleDir $d) | Out-Null
    }

    # --- tools: CMake (portable zip), Python (installer), ffmpeg ---
    Get-File $CMAKE_URL (Join-Path $BundleDir "tools\cmake-$CMakeVersion-windows-x86_64.zip")
    Get-File $PY_URL    (Join-Path $BundleDir "tools\python-$PythonVersion-amd64.exe")
    if (-not $SkipFfmpeg) { Get-File $FFMPEG_URL (Join-Path $BundleDir 'tools\ffmpeg-win64-gpl.zip') }

    # --- python wheels (sherpa-onnx + numpy) for the target's python ---
    $pip = Find-Exe 'python'
    if (-not $pip) { Die "Python/pip not found on this staging machine - install Python $PythonVersion first." }
    $abi = 'cp' + (($PythonVersion -split '\.')[0..1] -join '')   # 3.11.9 -> cp311
    $whl = Join-Path $BundleDir 'wheels'
    Say "downloading wheels for $abi/win_amd64"
    try {
        & python -m pip download sherpa-onnx numpy `
            --only-binary=:all: --platform win_amd64 `
            --python-version (($PythonVersion -split '\.')[0..1] -join '.') `
            --implementation cp --abi $abi -d $whl
        if ($LASTEXITCODE -ne 0) { throw "pip download exit $LASTEXITCODE" }
    } catch {
        Warn "constrained wheel download failed ($_). Falling back to this machine's interpreter -"
        Warn "make sure THIS machine runs Python $PythonVersion so the wheels match the target."
        & python -m pip download sherpa-onnx numpy -d $whl
        if ($LASTEXITCODE -ne 0) { Die "pip download failed." }
    }

    # --- models ---
    Get-File $WHISPER_MODEL (Join-Path $BundleDir 'models\ggml-base.en.bin')
    Get-File $EMB           (Join-Path $BundleDir 'models\nemo_en_titanet_small.onnx')
    Get-File $SEG           (Join-Path $BundleDir 'models\sherpa-onnx-pyannote-segmentation-3-0.tar.bz2')

    # --- repo copy (vendored SDL2 + ImGui already inside; skip build/.git) ---
    Say "copying repo -> bundle\repo"
    $null = robocopy $RepoRoot (Join-Path $BundleDir 'repo') /E /NFL /NDL /NJH /NJS /NP `
                /XD (Join-Path $RepoRoot 'build') (Join-Path $RepoRoot 'build-gui') `
                    (Join-Path $RepoRoot '.git') $BundleDir
    if ($LASTEXITCODE -ge 8) { Die "robocopy failed ($LASTEXITCODE)" }

    # --- optional: offline MSVC Build Tools layout ---
    if ($VsBootstrapper) {
        Say "creating offline VS Build Tools layout (this is several GB)..."
        & $VsBootstrapper --layout (Join-Path $BundleDir 'vs_buildtools') `
            --add Microsoft.VisualStudio.Workload.VCTools `
            --includeRecommended --lang en-US --quiet
        if ($LASTEXITCODE -ne 0) { Warn "VS layout returned $LASTEXITCODE - check it." }
    }

    # --- drop a copy of this script into the bundle so Install runs from there ---
    Copy-Item $PSCommandPath (Join-Path $BundleDir 'whisper-gui-airgap.ps1') -Force

    Say "STAGE COMPLETE."
    Write-Host @"

Bundle ready:  $BundleDir
Copy that whole folder to the air-gapped machine, then run from inside it:

    .\whisper-gui-airgap.ps1 -Mode Install

$(if (-not $VsBootstrapper) { "NOTE: the target must already have Visual Studio 2022 Build Tools with the
C++ workload. To stage MSVC offline too, re-run Stage with:
    -VsBootstrapper C:\path\to\vs_BuildTools.exe" })
"@ -ForegroundColor Green
}

# ============================================================= INSTALL =========
function Invoke-Install {
    if (-not $BundleDir) { $BundleDir = $PSScriptRoot }
    $BundleDir = [IO.Path]::GetFullPath($BundleDir)
    Say "installing from bundle: $BundleDir"
    foreach ($d in 'tools', 'wheels', 'models', 'repo') {
        if (-not (Test-Path (Join-Path $BundleDir $d))) { Die "bundle is missing '$d' - run -Mode Stage first." }
    }

    # --- CMake (portable) onto PATH for this session ---
    $cmakeZip = Get-ChildItem (Join-Path $BundleDir 'tools') -Filter 'cmake-*-windows-x86_64.zip' | Select-Object -First 1
    if (-not $cmakeZip) { Die "no CMake zip in bundle\tools" }
    $cmakeDir = Join-Path $BundleDir 'tools\cmake'
    if (-not (Test-Path $cmakeDir)) { Expand-Archive $cmakeZip.FullName $cmakeDir -Force }
    $cmakeBin = (Get-ChildItem $cmakeDir -Recurse -Filter 'cmake.exe' | Select-Object -First 1).DirectoryName
    if (-not $cmakeBin) { Die "cmake.exe not found after extract" }
    $env:PATH = "$cmakeBin;$env:PATH"
    Say "cmake: $(& cmake --version | Select-Object -First 1)"

    # --- Python (install silently if absent), then offline wheels ---
    if (-not (Find-Exe 'python')) {
        $pyExe = Get-ChildItem (Join-Path $BundleDir 'tools') -Filter 'python-*-amd64.exe' | Select-Object -First 1
        if (-not $pyExe) { Die "no Python installer in bundle\tools" }
        Say "installing Python silently..."
        Start-Process $pyExe.FullName -Wait -ArgumentList `
            '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_pip=1'
        $env:PATH = "$env:LOCALAPPDATA\Programs\Python\Python$((($PythonVersion -split '\.')[0..1] -join ''))\;" +
                    "$env:LOCALAPPDATA\Programs\Python\Python$((($PythonVersion -split '\.')[0..1] -join ''))\Scripts\;$env:PATH"
    }
    if (-not (Find-Exe 'python')) { Die "Python still not on PATH - open a new shell or install manually." }
    Say "pip install (offline) sherpa-onnx + numpy"
    & python -m pip install --no-index --find-links (Join-Path $BundleDir 'wheels') sherpa-onnx numpy
    if ($LASTEXITCODE -ne 0) { Die "offline pip install failed." }
    & python -c "import sherpa_onnx, numpy; print('python deps OK', sherpa_onnx.__version__)"

    # --- place models into repo\models (+ extract segmentation) ---
    $repo = Join-Path $BundleDir 'repo'
    $mdst = Join-Path $repo 'models'
    New-Item -ItemType Directory -Force -Path $mdst | Out-Null
    Copy-Item (Join-Path $BundleDir 'models\ggml-base.en.bin')           $mdst -Force
    Copy-Item (Join-Path $BundleDir 'models\nemo_en_titanet_small.onnx') $mdst -Force
    $segTar = Join-Path $BundleDir 'models\sherpa-onnx-pyannote-segmentation-3-0.tar.bz2'
    if (-not (Test-Path (Join-Path $mdst 'sherpa-onnx-pyannote-segmentation-3-0\model.onnx'))) {
        Say "extracting segmentation model"
        & tar -xf $segTar -C $mdst    # Windows 10+ bundles bsdtar (handles .bz2)
        if ($LASTEXITCODE -ne 0) { Warn "tar extract returned $LASTEXITCODE - extract the .tar.bz2 manually into $mdst" }
    }

    # --- ffmpeg onto PATH (optional) ---
    $ffZip = Join-Path $BundleDir 'tools\ffmpeg-win64-gpl.zip'
    if (Test-Path $ffZip) {
        $ffDir = Join-Path $BundleDir 'tools\ffmpeg'
        if (-not (Test-Path $ffDir)) { Expand-Archive $ffZip $ffDir -Force }
        $ffBin = (Get-ChildItem $ffDir -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1).DirectoryName
        if ($ffBin) { $env:PATH = "$ffBin;$env:PATH"; Say "ffmpeg available this session: $ffBin" }
    }

    # --- build whisper-gui.exe ---
    Say "configuring (cmake) - needs Visual Studio 2022 C++ Build Tools"
    Push-Location $repo
    try {
        & cmake -B build -DWHISPER_GUI=ON -DCMAKE_BUILD_TYPE=Release
        if ($LASTEXITCODE -ne 0) {
            Die "cmake configure failed. The usual cause is missing MSVC: install Visual Studio 2022 Build Tools with 'Desktop development with C++'."
        }
        Say "building (this compiles SDL2 from source the first time - a few minutes)"
        & cmake --build build --config Release --target whisper-gui --parallel
        if ($LASTEXITCODE -ne 0) { Die "build failed - see errors above." }
    } finally { Pop-Location }

    $exe = Get-ChildItem $repo -Recurse -Filter 'whisper-gui.exe' | Select-Object -First 1
    Say "INSTALL COMPLETE."
    Write-Host @"

Built: $($exe.FullName)

Run it FROM THE REPO ROOT so the model + helper paths resolve:
    cd "$repo"
    .\build\bin\Release\whisper-gui.exe

In the app: Browse a .wav -> Diarize = Accurate (sherpa-onnx) -> Speakers = 3 -> Transcribe.
(Model: models\ggml-base.en.bin   Diarization models: models\)
"@ -ForegroundColor Green
}

# ================================================================ main =========
switch ($Mode) {
    'Stage'   { Invoke-Stage }
    'Install' { Invoke-Install }
}
