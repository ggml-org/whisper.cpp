param(
    [Parameter(Position = 0)]
    [string] $Model,

    [Parameter(Position = 1)]
    [string] $ModelsPath,

    [switch] $List
)

$ErrorActionPreference = "Stop"

$Source = "https://huggingface.co"
$CollectionApi = "https://huggingface.co/api/collections/amd/ryzen-ai-whisper-npu-optimized-onnx-models"
$CollectionUrl = "https://huggingface.co/collections/amd/ryzen-ai-whisper-npu-optimized-onnx-models"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir -match "\\bin$") {
    $DefaultDownloadPath = (Get-Location).Path
} else {
    $DefaultDownloadPath = $ScriptDir
}

if (-not $ModelsPath) {
    $ModelsPath = $DefaultDownloadPath
}

function Get-HfHeaders {
    $headers = @{}
    if ($env:HF_TOKEN) {
        $headers["Authorization"] = "Bearer $env:HF_TOKEN"
    }
    return $headers
}

function Invoke-HfJson {
    param([string] $Uri)

    $headers = Get-HfHeaders
    if ($headers.Count -gt 0) {
        return Invoke-RestMethod -Uri $Uri -Headers $headers
    }

    return Invoke-RestMethod -Uri $Uri
}

function Normalize-ModelName {
    param([string] $Name)

    # HF currently publishes ggml-small-en-encoder-vitisai.rai, while the
    # matching ggml model is ggml-small.en.bin.
    if ($Name.EndsWith("-en")) {
        return $Name.Substring(0, $Name.Length - 3) + ".en"
    }

    return $Name
}

function Get-VitisAiModels {
    $collection = Invoke-HfJson -Uri $CollectionApi
    $seen = @{}
    $rows = New-Object System.Collections.Generic.List[object]

    foreach ($item in $collection.items) {
        if ($item.type -ne "model") {
            continue
        }

        $repo = $item.id
        if (-not $repo) {
            continue
        }

        $modelInfo = Invoke-HfJson -Uri "$Source/api/models/$repo"
        foreach ($sibling in $modelInfo.siblings) {
            $filename = [string] $sibling.rfilename
            $match = [regex]::Match($filename, "^ggml-(.+)-encoder-vitisai\.rai$")
            if (-not $match.Success) {
                continue
            }

            $rawName = $match.Groups[1].Value
            $modelName = Normalize-ModelName -Name $rawName
            if ($seen.ContainsKey($modelName)) {
                continue
            }

            $seen[$modelName] = $true
            $destination = "ggml-$modelName-encoder-vitisai.rai"
            $url = "$Source/$repo/resolve/main/$([uri]::EscapeDataString($filename))"

            $rows.Add([pscustomobject]@{
                Model = $modelName
                RawName = $rawName
                Repo = $repo
                SourceFile = $filename
                DestinationFile = $destination
                DownloadUrl = $url
            })
        }
    }

    $order = @{
        "tiny" = 10
        "tiny.en" = 11
        "base" = 20
        "base.en" = 21
        "small" = 30
        "small.en" = 31
        "medium" = 40
        "medium.en" = 41
        "large-v1" = 50
        "large-v2" = 60
        "large-v3" = 70
        "large-v3-turbo" = 80
    }

    return $rows | Sort-Object `
        @{ Expression = { if ($order.ContainsKey($_.Model)) { $order[$_.Model] } else { 1000 } } }, `
        @{ Expression = { $_.Model } }
}

function Show-Models {
    $models = Get-VitisAiModels

    Write-Host ""
    Write-Host "Available VitisAI encoder caches from ${CollectionUrl}:"
    foreach ($entry in $models) {
        if ($entry.Model -eq $entry.RawName) {
            Write-Host ("  {0,-18} {1}" -f $entry.Model, $entry.Repo)
        } else {
            Write-Host ("  {0,-18} {1} (source name: {2})" -f $entry.Model, $entry.Repo, $entry.RawName)
        }
    }
    Write-Host ""
}

function Show-Usage {
    Write-Host "Usage: download-vitisai-model.cmd --list"
    Write-Host "       download-vitisai-model.cmd <model> [models_path]"
    Write-Host ""
    Write-Host "Downloads ggml-<model>-encoder-vitisai.rai next to ggml-<model>.bin."
    Write-Host "Use the same model name as download-ggml-model.cmd."
    Write-Host ""
}

if ($List -or $Model -eq "--list" -or $Model -eq "-l" -or $Model -eq "list") {
    Show-Models
    exit 0
}

if (-not $Model) {
    Show-Usage
    Show-Models
    exit 1
}

$models = Get-VitisAiModels
$entry = $models | Where-Object { $_.Model -eq $Model -or $_.RawName -eq $Model } | Select-Object -First 1
if (-not $entry) {
    Write-Host "Invalid model: $Model"
    foreach ($available in $models) {
        Write-Host "  $($available.Model)"
    }
    exit 1
}

New-Item -ItemType Directory -Force -Path $ModelsPath | Out-Null
$destinationPath = Join-Path $ModelsPath $entry.DestinationFile

Write-Host "Downloading VitisAI encoder cache $($entry.Model) from '$($entry.Repo)' ..."
if (Test-Path $destinationPath) {
    Write-Host "VitisAI encoder cache $($entry.DestinationFile) already exists. Skipping download."
    exit 0
}

$headers = Get-HfHeaders
$downloaded = $false
for ($attempt = 1; $attempt -le 5; ++$attempt) {
    try {
        if ($headers.Count -gt 0) {
            Invoke-WebRequest -Uri $entry.DownloadUrl -Headers $headers -OutFile $destinationPath
        } else {
            Invoke-WebRequest -Uri $entry.DownloadUrl -OutFile $destinationPath
        }
        $downloaded = $true
        break
    } catch {
        if ($attempt -eq 5) {
            if (Test-Path $destinationPath) {
                Remove-Item -Force $destinationPath
            }
            Write-Host "Failed to download VitisAI encoder cache $($entry.Model) from $($entry.DownloadUrl)"
            throw
        }
        Start-Sleep -Seconds 5
    }
}

if (-not $downloaded) {
    exit 1
}

$whisperCmd = "whisper-cli"
if (-not (Get-Command whisper-cli -ErrorAction SilentlyContinue)) {
    $rootPath = Split-Path -Parent $ScriptDir
    $whisperCmd = Join-Path $rootPath "build\bin\Release\whisper-cli.exe"
}

Write-Host "Done! VitisAI encoder cache '$($entry.Model)' saved in '$destinationPath'"
if ($entry.RawName -ne $entry.Model) {
    Write-Host "Source cache '$($entry.SourceFile)' was renamed to match ggml model name '$($entry.Model)'."
}
Write-Host "Use it with the matching ggml model:"
Write-Host ""
Write-Host "  $ScriptDir\download-ggml-model.cmd $($entry.Model) $ModelsPath"
Write-Host "  $whisperCmd -m $ModelsPath\ggml-$($entry.Model).bin -f samples\jfk.wav"
Write-Host ""
