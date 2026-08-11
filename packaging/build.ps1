[CmdletBinding()]
param(
    [string]$PythonPath = "D:\python3.10.11\python.exe",
    [string]$FaceModelPath = "C:\Users\pc\.cache\eyetrax\mediapipe\face_landmarker.task"
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

function Assert-PathWithinRepository {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootPrefix = $repoRoot.TrimEnd("\") + "\"
    if (-not $fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside repository root: $fullPath"
    }
    return $fullPath
}

function ConvertFrom-CodePoints {
    param([Parameter(Mandatory = $true)][int[]]$CodePoints)

    return -join ($CodePoints | ForEach-Object { [char]$_ })
}

$releaseName = ConvertFrom-CodePoints @(0x7EAF, 0x773C, 0x52A8, 0x6253, 0x5B57, 0x7CFB, 0x7EDF)
$captureName = ConvertFrom-CodePoints @(0x773C, 0x52A8, 0x91C7, 0x96C6, 0x6821, 0x51C6)
$typingName = ConvertFrom-CodePoints @(0x7EAF, 0x773C, 0x52A8, 0x6253, 0x5B57, 0x5668)
$readmeName = ConvertFrom-CodePoints @(0x4F7F, 0x7528, 0x8BF4, 0x660E, 0x002E, 0x006D, 0x0064)

$python = [System.IO.Path]::GetFullPath($PythonPath)
$faceModel = [System.IO.Path]::GetFullPath($FaceModelPath)
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable was not found: $python"
}
if (-not (Test-Path -LiteralPath $faceModel -PathType Leaf)) {
    throw "FaceLandmarker model was not found: $faceModel"
}
if ((Get-Item -LiteralPath $faceModel).Length -le 0) {
    throw "FaceLandmarker model is empty: $faceModel"
}

$buildRoot = Assert-PathWithinRepository (Join-Path $repoRoot "build")
$distRoot = Assert-PathWithinRepository (Join-Path $repoRoot "dist")
$releaseRoot = Assert-PathWithinRepository (Join-Path $repoRoot (Join-Path "release" $releaseName))
$stagedModel = Assert-PathWithinRepository (Join-Path $repoRoot "resources\face_landmarker.task")
$captureSpec = Join-Path $repoRoot "packaging\pure_gaze_capture.spec"
$typingSpec = Join-Path $repoRoot "packaging\pure_gaze_typing.spec"
$smokeTest = Join-Path $repoRoot "packaging\smoke_test.py"

foreach ($target in @($buildRoot, $distRoot, $releaseRoot)) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

Copy-Item -LiteralPath $faceModel -Destination $stagedModel -Force
try {
    Push-Location $repoRoot
    try {
        & $python -m PyInstaller --noconfirm --clean `
            --workpath (Join-Path $buildRoot "capture") `
            --distpath (Join-Path $distRoot "capture") `
            $captureSpec
        if ($LASTEXITCODE -ne 0) {
            throw "Capture PyInstaller build failed with exit code $LASTEXITCODE"
        }

        & $python -m PyInstaller --noconfirm --clean `
            --workpath (Join-Path $buildRoot "typing") `
            --distpath (Join-Path $distRoot "typing") `
            $typingSpec
        if ($LASTEXITCODE -ne 0) {
            throw "Typing PyInstaller build failed with exit code $LASTEXITCODE"
        }

        New-Item -ItemType Directory -Path $releaseRoot | Out-Null
        Move-Item -LiteralPath (Join-Path $distRoot (Join-Path "capture" $captureName)) `
            -Destination (Join-Path $releaseRoot $captureName)
        Move-Item -LiteralPath (Join-Path $distRoot (Join-Path "typing" $typingName)) `
            -Destination (Join-Path $releaseRoot $typingName)
        Copy-Item -LiteralPath (Join-Path $repoRoot "README.md") `
            -Destination (Join-Path $releaseRoot $readmeName)

        & $python $smokeTest --release $releaseRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Release layout validation failed with exit code $LASTEXITCODE"
        }
        & $python $smokeTest --exe (Join-Path (Join-Path $releaseRoot $captureName) ($captureName + ".exe"))
        if ($LASTEXITCODE -ne 0) {
            throw "Capture executable self-test failed with exit code $LASTEXITCODE"
        }
        & $python $smokeTest --exe (Join-Path (Join-Path $releaseRoot $typingName) ($typingName + ".exe"))
        if ($LASTEXITCODE -ne 0) {
            throw "Typing executable self-test failed with exit code $LASTEXITCODE"
        }
        Write-Host "Build complete: $releaseRoot"
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -LiteralPath $stagedModel -Force -ErrorAction SilentlyContinue
}
