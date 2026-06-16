# Test script for Iris on Windows/MSVC
# Run from the iris.c project root directory
# IMPORTANT: Avoid 2>&1 in PowerShell — it corrupts ANSI stderr output
# Use environment variable IRIS_BUILD_DIR to select CUDA build: $env:IRIS_BUILD_DIR = "build_cuda"

$ErrorActionPreference = "Stop"
$ModelDir = "flux-klein-4b"
$BuildDir = if ($env:IRIS_BUILD_DIR) { $env:IRIS_BUILD_DIR } else { "build" }

# PATH setup for MKL DLLs (required at runtime)
$MklBinDir = "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin"
$MklCompilerBin = "C:\Program Files (x86)\Intel\oneAPI\2025.0\bin"
$env:PATH = "${MklBinDir};${MklCompilerBin};" + $env:PATH

# CUDA DLL path (if CUDA build)
$CudaBinDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
if (Test-Path $CudaBinDir) {
    $env:PATH = "${CudaBinDir};" + $env:PATH
}

if (-not (Test-Path $MklBinDir)) { Write-Error "MKL not found at $MklBinDir"; exit 1 }

# --- Build (skip if build dir exists and user sets SKIP_BUILD) ---
if (-not $env:SKIP_BUILD) {
    Write-Host "=== Configuring CMake (${BuildDir}) ===" -ForegroundColor Cyan
    cmake -B $BuildDir -DUSE_BLAS=ON -DMKL_INTERFACE_FULL=intel_lp64 | Out-Null
    if (-not $?) { exit 1 }

    Write-Host "=== Building Release ===" -ForegroundColor Cyan
    cmake --build $BuildDir --config Release --target iris | Out-Null
    if (-not $?) { exit 1 }
} else {
    Write-Host "=== Skipping build (SKIP_BUILD set) ===" -ForegroundColor Yellow
}

$Iris = ".\${BuildDir}\Release\iris.exe"
if (-not (Test-Path $Iris)) { Write-Error "Not found: $Iris"; exit 1 }

# --- Test 1: --help ---
Write-Host "`n=== Test 1: --help ===" -ForegroundColor Cyan
& $Iris --help
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "PASS" -ForegroundColor Green

# --- Test 2: model detection ---
Write-Host "`n=== Test 2: Model Detection ===" -ForegroundColor Cyan
$null = & $Iris -d $ModelDir -p "hello" -q -o "$env:TEMP\iris_test_detection.png"
if ($LASTEXITCODE -ne 0) { Write-Error "FAIL: exit code $LASTEXITCODE"; exit 1 }
Remove-Item "$env:TEMP\iris_test_detection.png" -ErrorAction SilentlyContinue
Write-Host "PASS" -ForegroundColor Green

# --- Test 3: Quick 64x64 1-step ---
Write-Host "`n=== Test 3: Quick Generation (64x64, 1 step) ===" -ForegroundColor Cyan
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& $Iris -d $ModelDir -p "cat" --seed 42 --steps 1 -o test_quick.png -W 64 -H 64 --no-mmap
$timer.Stop()
if ($LASTEXITCODE -ne 0) { exit 1 }

if (Test-Path "test_quick.png") {
    $size = (Get-Item "test_quick.png").Length
    Write-Host "PASS: ($($timer.Elapsed.TotalSeconds.ToString('0.0'))s, ${size}B)" -ForegroundColor Green
    Remove-Item "test_quick.png"
} else {
    Write-Error "FAIL: test_quick.png not created"
    exit 1
}

# --- Test 4: Full 256x256 4-step ---
Write-Host "`n=== Test 4: Full Generation (256x256, 4 steps) ===" -ForegroundColor Cyan
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& $Iris -d $ModelDir -p "a cat" --seed 42 -W 256 -H 256 -o test_cat.png
$timer.Stop()
if ($LASTEXITCODE -ne 0) { exit 1 }

if (Test-Path "test_cat.png") {
    $size = (Get-Item "test_cat.png").Length
    Write-Host "PASS: ($($timer.Elapsed.TotalSeconds.ToString('0.0'))s, ${size}B)" -ForegroundColor Green
} else {
    Write-Error "FAIL: test_cat.png not created"
    exit 1
}

Write-Host "`n=== All tests passed! ===" -ForegroundColor Cyan
Write-Host "Generated: test_cat.png (256x256, 4 steps, ~197KB)" -ForegroundColor Yellow
