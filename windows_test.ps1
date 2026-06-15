# Test script for Iris on Windows/MSVC with oneMKL
# Run from the iris.c project root directory

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# --- Config ---
$ModelDir = "flux-klein-4b"
$MklBinDir = "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin"
$CompilerBinDir = "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin"

# --- Locate MKL and compiler DLLs ---
if (-not (Test-Path $MklBinDir)) {
    Write-Error "MKL not found at $MklBinDir"
    exit 1
}
if (-not (Test-Path $CompilerBinDir)) {
    Write-Error "Intel compiler bin not found at $CompilerBinDir"
    exit 1
}

# --- Build (Release, BLAS with MKL LP64) ---
Write-Host "=== Configuring CMake (MKL LP64 interface) ===" -ForegroundColor Cyan
cmake -B build -DUSE_BLAS=ON -DMKL_INTERFACE_FULL=intel_lp64 | Out-Null
if (-not $?) { exit 1 }

Write-Host "=== Building Release ===" -ForegroundColor Cyan
cmake --build build --config Release --target iris | Out-Null
if (-not $?) { exit 1 }

# Also build GENERIC (pure C) for comparison
Write-Host "=== Building GENERIC (pure C, for comparison) ===" -ForegroundColor Cyan
cmake -B build_generic -DUSE_BLAS=OFF | Out-Null
cmake --build build_generic --config Release --target iris | Out-Null
if (-not $?) { exit 1 }

# --- Ensure both DLL dirs are in PATH ---
$env:PATH = "${MklBinDir};${CompilerBinDir};" + $env:PATH

# --- Test 1: --help ---
Write-Host "`n=== Test 1: --help ===" -ForegroundColor Cyan
& ".\build\Release\iris.exe" --help
if (-not $?) { exit 1 }

# --- Test 2: model detection ---
Write-Host "`n=== Test 2: Model Detection ===" -ForegroundColor Cyan
& ".\build\Release\iris.exe" -d $ModelDir -p "hello" 2>&1 | Select-String -Pattern "Model:"
if (-not $?) { exit 1 }

# --- Test 3: Quick sanity (64x64, 1 step) ---
Write-Host "`n=== Test 3: Quick Generation (64x64, 1 step) ===" -ForegroundColor Cyan
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& ".\build\Release\iris.exe" -d $ModelDir -p "cat" --seed 42 --steps 1 -o test_quick.png -W 64 -H 64 --no-mmap
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

# --- Test 4: Full generation (256x256, default 4 steps, seed=42) ---
Write-Host "`n=== Test 4: Full Generation (256x256, default 4 steps) ===" -ForegroundColor Cyan
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& ".\build\Release\iris.exe" -d $ModelDir -p "a cat" --seed 42 -W 256 -H 256 -o test_cat.png
$timer.Stop()
if ($LASTEXITCODE -ne 0) { exit 1 }

if (Test-Path "test_cat.png") {
    $size = (Get-Item "test_cat.png").Length
    Write-Host "PASS: ($($timer.Elapsed.TotalSeconds.ToString('0.0'))s, ${size}B)" -ForegroundColor Green
} else {
    Write-Error "FAIL: test_cat.png not created"
    exit 1
}

# --- Test 5: Compare BLAS output with GENERIC (no BLAS) ---
Write-Host "`n=== Test 5: BLAS vs GENERIC output (64x64, 1 step) ===" -ForegroundColor Cyan
& ".\build\Release\iris.exe" -d $ModelDir -p "cat" --seed 42 --steps 1 -o test_blas.png -W 64 -H 64 --no-mmap
if ($LASTEXITCODE -ne 0) { exit 1 }

# GENERIC uses pure C (slower) and no -o quoting issue in PowerShell
& ".\build_generic\Release\iris.exe" -d $ModelDir -p "cat" --seed 42 --steps 1 -o test_generic.png -W 64 -H 64 --no-mmap

if ((Get-Item "test_blas.png").Length -gt 100) {
    Write-Host "BLAS: $(Get-FileHash test_blas.png | Select-Object -ExpandProperty Hash)" -ForegroundColor Gray
    Write-Host "GENERIC: $(Get-FileHash test_generic.png | Select-Object -ExpandProperty Hash)" -ForegroundColor Gray
    Remove-Item test_blas.png, test_generic.png
}

Write-Host "`n=== All tests passed! ===" -ForegroundColor Cyan
Write-Host "Generated: test_cat.png (256x256, 4 steps)" -ForegroundColor Yellow
