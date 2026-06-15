# Script to compare GENERIC_BUILD vs BLAS_BUILD outputs
# Run from project root

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$MklBin = "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin"
$CompBin = "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin"

function Run-Iris {
    param($BuildDir, $Label, $UseBlas)
    $exe = "$BuildDir\Release\iris.exe"
    if (-not (Test-Path $exe)) {
        Write-Host "SKIP $Label: $exe not found" -ForegroundColor Yellow
        return $null
    }
    $out = "$BuildDir\Release\test_compare.png"
    $stderr = "$BuildDir\Release\test_compare_err.txt"
    $stdout = "$BuildDir\Release\test_compare_out.txt"

    Write-Host "Running $Label..." -ForegroundColor Cyan
    if ($UseBlas) {
        $env:PATH = "${MklBin};${CompBin};" + $env:PATH
    }
    
    $p = Start-Process -FilePath $exe -ArgumentList "-d flux-klein-4b -p `"a cat`" --seed 42 --steps 2 -W 64 -H 64 -o $out" -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $p.WaitForExit(600000)  # 10 min timeout
    
    if ($p.HasExited) {
        Write-Host "  Exit code: $($p.ExitCode)"
    } else {
        Write-Host "  TIMEOUT after 10 min" -ForegroundColor Yellow
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        return $null
    }
    
    if (Test-Path $out) {
        $hash = (Get-FileHash $out).Hash
        $size = (Get-Item $out).Length
        Write-Host "  Image: ${size} bytes, hash=${hash.Substring(0,16)}..." -ForegroundColor Green
        return @{Hash=$hash; Size=$size; Path=$out}
    } else {
        Write-Host "  NO image output" -ForegroundColor Red
        Get-Content $stderr
        return $null
    }
}

Write-Host "=== Comparing GENERIC_BUILD vs BLAS_BUILD ===" -ForegroundColor Magenta
Write-Host ""

$generic = Run-Iris -BuildDir "build_generic" -Label "GENERIC (pure C)" -UseBlas $false
$blas = Run-Iris -BuildDir "build" -Label "BLAS (MKL)" -UseBlas $true

Write-Host ""
if ($generic -and $blas) {
    if ($generic.Hash -eq $blas.Hash) {
        Write-Host "IDENTICAL OUTPUTS" -ForegroundColor Green
    } else {
        Write-Host "DIFFERENT OUTPUTS" -ForegroundColor Red
    }
} else {
    Write-Host "Could not compare (one or both missing)" -ForegroundColor Yellow
}
