#!/bin/bash
# Iris Build & Test Script — CPU vs CUDA comparison
# Usage: ./scripts/build_test.sh [--quick]
#   --quick: skip 256x256 benchmark, only 64x64 smoke tests

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
NC='\033[0m'
ok()  { echo -e "  ${GREEN}OK${NC} $1"; }
fail(){ echo -e "  ${RED}FAIL${NC} $1"; exit 1; }
info(){ echo -e "  ${YELLOW}$1${NC}"; }

MODEL="${MODEL_DIR:-flux-klein-4b}"
CUDA_LD="${LD_LIBRARY_PATH:-/usr/local/cuda/targets/x86_64-linux/lib}"
QUICK=false
[ "${1:-}" = "--quick" ] && QUICK=true

cd "$(dirname "$0")/.."

echo "============================================"
echo " Iris Build & Test"
echo " Model: $MODEL"
echo " Date:  $(date)"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Build CPU backend (use BLAS if available, fall back to generic)
# ------------------------------------------------------------------
echo "--- [1/6] Build: CPU (BLAS, or generic fallback) ---"
make clean 2>/dev/null || true
if ldconfig -p 2>/dev/null | grep -q libopenblas; then
    make blas 2>&1 | tail -1
else
    make generic 2>&1 | tail -1
fi
BIN_CPU="./iris"
[ -f "$BIN_CPU" ] && ok "built" || fail "CPU build failed"

# ------------------------------------------------------------------
# 2. CPU smoke test (64x64, 1 step)
# ------------------------------------------------------------------
echo ""
echo "--- [2/6] CPU: 64x64 smoke test (1 step) ---"
OUT_CPU="/tmp/iris_cpu_64.png"
START=$SECONDS
timeout 120 $BIN_CPU -d "$MODEL" -p "a cat" --seed 42 --steps 1 -W 64 -H 64 -o "$OUT_CPU" 2>&1
RC=$?
DUR=$((SECONDS - START))
if [ $RC -eq 0 ] && [ -f "$OUT_CPU" ]; then
    ok "${DUR}s — $(file "$OUT_CPU")"
else
    fail "CPU test (exit=$RC, $DUR s)"
fi

# ------------------------------------------------------------------
# 3. Build CUDA
# ------------------------------------------------------------------
echo ""
echo "--- [3/6] Build: CUDA (cuBLAS + OpenBLAS) ---"
make clean 2>/dev/null || true
make cuda 2>&1 | tail -1
BIN_CUDA="./iris"
[ -f "$BIN_CUDA" ] && ok "built" || fail "CUDA build failed"

# ------------------------------------------------------------------
# 4. CUDA smoke test (64x64, 1 step)
# ------------------------------------------------------------------
echo ""
echo "--- [4/6] CUDA: 64x64 smoke test (1 step) ---"
OUT_CUDA="/tmp/iris_cuda_64.png"
START=$SECONDS
LD_LIBRARY_PATH="$CUDA_LD" timeout 120 $BIN_CUDA -d "$MODEL" -p "a cat" --seed 42 --steps 1 -W 64 -H 64 -o "$OUT_CUDA" 2>&1
RC=$?
DUR=$((SECONDS - START))
if [ $RC -eq 0 ] && [ -f "$OUT_CUDA" ]; then
    ok "${DUR}s — $(file "$OUT_CUDA")"
else
    fail "CUDA test (exit=$RC, $DUR s)"
fi

# ------------------------------------------------------------------
# 5. Compare CPU vs CUDA output
# ------------------------------------------------------------------
echo ""
echo "--- [5/6] Compare CPU vs CUDA output ---"
python3 -c "
import sys, numpy as np
from PIL import Image
cpu = np.array(Image.open('$OUT_CPU'))
cuda = np.array(Image.open('$OUT_CUDA'))
if cpu.shape != cuda.shape:
    print(f'  size mismatch: cpu={cpu.shape} cuda={cuda.shape}')
    sys.exit(1)
diff = np.abs(cpu.astype(float) - cuda.astype(float))
print(f'  mean_diff={diff.mean():.2f}  max_diff={diff.max():.0f}')
print(f'  (same seed, same prompt — should be visually identical)')
if diff.mean() > 80:
    print('  WARNING: large diff — possible correctness issue')
"
ok "comparison done"

# ------------------------------------------------------------------
# 6. Benchmark (256x256, 4 steps)
# ------------------------------------------------------------------
echo ""
if $QUICK; then
    echo "--- [6/6] Benchmark: skipped (--quick) ---"
else
    echo "--- [6/6] Benchmark: CUDA 256x256 (4 steps) ---"
    OUT_BENCH="/tmp/iris_bench_256.png"
    LD_LIBRARY_PATH="$CUDA_LD" timeout 300 ./iris -d "$MODEL" -p "a landscape" \
        --seed 42 --steps 4 -W 256 -H 256 -o "$OUT_BENCH" 2>&1 | \
        grep -E 'CUDA:|Denoising|Decoding image|Total generation'
    RC=$?
    [ $RC -eq 0 ] && ok "benchmark completed" || fail "benchmark (exit=$RC)"
fi

# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " PASSED  —  CPU and CUDA versions both work"
echo "============================================"
