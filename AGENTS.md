# Iris — C image synthesis inference engine

Supports **Flux.2 Klein** (4B/9B distilled & base) and **Z-Image-Turbo** (6B). Model type/architecture is autodetected from config JSON at load time — do not hardcode dimensions.

## Naming Convention

- `iris_` prefix: shared/generic identifiers
- `_flux`/`_zimage` suffix: model-family-specific internals
- No suffix on public API — they route internally by model type

## Build & Test

**CMake** (cross-platform, replaces old Makefile):

```bash
cmake -B build -DUSE_BLAS=OFF            # Generic (pure C, works everywhere)
cmake -B build -DUSE_BLAS=ON             # BLAS (OpenBLAS / Accelerate / MKL)
cmake -B build -DUSE_METAL=ON            # macOS Apple Silicon only
cmake --build build                      # Compile
```

Platforms:
- **Linux/macOS**: CMake with any backend. Metal requires Apple Silicon.
- **Windows**: CMake (MSVC/MinGW). BLAS via OpenBLAS or oneMKL (`-DUSE_BLAS=ON`).
  Falls back to GENERIC_BUILD if no BLAS library found.
  Interactive CLI mode is stubbed (simple stdin input; no raw terminal).
- **macOS Legacy Makefile** (still available): `make mps|blas|generic|test|test-quick`

Test runner: `python3 run_test.py [--quick] [--full] [--flux-binary ./iris] [--model-dir flux-klein-4b]`

**CMake test targets**: `cmake --build build --target test` (or `test-quick`)

## Verification Order

```bash
make test-quick  # 64x64 2-step sanity (~1 min with model)
# Manual flux check:
./iris -d flux-klein-4b -p "A fluffy orange cat" --seed 42 --steps 2 -o /tmp/test.png -W 64 -H 64
# Manual Z-Image check:
./iris -d zimage-turbo -p "a fish" --seed 43 --steps 8 -o /tmp/zimage_test.png
```

## Critical Implementation Details

- **Flux attention concat order**: `[TEXT, IMAGE]` — NOT `[IMAGE, TEXT]`
- **AdaLN**: `out = (1 + scale) * norm(x) + shift`
- **Final layer modulation**: `(scale, shift)` — NOT `(shift, scale)`
- **RoPE pair rotation**: `out0 = cos*x0 - sin*x1`, `out1 = cos*x1 + sin*x0`
- **Z-Image transformer sequence**: `[IMAGE | CAPTION]` (opposite of Flux)
- **Qwen3 extraction** — Flux: concat layers 8, 17, 26 (0-indexed). Z-Image: `hidden_states[-2]` (layer 34)
- **Z-Image step previews**: must decode patchified latents (`patch_size=2`), otherwise preview differs from final

## Weight Loading

- **mmap is default and fastest on MPS** (zero-copy bf16). Use `--no-mmap` for BLAS with abundant RAM.
- Model weights: safetensors, memory-mapped. Blocks loaded on-demand and freed after use.
- Text encoder (~8GB) auto-released after encoding; reloads on different prompt.

## Known Pitfalls (Historical Bugs)

1. **MPS SGEMM B-cache**: Never cache dynamic K/V pointers as static weights. Use `iris_metal_sgemm` (generic, no cache) for temporaries; `iris_metal_sgemm_cached` only for static weight matrices.
2. **RMSNorm temp weight caching**: Temporary fused norm weights must not be pointer-cached as immutable weights.
3. **CPU/GPU position-id alignment**: Must match exactly — using non-padded caption length in one path and padded in another breaks RoPE indexing.
4. **GPU step param caching**: Timestep-dependent shift/scale/gate must not be cached as static weights.
5. **GPU RoPE indexing**: Must use consecutive pairs `(d, d+1)`, not axis-half indexing.
6. **Z-Image scheduler**: Sigma schedule must match official diffusers FlowMatch Euler (static shift). Incorrect sigma wastes steps.

## Constraints

- No external deps beyond BLAS/OpenBLAS and macOS Metal framework. Pure C standard library is the floor.
- `flux_env/` and `flux2/` are Python reference venv/code — never commit them.
- Download scripts (`download_model.sh`, `download_model.py`) — run only after user approval.
- Reusable debug scripts go in `./debug/`; throwaway debugging in `/tmp/`.
- Prefer substantial speed wins; reject tiny gains that add complexity.
- If optimizing one backend, verify others are not regressed.
- CLI interactive mode: `$N` references previous images; `!help` for commands.

## Debug / Parity

- Python venv in `./flux_env/`, official Flux repo in `./flux2/`.
- `debug/debug_img2img_compare.py` saves noise/latent/embeddings to `/tmp/` for C comparison with `--debug-py`.
- JPEG tests in `./jpg_test/`.
- Threshold for test comparison: `mean_diff <= 20` (accounts for GPU non-determinism).
