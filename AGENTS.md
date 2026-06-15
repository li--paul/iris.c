# Iris — C Inference Pipeline for Flux.2 Klein & Z-Image-Turbo

## Build & Test

```bash
make mps        # macOS Apple Silicon (fastest)
make blas       # macOS Intel / Linux + OpenBLAS
make cuda       # NVIDIA GPU + cuBLAS (fastest on CUDA hardware)
make generic    # Pure C, zero deps (slow)
make debug      # ASAN build (-g -O0 -fsanitize=address)
make info       # Show available backends for this platform

make test       # Python test runner (compare vs reference images)
make test-quick # Only the 64x64 2-step test
```

Quick manual sanity (no test vectors needed):
```bash
./iris -d flux-klein-4b -p "a cat" --seed 42 --steps 2 -W 64 -H 64 -o /tmp/t.png
```

Model architecture is autodetected from config JSONs in the model dir. Do not hardcode dimensions.

## Model Download

Must have user approval first. 9B models are gated (need HuggingFace token). Z-Image needs `huggingface_hub` Python package.

```bash
./download_model.sh 4b         # ~16GB, curl-based
python download_model.py 4b    # alternative via huggingface_hub
```

## Key Architecture Facts

Used by both Flux and Z-Image paths:
- **Naming**: `iris_` prefix (public API), `_flux`/`_zimage` postfix (model internals), `qwen3_*`/`safetensors_*` as component namespaces
- **Weights**: BF16 safetensors, memory-mapped by default. mmap is fastest on MPS; use `--no-mmap` for faster CPU inference if RAM permits (32GB+)
- **Text encoder**: Qwen3, auto-released after encoding to save ~8-16GB
- **Sampling**: Flux uses shifted sigmoid schedule; Z-Image uses FlowMatch Euler. Override with `--linear`, `--power`, `--sigmoid`, `--flowmatch`

## Critical Implementation Details

These differ from defaults you might assume:

- **Flux attention concat order**: `[TEXT, IMAGE]` (not `[IMAGE, TEXT]`)
- **AdaLN formula**: `out = (1 + scale) * norm(x) + shift` (not `scale * norm(x) + shift`)
- **Final layer modulation split**: `(scale, shift)` not `(shift, scale)`
- **RoPE pair rotation**: `out0 = cos*x0 - sin*x1, out1 = cos*x1 + sin*x0`
- **Flux text embedding extraction**: layers 8, 17, 26 (0-indexed) concatenated → [seq, text_dim]
- **Z-Image text embedding**: Qwen3 `hidden_states[-2]` only (layer 34/36) → [seq, 2560]
- **Z-Image sequence order**: `[IMAGE | CAPTION]` in transformer path

## WSL CUDA Note

CUDA on WSL2 may crash during `cudaMalloc` in `iris_cuda_attention` due to a JIT compiler mismatch between the WSL NVIDIA compute package and the Windows host driver. If affected, the attention falls back to BLAS (CPU). The linear layers still use CUDA. Set `LD_LIBRARY_PATH` to your CUDA toolkit's `lib/` directory at runtime.

## Historical Bugs (Not to Repeat)

1. **MPS SGEMM B-cache**: VAE attention K/V are dynamic temporaries. Use `iris_metal_sgemm` (no B-pointer cache), NOT `iris_metal_sgemm_cached` (static weights only). Wrong use caused hue/border artifacts in VAE decode.
2. **GPU RoPE indexing**: Must use consecutive pairs `(d, d+1)`, not axis-half indexing.
3. **RMSNorm temp weights**: Temporary fused norm weights must not be pointer-cached as immutable weights.
4. **CPU/GPU position-id mismatch**: Padded vs unpadded caption length changes RoPE indexing. Must match exactly.
5. **Z-Image step preview**: Must account for `patch_size=2` decode, else preview steps mislead.
6. **GPU timestep caching**: Step-dependent shift/scale/gate must NOT be cached as static weights.
7. **CLI base-model CFG**: Base models in interactive mode must use `iris_generate()` (CFG-aware), not the distilled-only embedding path.

## Python Debug Parity

```bash
# 1. Set up venv (flux_env/) with torch, diffusers, transformers
# 2. Run Python to dump inputs:
python debug/debug_img2img_compare.py
# 3. Run C with same inputs:
./iris -d flux-klein-4b --debug-py -W 256 -H 256 --steps 4 -o /tmp/c_debug.png
```

Debug scripts live in `debug/`. Python ref code lives in `flux2/` (git clone). Never commit `flux_env/` or `flux2/`.

## Verification

Always run before/after changes:
```bash
make test        # All available tests
make test-quick  # Fast smoke test
```

Benchmark iterations: use `--seed 42 -v` for timing, e.g. `/tmp/bench.png`.
