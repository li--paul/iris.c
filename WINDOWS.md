# Windows/MSVC 移植指南

## 环境

- **OS:** Windows 11 (10.0.26200)
- **Compiler:** MSVC 2022 (19.44) — Visual Studio 17 2022
- **SDK:** Windows SDK 10.0.26100.0
- **BLAS:** Intel oneMKL 2025.0 (`C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\`)
- **模型:** FLUX.2-klein-4B 蒸馏版（4步）

## CMake 构建

提供两种后端，方便对比调试：

```bash
# BLAS（oneMKL 加速，推荐）
cmake -B build -DUSE_BLAS=ON -DMKL_INTERFACE_FULL=intel_lp64
cmake --build build --config Release

# GENERIC（纯 C 无加速，用于对比验证）
cmake -B build_generic -DUSE_BLAS=OFF
cmake --build build_generic --config Release
```

### 关键选项说明

| 选项 | 说明 |
|------|------|
| `-DUSE_BLAS=ON` | 启用 BLAS 加速（纯 C 推理太慢，不实用） |
| `-DMKL_INTERFACE_FULL=intel_lp64` | **必须**。MKL 默认用 ILP64（64-bit 整数），但 iris 传的是 32-bit `int`，不匹配会导致崩溃。LP64 使用 32-bit 整数。 |
| `--config Release` | Debug 模式下 `/RTC1` 与 `/O2` 冲突，CMake 已用生成器表达式处理 |

## 编译问题及修复

### MSVC 专有

| 问题 | 文件 | 修复 |
|------|------|------|
| VLA（变长数组） | `iris_transformer_flux.c:1647-1649,1794-1796` `iris_transformer_zimage.c:804` | 改为 `malloc`/`free` |
| `sysconf(_SC_NPROCESSORS_ONLN)` | `iris_transformer_flux.c` | 改为 `GetSystemInfo` |
| POSIX `isatty` | `linenoise.c` | `iris_platform.h` 中 `#define isatty _isatty` |
| POSIX `strcasecmp` | `iris_platform.h` | `#define strcasecmp _stricmp` |
| POSIX `unlink` | `iris_platform.h` | `#define unlink _unlink` |
| POSIX `strdup` | `iris_qwen3_tokenizer.c` | `#define strdup _strdup`（在 `iris_platform.h` 中） |
| `STDIN_FILENO` 等常量 | `iris_platform.h` | `#define STDIN_FILENO 0` |
| `EAGAIN`/`EWOULDBLOCK` | `iris_platform.h` | 定义对应 Win32 值 |
| `munmap` | `iris_safetensors.c` | `iris_munmap()` 封装：POSIX 用 `munmap`，Win32 用 `UnmapViewOfFile` |
| `#include <cblas.h>` | 4 个源文件 | 条件包含：`#ifdef USE_MKL → <mkl_cblas.h>` |
| `mode_t`/`umask` | `linenoise.c` | `#ifndef _WIN32` 防护 |

### CMake 配置

| 问题 | 修复 |
|------|------|
| Debug 下 `/O2` 与 `/RTC1` 冲突 | 生成器表达式：`$<$<NOT:$<CONFIG:Debug>>:/O2>` |
| oneMKL 头文件路径 | `target_include_directories(iris PRIVATE ${MKL_INCLUDE})` |
| Windows 特有源文件 | `iris_getopt.c` 仅 Win32 编译 |
| `_CRT_SECURE_NO_WARNINGS` | Win32 添加编译定义 |

## 运行时问题及修复

### 1. 崩溃：`STATUS_FLOAT_MULTIPLE_FAULTS (0xC06D007E)`

**现象:** 进程在 `Encoding text...` 后立即崩溃，事件查看器中记录 APPCRASH，异常码 0xC06D007E。

**根因:** MKL 接口不匹配。`FindMKL.cmake` 默认选择 **ILP64** 接口（`MKL_INT` = 64-bit `long long`），但 iris 所有 `cblas_sgemm` 调用传的是 32-bit `int`。MKL 会把两个相邻 32-bit int 合并读成一个 64-bit int，导致维度变成天文数字，越界访问触发 FPU 异常。

**修复:** CMake 配置时指定 LP64 接口：

```bash
cmake -B build -DUSE_BLAS=ON -DMKL_INTERFACE_FULL=intel_lp64
```

### 2. 极度缓慢（等效纯 C 速度）

**现象:** `Encoding text...` 后跑 10 分钟以上无结果，CPU 占用低（单核）。`MKL_VERBOSE=1` 无任何输出。

**根因:** MKL 线程层 `mkl_intel_thread.2.dll` 依赖 Intel OpenMP 运行时 `libiomp5md.dll`，该 DLL 位于 `compiler/2025.0/bin/`，不在默认 PATH。找不到时 MKL 退化为单线程顺序执行，等效纯 C 的三重循环。

**修复:** 运行时将 Intel compiler bin 目录加入 PATH：

```powershell
$env:PATH = "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin;C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin;" + $env:PATH
```

## 生成参数说明

蒸馏模型（distilled）原生训练步数为 **4步**，减步会导致去噪不充分、输出接近噪声。

| 参数 | 相邻像素差异 (h_adj_diff) | 效果 |
|------|--------------------------|------|
| `64x64 --steps 2` | 19.3 | 接近噪声，不实用 |
| `64x64 --steps 4` | 17.0 | 略有改善，仍有限 |
| `256x256 --steps 4` | 8.0 | **空间相关性强，质量正常** |

**建议用法：** 不指定 `--steps`（自动使用默认4步），分辨率至少 256×256。

## 最终运行结果

### 256×256 4步（默认参数）

```
BLAS: enabled
Seed: 42
Model: flux-klein-4b
Prompt: a cat
Output: test_cat.png
Size: 256x256
Steps: 4
Loading VAE... done (0.1s)
Model: FLUX.2-klein-4B v1.0 (distilled, 4 steps, guidance 1.0)
Loading Qwen3 encoder... done (0.4s)
Encoding text... done (9.8s)
Loading FLUX.2 transformer... done (0.3s)
Denoising: Step 1/4 dddddssssF  Step 2/4 ... Step 3/4 ... Step 4/4 ...
Denoising timing: 53.9s total (4 steps)
  Double blocks: 14.4s (26.9%)
  Single blocks: 39.2s (73.1%)
Decoding image... done (4.9s)
Total: 70.2 seconds
```

### 64×64 1步（快速验证）

```
BLAS: enabled
Loading VAE... done (0.1s)
Loading Qwen3 encoder... done (6.9s)
Encoding text... done (10.8s)
Loading FLUX.2 transformer... done (14.1s)
Denoising: Step 1/1 (11.8s)
Decoding image... done (0.4s)
Total: 44.9 seconds
```

## 测试脚本

```powershell
# windows_test.ps1
# 构建 + 运行测试，同时编译 BLAS 和 GENERIC 两个版本

$MklBinDir = "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin"
$CompilerBinDir = "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin"

# 构建
cmake -B build -DUSE_BLAS=ON -DMKL_INTERFACE_FULL=intel_lp64
cmake --build build --config Release --target iris
cmake -B build_generic -DUSE_BLAS=OFF
cmake --build build_generic --config Release --target iris

# PATH
$env:PATH = "${MklBinDir};${CompilerBinDir};" + $env:PATH

# 快速验证（64x64 1步）
.\build\Release\iris.exe -d flux-klein-4b -p "cat" --seed 42 --steps 1 -o test_quick.png -W 64 -H 64 --no-mmap

# 完整生成（256x256 默认4步）
.\build\Release\iris.exe -d flux-klein-4b -p "a cat" --seed 42 -W 256 -H 256 -o test_cat.png
```

## 已知限制

- **mmap / no-mmap 输出一致**：已验证两种模式在同一参数下产生相同哈希，mmap 加载正确。
- **GENERIC_BUILD（纯 C）**：编译通过，可用于对比验证输出哈希，但推理时间极长（Qwen3 编码 >10 分钟）。
- **BLAS 单/多线程一致性**：单线程和多线程 MKL 产生不同输出（浮点运算顺序差异），多线程版本在多次运行间保持确定。
- **Debug 构建**：BLAS 也可用，但 `/RTC1` 可能降低性能。
- **MKL DLL 依赖**：`mkl_intel_thread.2.dll` 和 `libiomp5md.dll` 必须在运行时能找到。可考虑改用 `-DMKL_THREADING=sequential` 消除 OpenMP 依赖，或 `-DMKL_LINK=static` 静态链接。
