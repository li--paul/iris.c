cmake_minimum_required(VERSION 3.12)
project(iris C)

# ==============================================================================
# Options
# ==============================================================================
option(USE_BLAS  "Enable BLAS acceleration (OpenBLAS / Accelerate / MKL)" OFF)
option(USE_METAL "Enable Metal GPU on Apple Silicon (macOS only)"          OFF)
option(USE_CUDA  "Enable CUDA GPU acceleration (NVIDIA, Windows/Linux)"    OFF)

# ==============================================================================
# C standard
# ==============================================================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ==============================================================================
# Compiler flags
# No -march=native, -ffast-math, -lm: not portable.
# ==============================================================================
if(MSVC)
  add_compile_options(/W3 /fp:precise)
  # /O2 conflicts with Debug /RTC1; only apply to non-Debug configs
  add_compile_options($<$<NOT:$<CONFIG:Debug>>:/O2>)
else()
  add_compile_options(-Wall -Wextra -O3)
endif()

# ==============================================================================
# Sources
# ==============================================================================
set(IRIS_SOURCES
  iris.c
  iris_kernels.c
  iris_tokenizer.c
  iris_vae.c
  iris_transformer_flux.c
  iris_transformer_zimage.c
  iris_sample.c
  iris_image.c
  jpeg.c
  iris_safetensors.c
  iris_qwen3.c
  iris_qwen3_tokenizer.c
  terminals.c
  iris_cli.c
  linenoise.c
  embcache.c
  main.c
)

if(WIN32)
  list(APPEND IRIS_SOURCES iris_getopt.c)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
endif()

# ==============================================================================
# Target definition
# ==============================================================================
add_executable(iris ${IRIS_SOURCES})
target_include_directories(iris PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT WIN32)
  target_link_libraries(iris PRIVATE m)
endif()

if(APPLE)
  target_compile_definitions(iris PRIVATE ACCELERATE_NEW_LAPACK)
endif()

# ==============================================================================
# BLAS backend
# ==============================================================================
if(USE_BLAS)
  if(APPLE)
    find_library(ACCELERATE Accelerate)
    if(ACCELERATE)
      target_link_libraries(iris PRIVATE ${ACCELERATE})
      target_compile_definitions(iris PRIVATE USE_BLAS)
      message(STATUS "BLAS: macOS Accelerate")
    else()
      message(FATAL_ERROR "Accelerate framework not found")
    endif()

  elseif(WIN32)
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
      target_link_libraries(iris PRIVATE OpenBLAS::OpenBLAS)
      target_compile_definitions(iris PRIVATE USE_BLAS USE_OPENBLAS)
      message(STATUS "BLAS: OpenBLAS at ${OpenBLAS_DIR}")
    else()
      find_package(MKL QUIET)
      if(MKL_FOUND)
        target_link_libraries(iris PRIVATE MKL::MKL)
        target_compile_definitions(iris PRIVATE USE_BLAS USE_MKL)
        if(MKL_INCLUDE)
          target_include_directories(iris PRIVATE ${MKL_INCLUDE})
        endif()
        message(STATUS "BLAS: oneMKL")
      else()
        message(WARNING "USE_BLAS=ON but neither OpenBLAS nor MKL found. "
                        "Falling back to GENERIC_BUILD.")
        target_compile_definitions(iris PRIVATE GENERIC_BUILD)
      endif()
    endif()

  else()  # Linux
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
      target_link_libraries(iris PRIVATE OpenBLAS::OpenBLAS)
      target_compile_definitions(iris PRIVATE USE_BLAS USE_OPENBLAS)
      message(STATUS "BLAS: OpenBLAS")
    else()
      message(FATAL_ERROR "USE_BLAS=ON but OpenBLAS not found. "
                          "Install libopenblas-dev or set -DUSE_BLAS=OFF.")
    endif()
  endif()
else()
  target_compile_definitions(iris PRIVATE GENERIC_BUILD)
  message(STATUS "BLAS: disabled (GENERIC_BUILD pure C fallback)")
endif()

# ==============================================================================
# Metal backend (Apple Silicon only)
# ==============================================================================
if(USE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "USE_METAL requires macOS (Apple Silicon)")
  endif()

  enable_language(OBJC)
  list(APPEND IRIS_SOURCES iris_metal.m)
  target_sources(iris PRIVATE iris_metal.m)
  set_source_files_properties(iris_metal.m PROPERTIES LANGUAGE OBJC)

  # Bake metal shader source into a C header
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 needed to embed Metal shaders")
  endif()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/iris_shaders_source.h
    COMMAND ${Python3_EXECUTABLE} -c "
with open('${CMAKE_CURRENT_SOURCE_DIR}/iris_shaders.metal','rb') as f:
    data = f.read()
with open('${CMAKE_CURRENT_BINARY_DIR}/iris_shaders_source.h','w') as out:
    out.write('unsigned char iris_shaders_metal[] = {\\n')
    for i in range(0, len(data), 12):
        hexvals = ', '.join(f'0x{b:02x}' for b in data[i:i+12])
        out.write('  ' + hexvals + ',\\n')
    out.write('};\\n')
    out.write(f'unsigned int iris_shaders_metal_len = {len(data)};\\n')
"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/iris_shaders.metal
    COMMENT "Embedding Metal shaders"
  )

  target_include_directories(iris PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  find_library(METAL_FW Metal)
  find_library(MPS_FW MetalPerformanceShaders)
  find_library(MPSG_FW MetalPerformanceShadersGraph)
  find_library(FOUNDATION_FW Foundation)

  target_link_libraries(iris PRIVATE
    ${METAL_FW} ${MPS_FW} ${MPSG_FW} ${FOUNDATION_FW})

  target_compile_definitions(iris PRIVATE USE_BLAS USE_METAL ACCELERATE_NEW_LAPACK)
  message(STATUS "Metal GPU: enabled")
else()
  message(STATUS "Metal GPU: disabled")
endif()

# ==============================================================================
# CUDA backend (NVIDIA GPU, Windows/Linux)
# ==============================================================================
if(USE_CUDA)
  # Find CUDA toolkit (for cuda_runtime.h, cublas_v2.h, and nvcc)
  find_package(CUDAToolkit REQUIRED)
  find_program(IRIS_NVCC nvcc
    PATHS "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH)
  if(NOT IRIS_NVCC)
    message(FATAL_ERROR "nvcc not found in CUDA toolkit at ${CUDAToolkit_BIN_DIR}")
  endif()

  set(IRIS_CUDA_ARCHS "75;80;89")
  set(IRIS_CUDA_ARCH_FLAGS)
  foreach(arch ${IRIS_CUDA_ARCHS})
    list(APPEND IRIS_CUDA_ARCH_FLAGS
      --generate-code=arch=compute_${arch},code=[compute_${arch},sm_${arch}])
  endforeach()

  # Collect include dirs and defines from the iris target for NVCC
  set(IRIS_CUDA_INCLUDES
    -I"${CMAKE_CURRENT_SOURCE_DIR}"
    -I"${CUDAToolkit_INCLUDE_DIRS}"
  )
  if(DEFINED MKL_INCLUDE)
    list(APPEND IRIS_CUDA_INCLUDES -I"${MKL_INCLUDE}")
  endif()

  set(IRIS_CUDA_DEFINES
    -DUSE_CUDA -DUSE_BLAS -DUSE_MKL
    -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES -DWIN32
  )

  # NVCC host compiler flags — lists expand to multiple args in add_custom_command
  set(IRIS_CUDA_HOST_FLAGS -Xcompiler=/MD -Xcompiler=/O2)

  # Build the object file with nvcc directly, bypassing CMake CUDA integration
  # (avoid conflicts between CMake 4.x and CUDA 13.0 for -Xcompiler / -ccbin flags)
  set(IRIS_CUDA_OBJ "${CMAKE_CURRENT_BINARY_DIR}/iris_cuda.obj")
  add_custom_command(
    OUTPUT ${IRIS_CUDA_OBJ}
    COMMAND ${IRIS_NVCC}
      -ccbin "${CMAKE_C_COMPILER}"
      ${IRIS_CUDA_INCLUDES}
      --machine 64 --compile -cudart static
      ${IRIS_CUDA_ARCH_FLAGS}
      ${IRIS_CUDA_DEFINES}
      ${IRIS_CUDA_HOST_FLAGS}
      -o ${IRIS_CUDA_OBJ}
      "${CMAKE_CURRENT_SOURCE_DIR}/iris_cuda.cu"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/iris_cuda.cu"
    COMMENT "Compiling CUDA kernel (iris_cuda.cu → iris_cuda.obj)"
  )

  # Import the object file as a generated source
  target_sources(iris PRIVATE ${IRIS_CUDA_OBJ})
  set_source_files_properties(${IRIS_CUDA_OBJ} PROPERTIES GENERATED TRUE)

  target_include_directories(iris PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(iris PRIVATE CUDA::cudart CUDA::cublas)
  target_compile_definitions(iris PRIVATE USE_CUDA)

  message(STATUS "CUDA GPU: enabled (${CUDAToolkit_VERSION}, arch=${IRIS_CUDA_ARCHS})")
else()
  message(STATUS "CUDA GPU: disabled")
endif()

# ==============================================================================
# Static library
# ==============================================================================
add_library(libiris STATIC ${IRIS_SOURCES})
target_include_directories(libiris PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT WIN32)
  target_link_libraries(libiris PRIVATE m)
endif()
if(APPLE)
  target_compile_definitions(libiris PRIVATE ACCELERATE_NEW_LAPACK)
endif()
if(USE_CUDA)
  target_include_directories(libiris PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(libiris PRIVATE CUDA::cudart CUDA::cublas)
  set_property(TARGET libiris PROPERTY CUDA_ARCHITECTURES "${IRIS_CUDA_ARCHS}")
  target_compile_definitions(libiris PRIVATE USE_CUDA)
endif()

# ==============================================================================
# Test targets (requires model weights at runtime)
# ==============================================================================
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND)
  add_custom_target(test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py
            --flux-binary $<TARGET_FILE:iris>
    DEPENDS iris
    USES_TERMINAL
  )
  add_custom_target(test-quick
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py
            --flux-binary $<TARGET_FILE:iris> --quick
    DEPENDS iris
    USES_TERMINAL
  )
endif()

# ==============================================================================
# Install
# ==============================================================================
include(GNUInstallDirs)
install(TARGETS iris libiris
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES iris.h iris_kernels.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
