cmake_minimum_required(VERSION 3.12)
project(iris C)

# ==============================================================================
# Options
# ==============================================================================
option(USE_BLAS  "Enable BLAS acceleration (OpenBLAS / Accelerate / MKL)" OFF)
option(USE_METAL "Enable Metal GPU on Apple Silicon (macOS only)"          OFF)

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
