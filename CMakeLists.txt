# neighborlist: one LibTorch operator library (`torch.ops.neighborlist.*`),
# CPU always, CUDA when available. Built by scikit-build-core (pyproject.toml)
# against the torch installed in the build environment; no Python linkage, so
# the library loads through torch.ops.load_library.

cmake_minimum_required(VERSION 3.26)

# No build type means no optimisation flag at all. Release unless asked.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(neighborlist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The torch of the interpreter that builds the wheel. Its headers and libraries
# are declared directly rather than through find_package(Torch): torch's own
# config refuses a CUDA torch without a CUDA toolkit, even for a CPU-only build.
find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).parent); print(torch.version.cuda or ''); print(int(torch.compiled_with_cxx11_abi()))"
  OUTPUT_VARIABLE _torch_info OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" _torch_info "${_torch_info}")
list(GET _torch_info 0 _torch_root)
list(LENGTH _torch_info _torch_fields)
if(_torch_fields EQUAL 3)
  list(GET _torch_info 1 _torch_cuda)
  list(GET _torch_info 2 _torch_abi)
else()  # a CPU torch prints an empty CUDA line
  set(_torch_cuda "")
  list(GET _torch_info 1 _torch_abi)
endif()

# NEIGHBORLIST_CUDA: AUTO (a CUDA torch and a CUDA compiler), ON or OFF.
set(NEIGHBORLIST_CUDA AUTO CACHE STRING "Build the CUDA kernel: AUTO, ON or OFF")
set(_with_cuda OFF)
if(NEIGHBORLIST_CUDA STREQUAL "ON")
  set(_with_cuda ON)
elseif(NEIGHBORLIST_CUDA STREQUAL "AUTO" AND NOT _torch_cuda STREQUAL "")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(_with_cuda ON)
  endif()
endif()
if(_with_cuda)
  if(_torch_cuda STREQUAL "")
    message(FATAL_ERROR "NEIGHBORLIST_CUDA=ON needs a CUDA build of torch")
  endif()
  # Architectures: TORCH_CUDA_ARCH_LIST ("8.0;9.0", as torch spells it), else
  # every major architecture the compiler knows.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
      string(REPLACE "." "" _archs "$ENV{TORCH_CUDA_ARCH_LIST}")
      string(REPLACE " " ";" _archs "${_archs}")
      string(REPLACE "+PTX" "" _archs "${_archs}")
      set(CMAKE_CUDA_ARCHITECTURES "${_archs}")
    else()
      set(CMAKE_CUDA_ARCHITECTURES all-major)
    endif()
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  string(REGEX MATCH "^[0-9]+" _torch_cuda_major "${_torch_cuda}")
  if(NOT _torch_cuda_major STREQUAL CUDAToolkit_VERSION_MAJOR)
    message(FATAL_ERROR "torch is built with CUDA ${_torch_cuda} but the toolkit is ${CUDAToolkit_VERSION}; the major versions must match")
  endif()
endif()
message(STATUS "neighborlist: torch at ${_torch_root} (CUDA '${_torch_cuda}'), CUDA kernel=${_with_cuda}")

set(_torch_libraries c10 torch torch_cpu)
if(_with_cuda)
  list(APPEND _torch_libraries c10_cuda torch_cuda)
endif()
add_library(torch_deps INTERFACE)
target_include_directories(torch_deps SYSTEM INTERFACE
  "${_torch_root}/include" "${_torch_root}/include/torch/csrc/api/include")
target_compile_definitions(torch_deps INTERFACE _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
foreach(_library IN LISTS _torch_libraries)
  find_library(_torch_${_library} NAMES ${_library} PATHS "${_torch_root}/lib" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(torch_deps INTERFACE "${_torch_${_library}}")
endforeach()
if(_with_cuda)
  target_link_libraries(torch_deps INTERFACE CUDA::cudart)
endif()

set(_sources csrc/library.cpp csrc/neighbor_pairs.cpp)
if(_with_cuda)
  list(APPEND _sources csrc/neighbor_pairs.cu)
  # CPU and CUDA decide every candidate with the same IEEE operations, so they
  # return identical rows: no FMA contraction in the kernel.
  set_source_files_properties(csrc/neighbor_pairs.cu PROPERTIES COMPILE_OPTIONS "--fmad=false;--expt-relaxed-constexpr")
endif()

add_library(neighborlist_ops SHARED ${_sources})
target_include_directories(neighborlist_ops PRIVATE csrc)
target_link_libraries(neighborlist_ops PRIVATE torch_deps)

# at::parallel_for under torch's OpenMP backend is an inline `#pragma omp
# parallel`: without -fopenmp the CPU kernel silently runs on one thread.
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(neighborlist_ops PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS neighborlist_ops LIBRARY DESTINATION neighborlist RUNTIME DESTINATION neighborlist)
