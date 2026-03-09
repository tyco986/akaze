cmake_minimum_required(VERSION 3.15)
project(akazecu LANGUAGES CXX CUDA)
cmake_policy(SET CMP0104 OLD)  # Allow empty CUDA_ARCHITECTURES

set(CMAKE_CXX_STANDARD 11)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Dependencies (no OpenCV)
find_package(CUDA REQUIRED)
find_package(OpenMP)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1)
  FetchContent_MakeAvailable(pybind11)
endif()

# Compilation flags
if(UNIX)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC -msse2 -msse3 -msse4 -ffast-math")
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "75")
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options" "-O2")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Position-independent code required for shared library (Python module)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# AKAZE static library (no OpenCV, no Boost)
set(AKAZE_SRCS
  src/AKAZEConfig.h src/AKAZE.h src/AKAZE.cpp
  src/akaze_types.h
  src/fed.h src/fed.cpp)
add_library(AKAZE STATIC ${AKAZE_SRCS})
target_include_directories(AKAZE PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(AKAZE PUBLIC ${CUDA_LIBRARIES})

# AKAZE_CUDA static library
set(CUDA_SEPARABLE_COMPILATION ON)
add_library(AKAZE_CUDA STATIC
  src/cudaImage.cu src/cuda_akaze.cu)

# Python bindings (pybind11)
pybind11_add_module(libakaze_pybindings python/python_bindings.cpp)
target_include_directories(libakaze_pybindings PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/python
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(libakaze_pybindings PRIVATE
  AKAZE AKAZE_CUDA)
set_target_properties(libakaze_pybindings PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/akaze
  OUTPUT_NAME "libakaze_pybindings")
# Install for pip/skbuild
install(TARGETS libakaze_pybindings LIBRARY DESTINATION .)
