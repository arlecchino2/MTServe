#
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.18)
project(HSTUAttention LANGUAGES CXX CUDA)

message(STATUS "Building HSTUAttention kernels from source.")

set(TF_VERSION,"")
#check TF version

find_package(CUDAToolkit REQUIRED)
find_package(Threads)

set(CUDA_SEPARABLE_COMPILATION ON)

#if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  foreach(arch_name ${SM})
      if (arch_name STREQUAL 90 OR
          arch_name STREQUAL 80 OR
          arch_name STREQUAL 89 OR
          arch_name STREQUAL 75 OR
          arch_name STREQUAL 70)
          list(APPEND cuda_arch_list ${arch_name}-real)
      elseif (arch_name STREQUAL 61 OR
              arch_name STREQUAL 60)
          message(WARNING "-- The specified architecture ${arch_name} is excluded because it is not supported")
      else()
          message(FATAL_ERROR "-- ${arch_name} is an invalid or unsupported architecture")
      endif()
  endforeach()

  list(LENGTH cuda_arch_list cuda_arch_list_length)
  if(${cuda_arch_list_length} EQUAL 0)
      list(APPEND cuda_arch_list 89-real)
  endif()
  list(REMOVE_DUPLICATES cuda_arch_list)

  set(CMAKE_CUDA_ARCHITECTURES ${cuda_arch_list})
#endif()

message(STATUS "Target GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wno-error=cpp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# headers
include_directories(
    ${PROJECT_SOURCE_DIR}/csrc/hstu_attn/src
    ${PROJECT_SOURCE_DIR}/csrc/cutlass/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# libs
link_directories(
    /usr/local/cuda/lib64/
)

# code sources related to framework
file(GLOB srcs
    ${PROJECT_SOURCE_DIR}/csrc/hstu_attn/src/*.cc
    ${PROJECT_SOURCE_DIR}/csrc/hstu_attn/src/*.cpp
    ${PROJECT_SOURCE_DIR}/csrc/hstu_attn/src/*.cu
)
list(FILTER srcs EXCLUDE REGEX ".*\\hstu\.(cu|h)$")

# build dynamic lib
add_library(hstu_attn_kernels SHARED ${srcs})
set_target_properties(hstu_attn_kernels PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_compile_definitions(hstu_attn_kernels PRIVATE _GLIBCXX_USE_CXX11_ABI=1)