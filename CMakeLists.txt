cmake_minimum_required(VERSION 3.24...4.4.4)

project(vk_denoise_nrd LANGUAGES C CXX)

set(NSIGHT_ENABLE_NVTX ON CACHE BOOL "Enable NVTX Markers" FORCE)

# CMake module FetchContent
include(FetchContent)

#-------------------------------------------------------------------------------
# NRD dependency

# Download a specific version of NRD's MathLib, so that we can still compile on
# CMake 3.28.3, the version provided by Ubuntu 24.04 LTS' packages.
# This can be removed once https://github.com/NVIDIA-RTX/NRD/pull/112 is merged in.
FetchContent_Declare(
	mathlib
	DOWNLOAD_EXTRACT_TIMESTAMP 1
	DOWNLOAD_NO_PROGRESS 1
	URL "https://github.com/NVIDIA-RTX/MathLib/archive/e3eee9362880b708a315b9946df51dae332bb46f.zip"
)
FetchContent_MakeAvailable(mathlib)

# Tell CMake where to find NRDSDK, how to get it and where to place it locally.
# CMake will automatically look for CMakeLists.txt in there
FetchContent_Declare(
    NRDSDK
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRD.git
    GIT_TAG v4.17.2
    GIT_SHALLOW
    GIT_SUBMODULES
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/nrd
)

# Configure the NRD build to only produce SPIR-V binaries and embed them into the NRD library
option (NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" OFF)
option (NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" OFF)
option (NRD_STATIC_LIBRARY "NRD linked as static library into executable" ON)

# Cause CMake to download and/or update our local copy of NRDSDK.
FetchContent_MakeAvailable(NRDSDK)

#-------------------------------------------------------------------------------
# nvpro_core2 dependency
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

#--------------------------------------------------------------------------------------------------
# Add example
add_subdirectory(RealtimeDenoiser)
