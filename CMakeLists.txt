cmake_minimum_required(VERSION 3.11...3.31)

project(vk_denoise_nrd LANGUAGES C CXX)

set(NSIGHT_ENABLE_NVTX ON CACHE BOOL "Enable NVTX Markers" FORCE)

# CMake module FetchContent
include(FetchContent)

#-------------------------------------------------------------------------------
# NRD dependency

# Tell CMake where to find NRDSDK, how to get it and where to place it locally.
# CMake will automatically look for CMakeLists.txt in there
FetchContent_Declare(
    NRDSDK
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRD.git
    GIT_TAG v4.16.1
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
