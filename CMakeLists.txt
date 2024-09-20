# Common default Cmake header for Nvpro-core samples
cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include(FetchContent)

project(${PROJECT_NAME} LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 17)

FetchContent_Declare(
    NRDSDK
    GIT_REPOSITORY https://github.com/NVIDIAGameWorks/RayTracingDenoiser.git
    GIT_SHALLOW
	GIT_SUBMODULES
	SOURCE_DIR	${CMAKE_CURRENT_SOURCE_DIR}/externals/nrd
)

#--------------------------------------------------------------------------------------------------
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()

## Various functions and macros REQUIRED
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

_add_project_definitions(${PROJECT_NAME})
set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

if(MSVC)
    add_definitions(/wd26812)  # 'enum class' over 'enum'
    add_definitions(/wd26451)  # Arithmetic overflow, casting 4 byte value to 8 byte value
endif()


# WE don't need the DXIL/DXBC shaders for Vulkan
option (NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" OFF)
option (NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" OFF)
set(NRD_EMBEDS_DXIL_SHADERS OFF)
set(NRD_EMBEDS_DXBC_SHADERS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_PATH}/$<CONFIG>/" CACHE PATH "")

FetchContent_MakeAvailable(NRDSDK)

#--------------------------------------------------------------------------------------------------
# Add example
add_subdirectory(RealtimeDenoiser)

#--------------------------------------------------------------------------------------------------
# Install - copying the media directory
message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})
install(DIRECTORY "media" CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
