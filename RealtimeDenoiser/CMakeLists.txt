#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------

message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

file(GLOB CPP_SOURCES src/*.cpp src/*.hpp src/*.h)

add_executable(${PROJECT_NAME} ${CPP_SOURCES})

find_package(OpenMP REQUIRED COMPONENTS CXX)
add_compile_options(-fno-omit-frame-pointer)

target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_DEFINE_MATH_OPERATORS)

target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvshaders_host
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvgui
  nvpro2::nvvkgltf
  nvpro2::nvaftermath
  NRD
  OpenMP::OpenMP_CXX
)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(NRD_SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../externals/nrd/Shaders/Include)

# Local shaders - Slang (new)
file(GLOB SHD_SLANG_HDR ${SHD_DIR}/*.h ${NRD_SHD_DIR}/*.hlsli)
file(GLOB SHD_SLANG_SRC ${SHD_DIR}/*.slang)

# Shaders to the build
list(APPEND SHADER_SLANG_FILES
    ${NVSHADERS_DIR}/nvshaders/sky_physical.slang
    ${NVSHADERS_DIR}/nvshaders/sky_simple.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_diffuse.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_glossy.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_dome.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_integrate_brdf.slang
    ${NVSHADERS_DIR}/nvshaders/tonemapper.slang

    ${SHD_DIR}/nrd.slang
    ${SHD_DIR}/nrd_rmiss.slang
    ${SHD_DIR}/nrd_rchit.slang
    ${SHD_DIR}/pathtrace_rchit.slang
    ${SHD_DIR}/pathtrace_rahit.slang
    ${SHD_DIR}/pathtrace_rmiss.slang
    ${SHD_DIR}/compositing.slang
    ${SHD_DIR}/taa.slang
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/_autogen")

# Compiling shaders to Spir-V header
compile_slang(
    "${SHADER_SLANG_FILES}"
    "${SHADER_OUTPUT_DIR}"
    HEADERS_VAR GENERATED_SHADER_HEADERS
    EXTRA_FLAGS "-I${NVSHADERS_DIR}" "-I${SHD_DIR}" "-I${NRD_SHD_DIR}"
    CAPABILITIES spvGroupNonUniform spvGroupNonUniformArithmetic
)

message(STATUS "NVSHADERS_DIR ${NVSHADERS_DIR}")

source_group("Shaders" FILES  ${SHD_SLANG_SRC} ${SHD_SLANG_HDR} ${SHADER_SLANG_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_HEADERS})
source_group("Documentation" FILES "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")

target_sources(${PROJECT_NAME} PRIVATE ${SHD_SLANG_SRC} ${SHD_SLANG_HDR} ${GENERATED_SHADER_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/../README.md)

# Let it find the "./_autogen/" shaders
target_include_directories(${PROJECT_NAME} PRIVATE ${SHADER_OUTPUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})


download_files(FILENAMES "ABeautifulGame.zip" EXTRACT)
download_files(FILENAMES "daytime.hdr" "environment.hdr" "hdri-19_color_small.hdr" "std_env.hdr")

add_project_definitions(${PROJECT_NAME})

# Make Visual Studio use this project as the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
