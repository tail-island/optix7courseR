cmake_minimum_required(VERSION 3.18)

project(optix7courseR LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH}
)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL      REQUIRED)
find_package(OptiX       REQUIRED)

include(cmake/Eigen.cmake)
include(cmake/glfw.cmake)
include(cmake/stb.cmake)
include(cmake/tinyobjloader.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# OptiXではbin2cしなければならないらしい。。。

find_program(CUDA_BIN2C bin2c DOC "Path to the cuda-sdk bin2c executable." REQUIRED)

function(bin2c TARGET EMBEDDED_FILE)
    add_custom_command(
        OUTPUT   ${EMBEDDED_FILE}
        COMMAND  ${CMAKE_COMMAND} -DCUDA_BIN2C=${CUDA_BIN2C} -DOBJECTS=$<TARGET_OBJECTS:${TARGET}> -DEMBEDDED_FILE=${EMBEDDED_FILE} -P${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
        VERBATIM
        DEPENDS  $<TARGET_OBJECTS:${TARGET}>
        COMMENT  Convert PTXs to a C file.
    )
endfunction()

add_subdirectory(example01-hello-optix)
add_subdirectory(example02-pipeline-and-raygen)
add_subdirectory(example03-in-glfw-window)
add_subdirectory(example04-first-triangle-mesh)
add_subdirectory(example05-first-shading)
add_subdirectory(example06-multiple-objects)
add_subdirectory(example07-first-real-model)
add_subdirectory(example08-textures)
add_subdirectory(example09-shadow-rays)
add_subdirectory(example10-soft-shadows)
add_subdirectory(example11-accumulate)
