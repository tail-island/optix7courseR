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

add_subdirectory(example01_helloOptiX)
add_subdirectory(example02_pipelineAndRayGen)
add_subdirectory(example03_inGLFWindow)
add_subdirectory(example04_firstTriangleMesh)
