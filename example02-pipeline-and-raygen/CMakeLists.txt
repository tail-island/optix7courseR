set(TARGET example02-pipeline-and-raygen)

# PTXファイルを作成します。

add_library(${TARGET}_ptx OBJECT
    DeviceProgram.cu
    OptixParams.h
)

target_compile_features(${TARGET}_ptx PRIVATE
    cuda_std_17
)

target_include_directories(${TARGET}_ptx PRIVATE
    ${OptiX_INCLUDE}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_property(TARGET ${TARGET}_ptx PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ${TARGET}_ptx PROPERTY CUDA_ARCHITECTURES   OFF)  # WindowsでCUDA_ARCHITECTUREを指定すると、Visual Studio 2019のプロジェクトのCUDA C/C++ > Device > Code Generationの値が不正になるので、指定せずにOFFにしました。。。

# bin2cして、cppファイルを作成します。

set(EMBEDDED_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_ptx_embedded.cpp)

bin2c(${TARGET}_ptx ${EMBEDDED_FILE})

# 実行ファイルを作成します。

add_executable(${TARGET}
    ../common/DeviceBuffer.h
    ../common/DeviceVectorBuffer.h
    ../common/Util.h
    Main.cpp
    OptixParams.h
    OptixState.h
    Renderer.h
    ${EMBEDDED_FILE}
)

target_compile_features(${TARGET} PRIVATE
    cxx_std_17
)

target_include_directories(${TARGET} PRIVATE
    ${OptiX_INCLUDE}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${TARGET}
    CUDA::cudart
    CUDA::cuda_driver
    dl
)
