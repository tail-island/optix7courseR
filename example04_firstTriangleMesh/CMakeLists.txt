set(TARGET ex04_firstTriangleMesh)

# PTXファイルを作成します。

add_library(${TARGET}_ptx OBJECT
    Params.h
    DeviceProgram.cu
)

target_compile_features(${TARGET}_ptx PUBLIC
    cuda_std_17
)

target_compile_options(${TARGET}_ptx PUBLIC
    -Xcudafe
    --diag_suppress=esa_on_defaulted_function_ignored
    --expt-relaxed-constexpr
)

target_include_directories(${TARGET}_ptx PUBLIC
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
    ../common/Window.h
    Main.cpp
    Model.h
    OptixState.h
    Params.h
    Renderer.h
    Window.h
    ${EMBEDDED_FILE}
)

target_compile_features(${TARGET} PUBLIC
    cxx_std_17
)

target_include_directories(${TARGET} PUBLIC
    ${OptiX_INCLUDE}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${TARGET} PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    glfw
    OpenGL::GL
)
