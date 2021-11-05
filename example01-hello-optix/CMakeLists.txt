set(TARGET example01-hello-optix)

add_executable(${TARGET}
    Main.cpp
    HelloOptix.h
    ../common/Util.h
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
    dl
)
