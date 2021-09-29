set(TARGET ex01_helloOptiX)

add_executable(${TARGET}
    Main.cpp
    HelloOptiX.h
    ../common/Util.h
)

target_compile_features(${TARGET} PUBLIC
    cxx_std_17
)

target_include_directories(${TARGET} PUBLIC
    ${OptiX_INCLUDE}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${TARGET} PUBLIC
    CUDA::cudart
)
