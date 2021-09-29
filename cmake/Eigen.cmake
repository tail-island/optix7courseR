include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen/
    GIT_TAG        3.4.0
)

FetchContent_GetProperties(eigen)

if(NOT eigen_POPULATED)
    message(STATUS "Fetch Eigen")
    FetchContent_Populate(eigen)

    include_directories(${eigen_SOURCE_DIR}/)
endif()
