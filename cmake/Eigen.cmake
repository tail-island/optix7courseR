include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen/
    GIT_TAG        3.4.0
)

FetchContent_MakeAvailable(eigen)
