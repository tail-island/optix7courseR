include(FetchContent)

FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader/
    GIT_TAG        v1.0.6
)

FetchContent_MakeAvailable(tinyobjloader)
