include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.3.4
)

FetchContent_GetProperties(glfw)

if(NOT glfw_POPULATED)
    message(STATUS "Fetch glfw")
    FetchContent_Populate(glfw)

    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    add_subdirectory(${glfw_SOURCE_DIR})
endif()
