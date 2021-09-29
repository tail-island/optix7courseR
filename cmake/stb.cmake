include(FetchContent)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb
)

FetchContent_GetProperties(stb)

if(NOT stb_POPULATED)
    message(STATUS "Fetch stb")
    FetchContent_Populate(stb)

    include_directories(${stb_SOURCE_DIR}/)
endif()
