set(EMBEDDED_CONTENTS)

foreach(OBJECT ${OBJECTS})
    get_filename_component(OBJECT_EXT       ${OBJECT} EXT)
    get_filename_component(OBJECT_NAME      ${OBJECT} NAME_WE)
    get_filename_component(OBJECT_DIRECTORY ${OBJECT} DIRECTORY)

    if(OBJECT_EXT MATCHES ".ptx")
        execute_process(
            COMMAND ${CUDA_BIN2C} --padd 0 --type char --name ${OBJECT_NAME} ${OBJECT}
            WORKING_DIRECTORY ${OBJECT_DIRECTORY}
            OUTPUT_VARIABLE OUTPUT
        )

        set(EMBEDDED_CONTENTS "${EMBEDDED_CONTENTS}\n${OUTPUT}")
    endif()
endforeach()

file(WRITE ${EMBEDDED_FILE} "${EMBEDDED_CONTENTS}")
