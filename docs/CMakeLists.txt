find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

file(GLOB_RECURSE ATCG_LIB_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../atcg_lib/include/**.h"
                                          "${CMAKE_CURRENT_SOURCE_DIR}/../atcg_lib/platform/glfw/**.h")
set(DOXYGEN_INDEX_FILE "${CMAKE_CURRENT_SOURCE_DIR}/html/index.html")

#set(DOXYGEN_INPUT_DIR "${PROJECT_SOURCE_DIR}/atcg_lib/include")
set(DOXYGEN_DIRS
  ${PROJECT_SOURCE_DIR}/atcg_lib/include
  ${PROJECT_SOURCE_DIR}/atcg_lib/platform/glfw/include
)
string(REPLACE ";" " " DOXYGEN_DIRS_STR "${DOXYGEN_DIRS}")  
set(DOXYGEN_INPUT_DIR ${DOXYGEN_DIRS_STR})
set(DOXYGEN_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Doxyfile)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/docs/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${ATCG_LIB_PUBLIC_HEADERS}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   MAIN_DEPENDENCY ${DOXYFILE_IN}
                   COMMENT "Generating docs"
                   VERBATIM)

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND 
                     ${SPHINX_EXECUTABLE} -b html
                     # Tell Breathe where to find the Doxygen output
                     -Dbreathe_projects.ATCGLIB=${DOXYGEN_OUTPUT_DIR}/xml
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                   # Other docs files you want to track should go here (or in some variable)
                   ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                   ${DOXYGEN_INDEX_FILE}
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})