include(FilterPTX) 

# Collect all source files in this directory
if(ATCG_CUDA_BACKEND)
    file(GLOB_RECURSE source "**.h*" "**.c*")
endif()

list(REMOVE_ITEM source "${CMAKE_CURRENT_SOURCE_DIR}/source_diffrend.cpp")
list(REMOVE_ITEM source "${CMAKE_CURRENT_SOURCE_DIR}/source_multiview.cpp")

# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(ATCG_CUDA_BACKEND)
    filter_ptx_sources("${source}" diffrend_ptx_files)
    list(REMOVE_ITEM source ${diffrend_ptx_files})
    add_ptx_module(${TARGET_NAME} "${diffrend_ptx_files}")
endif()

# Add target executable
set(source_diffrend ${source})
list(APPEND source_diffrend source_diffrend.cpp)

set(source_multiview ${source})
list(APPEND source_multiview source_multiview.cpp)
ATCG_add_executable(DiffRend target_name "${source_diffrend}")
ATCG_add_executable(DiffRend_Multiview target_name "${source_multiview}")

if(ATCG_CUDA_BACKEND)
    set(ptx_target "${TARGET_NAME}_ptxmodules")

    add_dependencies(DiffRend ${ptx_target} ${ptx_target}-copy)
    add_dependencies(DiffRend_Multiview ${ptx_target} ${ptx_target}-copy)
endif()