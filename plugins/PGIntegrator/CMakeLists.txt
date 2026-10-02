include(FilterPTX) 

file(GLOB_RECURSE source "**.h*" "**.c*")

# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(ATCG_CUDA_BACKEND)
    filter_ptx_sources("${source}" pgintegator_ptx_files)
    list(REMOVE_ITEM source ${pgintegator_ptx_files})
    add_ptx_module(${TARGET_NAME} "${pgintegator_ptx_files}")
endif()

# Add target executable
ATCG_add_plugin(${TARGET_NAME} target_name "${source}")

if(ATCG_CUDA_BACKEND)
    set(ptx_target "${TARGET_NAME}_ptxmodules")

    add_dependencies(${TARGET_NAME} ${ptx_target} ${ptx_target}-copy)
endif()

set(PLUGIN_TARGET ${target_name} PARENT_SCOPE)