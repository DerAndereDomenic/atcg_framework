if(ATCG_CUDA_BACKEND)

# Collect all source files in this directory
file(GLOB_RECURSE source "**.h*" "**.c*")
# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

filter_ptx_sources("${source}" ptx_files)
list(REMOVE_ITEM source ${ptx_files})
add_ptx_module(${TARGET_NAME} "${ptx_files}")

# Add target executable
ATCG_add_plugin(${TARGET_NAME} target_name "${source}")
set(PLUGIN_TARGET ${target_name} PARENT_SCOPE)

set(ptx_target "${TARGET_NAME}_ptxmodules")

add_dependencies(${TARGET_NAME} ${ptx_target} ${ptx_target}-copy)

endif()