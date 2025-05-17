include(FilterPTX) 

# Collect all source files in this directory
file(GLOB_RECURSE source "**.h*" "**.c*")

# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

filter_ptx_sources("${source}" radiosity_ptx_files)
list(REMOVE_ITEM source ${radiosity_ptx_files})
add_ptx_module(${TARGET_NAME} "${radiosity_ptx_files}")

# Add target executable
ATCG_add_executable(${TARGET_NAME} target_name "${source}")
set(ptx_target "${TARGET_NAME}_ptxmodules")

add_dependencies(${TARGET_NAME} ${ptx_target} ${ptx_target}-copy)