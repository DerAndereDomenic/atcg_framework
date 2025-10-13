include(FilterPTX) 

# Collect all source files in this directory
if(ATCG_ENABLE_OPTIX)
    file(GLOB_RECURSE source "**.h*" "**.c*")
else()
    set(source "Radiosity.cpp"
               "Radiosity.h"
               "source.cpp")
endif()

# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if(ATCG_ENABLE_OPTIX)
    filter_ptx_sources("${source}" radiosity_ptx_files)
    list(REMOVE_ITEM source ${radiosity_ptx_files})
    add_ptx_module(${TARGET_NAME} "${radiosity_ptx_files}")
endif()

# Add target executable
ATCG_add_executable(${TARGET_NAME} target_name "${source}")

if(ATCG_ENABLE_OPTIX)
    set(ptx_target "${TARGET_NAME}_ptxmodules")

    add_dependencies(${TARGET_NAME} ${ptx_target} ${ptx_target}-copy)
endif()