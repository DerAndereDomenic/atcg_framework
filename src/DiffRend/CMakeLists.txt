include(FilterPTX) 


# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add target executable
set(source_diffrend source_diffrend.cpp)
set(source_multiview source_multiview.cpp)

ATCG_add_executable(DiffRend target_name "${source_diffrend}")
ATCG_add_executable(DiffRend_Multiview target_name "${source_multiview}")

add_dependencies(DiffRend AllPlugins)
add_dependencies(DiffRend_Multiview AllPlugins)