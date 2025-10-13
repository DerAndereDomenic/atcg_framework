# Collect all source files in this directory
if(ATCG_CUDA_BACKEND)
    set(source source.cpp kernels.h kernels.cu)
else()
    set(source source.cpp kernels.h kernels.cpp)
endif()

# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add target executable
ATCG_add_executable(${TARGET_NAME} target_name "${source}")