# Define the target
charonload_add_torch_library(atcg_lib STATIC)

# Link against cuda if requested
if(ATCG_CUDA_BACKEND)
    set(ATCG_BACKEND "cuda")

    if(NOT CUDAToolkit_FOUND)
        find_package(CUDAToolkit REQUIRED)
    endif()

    set_target_properties(atcg_lib PROPERTIES
        CUDA_ARCHITECTURES native
    )
    target_compile_options(atcg_lib PUBLIC 
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )

    target_include_directories(atcg_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    if(ATCG_ENABLE_OPTIX)
        target_include_directories(atcg_lib PUBLIC ${OptiX_INCLUDE})
        target_compile_definitions(atcg_lib PUBLIC ATCG_ENABLE_OPTIX)
    endif()

    target_compile_definitions(atcg_lib PUBLIC ATCG_CUDA_BACKEND)
    target_link_libraries(atcg_lib PUBLIC CUDA::cudart_static)
    target_link_libraries(atcg_lib PUBLIC CUDA::nvjpeg)
else()
    set(ATCG_BACKEND "cpu")
endif()

if(ATCG_HEADLESS)
    target_compile_definitions(atcg_lib PUBLIC ATCG_HEADLESS)
    set(ATCG_CONTEXT "headless")
    target_link_libraries(atcg_lib PUBLIC OpenGL::EGL)
else()
    set(ATCG_CONTEXT "glfw")
    target_link_libraries(atcg_lib PUBLIC glfw
                                          atcg::imgui
                                          atcg::imguizmo
                                          atcg::implot)
endif()

# Find backend files
file (GLOB_RECURSE atcg_lib_files LIST_DIRECTORIES false "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/src/**.c*"
                                                          "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/include/**.h")

file(GLOB_RECURSE atcg_backend_files LIST_DIRECTORIES false "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/${ATCG_BACKEND}/**.c*"
                                                          "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/${ATCG_BACKEND}/**.h")

file (GLOB_RECURSE atcg_renderer_files LIST_DIRECTORIES false "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/opengl/**.c*"
                                                              "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/opengl/**.h")


file (GLOB_RECURSE atcg_window_files LIST_DIRECTORIES false "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/**.c*"
                                                            "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/**.h")

if(ATCG_ENABLE_OPTIX)
    list(APPEND CMAKE_MODULE_PATH "${ATCG_PROJECT_ROOT_DIR}/cmake")
    include(FilterPTX)

    file (GLOB_RECURSE atcg_optix_files LIST_DIRECTORIES false "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/optix/**.c*"
                                                               "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/optix/**.h")

    filter_ptx_sources("${atcg_optix_files}" ptx_files)
    list(REMOVE_ITEM atcg_optix_files ${ptx_files})
    add_ptx_module(atcg "${ptx_files}")
endif()

target_sources(atcg_lib PRIVATE ${atcg_lib_files} 
                                ${atcg_backend_files} 
                                ${atcg_renderer_files} 
                                ${atcg_window_files}
                                ${atcg_optix_files})

if (NOT ATCG_PYTHON_MODULE)
    file (GLOB_RECURSE atcg_lib_python_files LIST_DIRECTORIES false "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/python/**.h"
                                                             "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/python/**.c*")
    target_sources(atcg_lib PRIVATE ${atcg_lib_python_files})
    target_compile_definitions(atcg_lib PUBLIC TORCH_EXTENSION_NAME=pyatcg)
    target_compile_definitions(atcg_lib PUBLIC ATCG_INIT_PYTHON)
endif()

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES 
                                            NO_WARN_MBCS_MFC_DEPRECATION 
                                            _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS 
                                            _CRT_NONSTDC_NO_WARNINGS
                                            ATCG_BASE_DIR="${ATCG_PROJECT_ROOT_DIR}"
                                            ATCG_SHADER_DIR="${ATCG_PROJECT_ROOT_DIR}/shader"
                                            ATCG_RESOURCE_DIR="${ATCG_PROJECT_ROOT_DIR}/res")
target_precompile_headers(atcg_lib PRIVATE "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/include/atcgpch.h")

target_include_directories(atcg_lib PUBLIC "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/include"
                                           "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/${ATCG_BACKEND}/include"
                                           "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/include"
                                           "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/opengl/include"
                                           "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/optix/include"
                                           "${ATCG_PROJECT_ROOT_DIR}/atcg_lib/platform/python"
)

target_link_libraries(atcg_lib PUBLIC atcg::glad
                                      OpenGL::GL
                                      OpenMeshCore
                                      OpenMeshTools
                                      stb_image
                                      atcg::tinyobjloader
                                      atcg::entt
                                      atcg::nanort
                                      atcg::portable_file_dialogs
                                      nanoflann
                                      glm
                                      spdlog_header_only
                                      atcg::json
                                      SFML::System
                                      SFML::Network
                                      Python::Python
                                      torch_python_interface
                                      atcg::openvr
)