# Define the target
charonload_add_torch_library(atcg_lib STATIC)

# Link against cuda if requested
if(ATCG_CUDA_BACKEND)
    set(ATCG_BACKEND "cuda")

    find_package(CUDAToolkit REQUIRED)

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
                                          imgui
                                          imguizmo
                                          implot)
endif()

# Find backend files
file (GLOB_RECURSE atcg_lib_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/src/**.c*"
                                                          "${CMAKE_SOURCE_DIR}/atcg_lib/include/**.h")

file(GLOB_RECURSE atcg_backend_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/**.c*"
                                                          "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/**.h")

file (GLOB_RECURSE atcg_renderer_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/**.c*"
                                                              "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/**.h")


file (GLOB_RECURSE atcg_window_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/**.c*"
                                                            "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/**.h")

if(ATCG_ENABLE_OPTIX)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    include(FilterPTX)

    file (GLOB_RECURSE atcg_optix_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/optix/**.c*"
                                                            "${CMAKE_SOURCE_DIR}/atcg_lib/platform/optix/**.h")

    filter_ptx_sources(atcg_optix_files ptx_files)
    list(REMOVE_ITEM atcg_optix_files ${ptx_files})

    if(ptx_files)
        add_library(atcg_ptxmodules OBJECT ${ptx_files})
        set_target_properties(atcg_ptxmodules PROPERTIES
            CUDA_PTX_COMPILATION ON
            CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(atcg_ptxmodules PUBLIC ATCG_CUDA_BACKEND _USE_MATH_DEFINES)
        target_compile_options(atcg_ptxmodules PRIVATE -ptx)
        target_link_libraries(atcg_ptxmodules PRIVATE atcg_lib)

        set(ATCG_PTX_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        add_custom_target(atcg_ptxmodules-copy
            DEPENDS atcg_ptxmodules
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:atcg_ptxmodules> ${ATCG_PTX_OUTPUT_DIRECTORY}
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

target_sources(atcg_lib PRIVATE ${atcg_lib_files} 
                                ${atcg_backend_files} 
                                ${atcg_renderer_files} 
                                ${atcg_window_files}
                                ${atcg_optix_files})

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES 
                                            NO_WARN_MBCS_MFC_DEPRECATION 
                                            _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS 
                                            _CRT_NONSTDC_NO_WARNINGS
                                            ATCG_BASE_DIR="${CMAKE_SOURCE_DIR}"
                                            ATCG_SHADER_DIR="${CMAKE_SOURCE_DIR}/shader"
                                            ATCG_RESOURCE_DIR="${CMAKE_SOURCE_DIR}/res")
target_precompile_headers(atcg_lib PRIVATE "${CMAKE_SOURCE_DIR}/atcg_lib/include/atcgpch.h")

target_include_directories(atcg_lib PUBLIC "${CMAKE_SOURCE_DIR}/atcg_lib/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/optix/include"
                                           "python"
)

target_link_libraries(atcg_lib PUBLIC glad
                                      OpenGL::GL
                                      OpenMeshCore
                                      OpenMeshTools
                                      stb_image
                                      tinyobjloader
                                      entt
                                      nanort
                                      portable_file_dialogs
                                      nanoflann
                                      glm
                                      spdlog_header_only
                                      json
                                      SFML::System
                                      SFML::Network
                                      ${ATCG_OPENVR_TARGET}
)