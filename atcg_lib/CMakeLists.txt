# Define the target
charonload_add_torch_library(atcg_lib STATIC)

# Link against cuda if requested
if(ATCG_CUDA_BACKEND)
    set(ATCG_BACKEND "cuda")

    find_package(CUDAToolkit REQUIRED)

    target_include_directories(atcg_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(atcg_lib PUBLIC ATCG_CUDA_BACKEND)
    target_link_libraries(atcg_lib PUBLIC CUDA::cudart_static)
    target_link_libraries(atcg_lib PUBLIC CUDA::nvjpeg)
else()
    set(ATCG_BACKEND "cpu")
endif()

# Find backend files
file (GLOB_RECURSE atcg_lib_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/src/**.c*"
                                                          "${CMAKE_SOURCE_DIR}/atcg_lib/include/**.h")

file(GLOB_RECURSE atcg_backend_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/**.c*"
                                                          "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/**.h")

file (GLOB_RECURSE atcg_renderer_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/**.c*"
                                                              "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/**.h")

file (GLOB_RECURSE atcg_window_files LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/atcg_lib/platform/glfw/**.c*"
                                                            "${CMAKE_SOURCE_DIR}/atcg_lib/platform/glfw/**.h")

target_sources(atcg_lib PRIVATE ${atcg_lib_files} 
                                ${atcg_backend_files} 
                                ${atcg_renderer_files} 
                                ${atcg_window_files})

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES NO_WARN_MBCS_MFC_DEPRECATION _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
target_precompile_headers(atcg_lib PRIVATE "${CMAKE_SOURCE_DIR}/atcg_lib/include/atcgpch.h")

target_include_directories(atcg_lib PUBLIC "${CMAKE_SOURCE_DIR}/atcg_lib/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/include"
                                           "python"
)

target_link_libraries(atcg_lib PUBLIC glfw
                                      imgui
                                      imguizmo
                                      implot
                                      glad
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