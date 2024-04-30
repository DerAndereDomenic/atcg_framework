cmake_minimum_required(VERSION 3.24)

project(ATCG_FRAMEWORK LANGUAGES CXX C VERSION 1.0)

option(ATCG_CUDA_BACKEND "Compile cuda backend" OFF)
option(ATCG_PYTHON_BINDINGS "Compile python bindings" OFF)
option(ATCG_BUILD_DOCS "Build documentation" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# GLFW
add_subdirectory(external/glfw)

# OpenMesh
add_subdirectory(external/openmesh)
target_compile_definitions(OpenMeshCore PUBLIC OM_STATIC_BUILD)
target_compile_definitions(OpenMeshTools PUBLIC OM_STATIC_BUILD)

# stbimage
add_subdirectory(external/stbimage)
target_include_directories(stb_image PUBLIC "external/stbimage")

# imguizmo
add_library(imguizmo STATIC)
target_sources(imguizmo PRIVATE "external/imguizmo/GraphEditor.cpp"
                                "external/imguizmo/GraphEditor.h"
                                "external/imguizmo/ImCurveEdit.cpp"
                                "external/imguizmo/ImCurveEdit.h"
                                "external/imguizmo/ImGradient.cpp"
                                "external/imguizmo/ImGradient.h"
                                "external/imguizmo/ImGuizmo.cpp"
                                "external/imguizmo/ImGuizmo.h"
                                "external/imguizmo/ImSequencer.cpp"
                                "external/imguizmo/ImSequencer.h"
                                "external/imguizmo/ImZoomSlider.h")
target_include_directories(imguizmo PUBLIC "external/imguizmo")
target_link_libraries(imguizmo PUBLIC imgui)

# charonload
add_subdirectory(external/charonload)

# glad
add_library(glad STATIC)
target_sources(glad PRIVATE "external/glad/src/glad.c")
target_include_directories(glad PUBLIC "external/glad/include")

# imgui
add_library(imgui STATIC)
target_sources(imgui PRIVATE "external/imgui/imgui.cpp"
                             "external/imgui/imgui_demo.cpp"
                             "external/imgui/imgui_draw.cpp"
                             "external/imgui/imgui_tables.cpp"
                             "external/imgui/imgui_widgets.cpp"
                             "external/imgui/backends/imgui_impl_glfw.cpp"
                             "external/imgui/backends/imgui_impl_opengl3.cpp")
target_include_directories(imgui PUBLIC "external/imgui")
target_link_libraries(imgui PRIVATE glfw)

# implot
add_library(implot STATIC)
target_sources(implot PRIVATE "external/implot/implot.cpp"
                              "external/implot/implot_items.cpp")
target_include_directories(implot PUBLIC "external/implot")
target_link_libraries(implot PUBLIC imgui)

# tinyobjloader
add_library(tinyobjloader STATIC "external/tinyobjloader/tiny_obj_loader.cc")
target_include_directories(implot PUBLIC "external/tinyobjloader")

# entt
add_library(entt INTERFACE)
target_include_directories(entt INTERFACE "external/entt/include")

# nanort
add_library(nanort INTERFACE)
target_include_directories(nanort INTERFACE "external/nanort")

# portable-file-dialogs
add_library(portable_file_dialogs INTERFACE)
target_include_directories(portable_file_dialogs INTERFACE "external/portable-file-dialogs")

# nanoflann
set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/nanoflann)

# glm
add_subdirectory(external/glm)

# spdlog
add_subdirectory(external/spdlog)
if(NOT MSVC)
    target_compile_definitions(spdlog_header_only INTERFACE FMT_USE_CONSTEXPR)
endif()

# openvr
set(BUILD_SHARED ON CACHE BOOL "Builds the library as shared library" FORCE)
add_subdirectory(external/openvr)
if(TARGET openvr_api)
    set(ATCG_OPENVR_TARGET openvr_api)
elseif(TARGET openvr_api64)
    set(ATCG_OPENVR_TARGET openvr_api64)
endif()
set_target_properties(${ATCG_OPENVR_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                                                       RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                                                       ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# json
add_library(json INTERFACE)
target_include_directories(json INTERFACE "external/json")

if(ATCG_CUDA_BACKEND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    set(ATCG_BACKEND "cuda")
    find_package(OptiX REQUIRED)
else()
    set(ATCG_BACKEND "cpu")
endif()

file (GLOB_RECURSE atcg_lib_files LIST_DIRECTORIES false "atcg_lib/src/**.c*"
                                                          "atcg_lib/include/**.h")

file(GLOB_RECURSE atcg_backend_files LIST_DIRECTORIES false "atcg_lib/platform/${ATCG_BACKEND}/**.c*"
                                                          "atcg_lib/platform/${ATCG_BACKEND}/**.h")

file (GLOB_RECURSE atcg_renderer_files LIST_DIRECTORIES false "atcg_lib/platform/opengl/**.c*"
                                                              "atcg_lib/platform/opengl/**.h")

set(source_files "${atcg_lib_files}" "${atcg_backend_files}" "${atcg_renderer_files}")

set(ptx_files "")
if(ATCG_CUDA_BACKEND)
    foreach(source_file IN LISTS source_files)
        # Define the pragma we use to determine the value of CUDA_SOURCE_PROPERTY_FORMAT
        set(pragma_expression "^[ \t]*#[ \t]*pragma[ \t]+cuda_source_property_format[ \t]+=[ \t]+([a-z,A-Z]+)[ \t]*$")
        # Process any *.cu source files
        if(${source_file} MATCHES "\\.cu$")
            # Read all lines from the source file defining the pragma we are looking for.
            file(STRINGS ${source_file} matched_content REGEX ${pragma_expression})
            if (matched_content)
                # For all pragma definitions that we found, should only be one usually...
                foreach(line_string ${matched_content})
                    # Filter out the defined value
                    string(REGEX REPLACE ${pragma_expression} "\\1" source_property_format ${line_string})
                    # Apply the property to the source file
                    if(${source_property_format} MATCHES "PTX")
                        list(APPEND ptx_files ${source_file})
                    endif()
                endforeach()
            endif()
        endif()
    endforeach()
endif()

list(REMOVE_ITEM source_files ${ptx_files})

charonload_add_torch_library(atcg_lib STATIC)
target_sources(atcg_lib PRIVATE ${source_files})

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES NO_WARN_MBCS_MFC_DEPRECATION _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
target_precompile_headers(atcg_lib PRIVATE "atcg_lib/include/atcgpch.h")

target_include_directories(atcg_lib PUBLIC "atcg_lib/include"
                                           "atcg_lib/platform/${ATCG_BACKEND}/include"
                                           "atcg_lib/platform/opengl/include"
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
                                      ${ATCG_OPENVR_TARGET}
)

if(ATCG_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED)

    target_include_directories(atcg_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
                                               "${OptiX_INCLUDE}")
    target_compile_definitions(atcg_lib PUBLIC ATCG_CUDA_BACKEND)
    target_link_libraries(atcg_lib PUBLIC CUDA::cudart_static)

    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")

    add_library(ptxmodules OBJECT ${ptx_files})
    set_target_properties(ptxmodules PROPERTIES
        CUDA_PTX_COMPILATION ON
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(ptxmodules PUBLIC ATCG_CUDA_BACKEND _USE_MATH_DEFINES)
    target_compile_options(ptxmodules PRIVATE -ptx)
    target_link_libraries(ptxmodules PRIVATE atcg_lib)
endif()

function(ATCG_add_executable target_name_base target_name_var source)
    set(target_name ${target_name_base})
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    if(ATCG_PYTHON_BINDINGS)
        return()
    endif()

    add_executable(${target_name})
    
    target_sources(${target_name} PRIVATE ${source})
    target_link_libraries(${target_name} PRIVATE atcg_lib)

	if(MSVC)
        get_target_property(TORCH_LIBRARY_LOCATION torch LOCATION)
        get_filename_component(TORCH_LIB_SEARCH_PATH ${TORCH_LIBRARY_LOCATION} DIRECTORY)
        file(GLOB TORCH_DLLS "${TORCH_LIB_SEARCH_PATH}/*.dll")
        add_custom_command(TARGET ${target_name}
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:${target_name}>)
		set_target_properties(${target_name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
	endif()
endfunction()

add_subdirectory(exercises)

# PYTHON BINDINGS

if(ATCG_PYTHON_BINDINGS)
    find_package(charonload)

    if(charonload_FOUND)
        charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)
        target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/pyatcg.cpp)
        target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE atcg_lib)
    endif()
endif()

# DOCS

if(ATCG_BUILD_DOCS)
    add_subdirectory(docs)
endif()