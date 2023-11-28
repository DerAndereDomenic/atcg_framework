cmake_minimum_required(VERSION "3.24")

project(ATCG_FRAMEWORK LANGUAGES CXX C VERSION 1.0)

option(ATCG_CUDA_BACKEND "Compile cuda backend" OFF)
option(ATCG_PYTHON_BINDINGS "Compile python bindings" OFF)
option(ATCG_BUILD_DOCS "Build documentation" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

add_subdirectory(external/glfw)
add_subdirectory(external/openmesh)
add_subdirectory(external/stbimage)

set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Enable parse tools")
add_subdirectory(external/yaml-cpp)

add_library(imguizmo STATIC "external/imguizmo/GraphEditor.cpp"
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

add_subdirectory(external/charonload)

add_library(glad STATIC "external/glad/src/glad.c")
add_library(imgui STATIC "external/imgui/imgui.cpp"
                         "external/imgui/imgui_demo.cpp"
                         "external/imgui/imgui_draw.cpp"
                         "external/imgui/imgui_tables.cpp"
                         "external/imgui/imgui_widgets.cpp"
                         "external/imgui/backends/imgui_impl_glfw.cpp"
                         "external/imgui/backends/imgui_impl_opengl3.cpp")

add_library(implot STATIC "external/implot/implot.cpp"
                          "external/implot/implot_items.cpp")

add_library(tinyobjloader STATIC "external/tinyobjloader/tiny_obj_loader.cc")

if(ATCG_CUDA_BACKEND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    set(BACKEND "cuda")
    file (GLOB_RECURSE atcg_backend_files LIST_DIRECTORIES false "atcg_lib/platform/cuda/**.c*"
                                                            "atcg_lib/platform/cuda/**.h")
else()
    set(BACKEND "cpu")
    file (GLOB_RECURSE atcg_backend_files LIST_DIRECTORIES false "atcg_lib/platform/cpu/**.c*"
                                                            "atcg_lib/platform/cpu/**.h")
endif()

file (GLOB_RECURSE atcg_lib_files LIST_DIRECTORIES false "atcg_lib/src/**.c*"
                                                          "atcg_lib/include/**.h")

file (GLOB_RECURSE atcg_renderer_files LIST_DIRECTORIES false "atcg_lib/platform/opengl/**.c*"
                                                              "atcg_lib/platform/opengl/**.h")

charonload_add_torch_library(atcg_lib STATIC)
target_sources(atcg_lib PRIVATE "${atcg_lib_files}" "${atcg_backend_files}" "${atcg_renderer_files}")

if(NOT MSVC)
    add_definitions(-DFMT_USE_CONSTEXPR)
endif()

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES OM_STATIC_BUILD YAML_CPP_STATIC_DEFINE)
target_compile_definitions(OpenMeshCore PUBLIC OM_STATIC_BUILD)
target_compile_definitions(OpenMeshTools PUBLIC OM_STATIC_BUILD)
target_precompile_headers(atcg_lib PRIVATE atcg_lib/include/atcgpch.h)

if(ATCG_CUDA_BACKEND)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(atcg_lib PUBLIC ATCG_CUDA_BACKEND)
    string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")
    string(APPEND CUDA_LIBRARIES "cudart_static")
endif()

include_directories(atcg_lib/include
                    atcg_lib/platform/${BACKEND}/include
                    atcg_lib/platform/opengl/include
                    external/glad/include
                    external/glfw/include
                    external/imgui
                    external/implot
                    external/glm
                    external/openmesh/src
                    external/nanoflann/include
                    external/spdlog/include
                    external/imguizmo
                    external/nanort
                    external/entt/include
                    external/yaml-cpp/include
                    external/stbimage
                    external/portable-file-dialogs
                    external/tinyobjloader)

function(ATCG_add_executable target_name_base target_name_var source)
    set( target_name ${target_name_base} )
    set( ${target_name_var} ${target_name} PARENT_SCOPE )

    if(ATCG_PYTHON_BINDINGS)
        return()
    endif()

    add_executable(${target_name}
        ${source}
        )
    
    target_link_libraries( ${target_name}
        atcg_lib
        glfw
        imgui
        imguizmo
        implot
        glad
        OpenGL::GL
        OpenMeshCore
        OpenMeshTools
        yaml-cpp
        stb_image
        tinyobjloader
        ${CUDA_LIBRARIES}
    )

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
    target_compile_definitions(${target_name} PUBLIC _USE_MATH_DEFINES OM_STATIC_BUILD)
endfunction()

add_subdirectory(exercises)

# PYTHON BINDINGS

if(ATCG_PYTHON_BINDINGS)

    find_package(charonload)

    if(charonload_FOUND)
        charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)
        target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/pyatcg.cpp)
        target_compile_definitions(${TORCH_EXTENSION_NAME} PUBLIC _USE_MATH_DEFINES OM_STATIC_BUILD)
        target_link_libraries( ${TORCH_EXTENSION_NAME} PRIVATE
            atcg_lib
            glfw
            imgui
            imguizmo
            implot
            glad
            OpenGL::GL
            OpenMeshCore
            OpenMeshTools
            yaml-cpp
            stb_image
            tinyobjloader
            ${CUDA_LIBRARIES}
            #stdc++
        )
    endif()
endif()

# DOCS

if(ATCG_BUILD_DOCS)
    add_subdirectory(docs)
endif()