cmake_minimum_required(VERSION "3.18")

project(ATCG_FRAMEWORK LANGUAGES CXX C VERSION 1.0)

option(ATCG_CUDA_BACKEND "Compile cuda backend" OFF)
option(ATCG_PYTHON_BINDINGS "Compile python bindings" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

add_subdirectory(external/glfw)
add_subdirectory(external/openmesh)

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

set(PYBIND11_PYTHON_VERSION 3)
set(PYBIND11_FINDPYTHON ON)
add_subdirectory(external/pybind11)

add_library(glad STATIC "external/glad/src/glad.c")
add_library(imgui STATIC "external/imgui/imgui.cpp"
                         "external/imgui/imgui_demo.cpp"
                         "external/imgui/imgui_draw.cpp"
                         "external/imgui/imgui_tables.cpp"
                         "external/imgui/imgui_widgets.cpp"
                         "external/imgui/backends/imgui_impl_glfw.cpp"
                         "external/imgui/backends/imgui_impl_opengl3.cpp")

if(ATCG_CUDA_BACKEND)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
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

add_library(atcg_lib STATIC "${atcg_lib_files}" "${atcg_backend_files}" "${atcg_renderer_files}") 

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES OM_STATIC_BUILD)
target_compile_definitions(OpenMeshCore PUBLIC OM_STATIC_BUILD)
target_compile_definitions(OpenMeshTools PUBLIC OM_STATIC_BUILD)

if(ATCG_CUDA_BACKEND)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(atcg_lib PUBLIC ATCG_CUDA_BACKEND)
    string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")
endif()

include_directories(atcg_lib/include
                    atcg_lib/platform/${BACKEND}/include
                    atcg_lib/platform/opengl/include
                    external/glad/include
                    external/glfw/include
                    external/imgui
                    external/eigen
                    external/glm
                    external/openmesh/src
                    external/nanoflann/include
                    external/pybind11/include
                    external/spdlog/include
                    external/imguizmo)

function(ATCG_add_executable target_name_base target_name_var source)
    set( target_name ${target_name_base} )
    set( ${target_name_var} ${target_name} PARENT_SCOPE )

    add_executable(${target_name}
        ${source}
        )
    
    target_link_libraries( ${target_name}
        atcg_lib
        glfw
        imgui
        imguizmo
        glad
        OpenGL::GL
        OpenMeshCore
        OpenMeshTools
        ${CUDA_LIBRARIES}
        #stdc++
    )

	if(MSVC)
		set_target_properties(${target_name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
	endif()
    target_compile_definitions(${target_name} PUBLIC _USE_MATH_DEFINES OM_STATIC_BUILD)
endfunction()

add_subdirectory(exercises)

# PYTHON BINDINGS

if(ATCG_PYTHON_BINDINGS)
    pybind11_add_module(pyatcg python/pyatcg.cpp)
    target_compile_definitions(pyatcg PUBLIC _USE_MATH_DEFINES OM_STATIC_BUILD)
    target_link_libraries( pyatcg PRIVATE
        glad
        atcg_lib
        glfw
        imgui
        imguizmo
        OpenGL::GL
        OpenMeshCore
        OpenMeshTools
        ${CUDA_LIBRARIES}
        #stdc++
    )
endif()