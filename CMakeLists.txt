cmake_minimum_required(VERSION 3.27)

project(ATCG_FRAMEWORK LANGUAGES CXX C VERSION 0.2.1)

#
# SETUP
#

set(ATCG_PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(PREPEND CMAKE_MODULE_PATH "${ATCG_PROJECT_ROOT_DIR}/cmake")

include(ATCGFunctions)

# Configure C++
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Manage options
option(ATCG_CUDA_BACKEND "Compile cuda backend" ON)
option(ATCG_PYTHON_MODULE "Build atcg_lib as python binding" OFF)
option(ATCG_OPTIX_PATHTRACING "Build the Optix Pathtracer" ON)
option(ATCG_BUILD_DOCS "Build documentation" OFF)
option(ATCG_BUILD_TESTS "Build tests" OFF)
option(ATCG_HEADLESS "Build in headless mode (cuda and linux only)" OFF)
option(ATCG_BUILD_EXAMPLES "Build atcg examples" ON)

# OpenMesh and OpenVR use Cmake < 3.10
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

if(ATCG_HEADLESS)
    if(NOT UNIX)
        message(FATAL_ERROR "Headless mode only supported on linux.")
    endif()

    if(NOT ATCG_CUDA_BACKEND)
        message(FATAL_ERROR "Headless mode only supported with CUDA backend.")
    endif()
else()
    # GLFW
    if(NOT TARGET glfw)
        add_subdirectory(external/glfw)
    endif()
endif()

# Check for CUDA
if(ATCG_CUDA_BACKEND AND NOT CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

if(ATCG_OPTIX_PATHTRACING AND CMAKE_CUDA_COMPILER)
    find_package(OptiX)
    if(OptiX_INCLUDE)
        message("FOUND OPTIX --- building with OptiX pathtracing")
        set(ATCG_ENABLE_OPTIX ON)
    else()
        message("Could not find OptiX --- building without pathtracing")
    endif()
endif()

#
# THIRD PARTY LIBRATIES
#

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# OpenMesh
if(NOT TARGET OpenMeshCore AND NOT TARGET OpenMeshTools)
    add_subdirectory(external/openmesh)
endif()
target_compile_definitions(OpenMeshCore PUBLIC OM_STATIC_BUILD)
target_compile_definitions(OpenMeshTools PUBLIC OM_STATIC_BUILD)

# stbimage
if(NOT TARGET stb_image)
    add_subdirectory(external/stbimage)
endif()
target_include_directories(stb_image PUBLIC "external/stbimage")

# charonload
add_subdirectory(external/charonload)
add_library(torch_python_interface INTERFACE)
target_link_libraries(torch_python_interface INTERFACE charonload::torch_python)

# glad
add_library(atcg_glad STATIC)
target_sources(atcg_glad PRIVATE "external/glad/src/glad.c")
target_include_directories(atcg_glad PUBLIC "external/glad/include")
add_library(atcg::glad ALIAS atcg_glad)

if(NOT ATCG_HEADLESS)
    # imgui
    add_library(atcg_imgui STATIC)
    target_sources(atcg_imgui PRIVATE "external/imgui/imgui.cpp"
                                "external/imgui/imgui_demo.cpp"
                                "external/imgui/imgui_draw.cpp"
                                "external/imgui/imgui_tables.cpp"
                                "external/imgui/imgui_widgets.cpp"
                                "external/imgui/backends/imgui_impl_glfw.cpp"
                                "external/imgui/backends/imgui_impl_opengl3.cpp")
    target_include_directories(atcg_imgui PUBLIC "external/imgui")
    target_link_libraries(atcg_imgui PRIVATE glfw)
    add_library(atcg::imgui ALIAS atcg_imgui)

    # implot
    add_library(atcg_implot STATIC)
    target_sources(atcg_implot PRIVATE "external/implot/implot.cpp"
                                "external/implot/implot_items.cpp")
    target_include_directories(atcg_implot PUBLIC "external/implot")
    target_link_libraries(atcg_implot PUBLIC atcg::imgui)
    add_library(atcg::implot ALIAS atcg_implot)

    # imguizmo
    add_library(atcg_imguizmo STATIC)
    target_sources(atcg_imguizmo PRIVATE "external/imguizmo/GraphEditor.cpp"
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
    target_include_directories(atcg_imguizmo PUBLIC "external/imguizmo")
    target_link_libraries(atcg_imguizmo PUBLIC atcg::imgui)
    add_library(atcg::imguizmo ALIAS atcg_imguizmo)
endif()

# tinyobjloader
add_library(atcg_tinyobjloader STATIC "external/tinyobjloader/tiny_obj_loader.cc")
target_include_directories(atcg_tinyobjloader PUBLIC "external/tinyobjloader")
add_library(atcg::tinyobjloader ALIAS atcg_tinyobjloader)

# entt
add_library(atcg_entt INTERFACE)
target_include_directories(atcg_entt INTERFACE "external/entt/include")
add_library(atcg::entt ALIAS atcg_entt)

# nanort
add_library(atcg_nanort INTERFACE)
target_include_directories(atcg_nanort INTERFACE "external/nanort")
add_library(atcg::nanort ALIAS atcg_nanort)

# portable-file-dialogs
add_library(atcg_portable_file_dialogs INTERFACE)
target_include_directories(atcg_portable_file_dialogs INTERFACE "external/portable-file-dialogs")
add_library(atcg::portable_file_dialogs ALIAS atcg_portable_file_dialogs)

# nanoflann
set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MASTER_PROJECT_HAS_TARGET_UNINSTALL ON CACHE BOOL "" FORCE) # Make sure nanoflann does not try to define uninstall target
set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(NOT TARGET nanoflann)
    add_subdirectory(external/nanoflann)
endif()

# glm
if(NOT TARGET glm)
    add_subdirectory(external/glm)
endif()

# spdlog
if(NOT TARGET spdlog_header_only)
    add_subdirectory(external/spdlog)
endif()
if(NOT MSVC)
    target_compile_definitions(spdlog_header_only INTERFACE FMT_USE_CONSTEXPR)
endif()

# openvr
set(BUILD_SHARED ON CACHE BOOL "Builds the library as shared library" FORCE)
if(NOT TARGET openvr_api OR NOT TARGET openvr_api64)
    add_subdirectory(external/openvr)
endif()
if(TARGET openvr_api)
    # Set properties on the real target before alias creation
    set_target_properties(openvr_api PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    add_library(atcg::openvr ALIAS openvr_api)
elseif(TARGET openvr_api64)
    # Set properties on the real target before alias creation
    set_target_properties(openvr_api64 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    add_library(atcg::openvr ALIAS openvr_api64)
endif()

# json
add_library(atcg_json INTERFACE)
target_include_directories(atcg_json INTERFACE "external/json")
add_library(atcg::json ALIAS atcg_json)

#sfml
set(SFML_BUILD_WINDOW OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_GRAPHICS OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
if(NOT TARGET sfml)
    add_subdirectory(external/sfml)
endif()

#
# ATCG_LIB
#

add_subdirectory(atcg_lib)

#
# EXECUTABLES
#

if(ATCG_BUILD_EXAMPLES)
    add_subdirectory(src)
endif()

#
# PYTHON BINDINGS
#

if(ATCG_PYTHON_MODULE)
    find_package(charonload)

    if(charonload_FOUND)
        charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)
        target_sources(${TORCH_EXTENSION_NAME} PRIVATE atcg_lib/platform/python/pyatcg.cpp)
        target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE atcg_lib)
        target_compile_definitions(${TORCH_EXTENSION_NAME} PUBLIC ATCG_PYTHON_MODULE)
        target_compile_definitions(atcg_lib PUBLIC TORCH_EXTENSION_NAME=${TORCH_EXTENSION_NAME})
    endif()
endif()

#
# DOCS
#

if(ATCG_BUILD_DOCS)
    add_subdirectory(docs)
endif()

#
# TESTS
#

if(ATCG_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(external/googletest)
    add_subdirectory(tests)
endif()