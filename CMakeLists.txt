cmake_minimum_required(VERSION 3.27)

project(ATCG_FRAMEWORK LANGUAGES CXX C VERSION 0.1.1)

#
# SETUP
#

set(ATCG_PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(PREPEND CMAKE_MODULE_PATH "${ATCG_PROJECT_ROOT_DIR}/cmake")

include(ATCGFunctions)

# Configure C++
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ATCG_PROJECT_ROOT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ATCG_PROJECT_ROOT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ATCG_PROJECT_ROOT_DIR}/bin)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Manage options
option(ATCG_CUDA_BACKEND "Compile cuda backend" ON)
option(ATCG_PYTHON_MODULE "Build atcg_lib as python binding" OFF)
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
    add_subdirectory(external/glfw)
endif()

# Check for CUDA
if(ATCG_CUDA_BACKEND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")
endif()

#
# THIRD PARTY LIBRATIES
#

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# OpenMesh
add_subdirectory(external/openmesh)
target_compile_definitions(OpenMeshCore PUBLIC OM_STATIC_BUILD)
target_compile_definitions(OpenMeshTools PUBLIC OM_STATIC_BUILD)

# stbimage
add_subdirectory(external/stbimage)
target_include_directories(stb_image PUBLIC "external/stbimage")

# charonload
add_subdirectory(external/charonload)
add_library(torch_python_interface INTERFACE)
target_link_libraries(torch_python_interface INTERFACE charonload::torch_python)

# glad
add_library(glad STATIC)
target_sources(glad PRIVATE "external/glad/src/glad.c")
target_include_directories(glad PUBLIC "external/glad/include")

if(NOT ATCG_HEADLESS)
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
endif()

# tinyobjloader
add_library(tinyobjloader STATIC "external/tinyobjloader/tiny_obj_loader.cc")
target_include_directories(tinyobjloader PUBLIC "external/tinyobjloader")

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
set(MASTER_PROJECT_HAS_TARGET_UNINSTALL ON CACHE BOOL "" FORCE) # Make sure nanoflann does not try to define uninstall target
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

#sfml
set(SFML_BUILD_WINDOW OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_GRAPHICS OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
add_subdirectory(external/sfml)

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