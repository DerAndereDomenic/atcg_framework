# Define the target
charonload_add_torch_library(atcg_lib STATIC)

# Link against cuda if requested
if(ATCG_CUDA_BACKEND)
    set(ATCG_BACKEND "cuda")

    find_package(CUDAToolkit REQUIRED)
    find_package(OptiX REQUIRED)

    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")

    target_include_directories(atcg_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(atcg_lib PUBLIC ATCG_CUDA_BACKEND)
    target_link_libraries(atcg_lib PUBLIC CUDA::cudart_static)
    target_link_libraries(atcg_lib PUBLIC CUDA::nvjpeg)
else()
    set(ATCG_BACKEND "cpu")
endif()

if(ATCG_HEADLESS)
    target_compile_definitions(atcg_lib PUBLIC ATCG_HEADLESS)
    set(ATCG_CONTEXT "headless")
else()
    set(ATCG_CONTEXT "glfw")
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

set(atcg_source_files "${atcg_lib_files}" 
                      "${atcg_backend_files}" 
                      "${atcg_renderer_files}"
                      "${atcg_window_files}")

set(ptx_files "")
foreach(source_file IN LISTS atcg_source_files)
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
list(REMOVE_ITEM atcg_source_files ${ptx_files})

target_sources(atcg_lib PRIVATE ${atcg_source_files})

target_compile_definitions(atcg_lib PUBLIC _USE_MATH_DEFINES NO_WARN_MBCS_MFC_DEPRECATION _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
target_precompile_headers(atcg_lib PRIVATE "${CMAKE_SOURCE_DIR}/atcg_lib/include/atcgpch.h")

target_include_directories(atcg_lib PUBLIC "${CMAKE_SOURCE_DIR}/atcg_lib/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_BACKEND}/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/${ATCG_CONTEXT}/include"
                                           "${CMAKE_SOURCE_DIR}/atcg_lib/platform/opengl/include"
                                           "python"
)

target_link_libraries(atcg_lib PUBLIC ${ATCG_CONTEXT_LIBRARY}
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

# ptxmodules
if(ATCG_CUDA_BACKEND)
    target_include_directories(atcg_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
                                               "${OptiX_INCLUDE}")

    add_library(ptxmodules OBJECT ${ptx_files})
    set_target_properties(ptxmodules PROPERTIES
        CUDA_PTX_COMPILATION ON
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(ptxmodules PUBLIC ATCG_CUDA_BACKEND _USE_MATH_DEFINES)
    target_compile_options(ptxmodules PRIVATE -ptx)
    target_link_libraries(ptxmodules PRIVATE atcg_lib)

    set(ATCG_PTX_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    add_custom_target(ptxmodules-copy
        DEPENDS ptxmodules
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_OBJECTS:ptxmodules> ${ATCG_PTX_OUTPUT_DIRECTORY}
        COMMAND_EXPAND_LISTS
    )
else()
    foreach(SOURCE_FILE ${ptx_files})
        # Get the filename without the directory and extension
        get_filename_component(FILENAME_WE ${SOURCE_FILE} NAME_WE)
        set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CXX)

        # Create a target for each file
        add_library(${FILENAME_WE} SHARED ${SOURCE_FILE})
        target_link_libraries(${FILENAME_WE} PRIVATE atcg_lib)
        target_compile_definitions(${FILENAME_WE} PRIVATE _USE_MATH_DEFINES ATCG_RT_MODULE)

        set_target_properties(${FILENAME_WE} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin"
            OUTPUT_NAME "${FILENAME_WE}" 
            SUFFIX ".ptx"                
        )
    endforeach()
endif()