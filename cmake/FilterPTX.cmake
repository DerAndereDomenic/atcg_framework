function(filter_ptx_sources source_files output_var)

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

    # Return the list via the output variable
    set(${output_var} ${ptx_files} PARENT_SCOPE)
endfunction()

function(add_ptx_module target_prefix ptx_files)
    if(ptx_files)
        set(ptx_target "${target_prefix}_ptxmodules")

        add_library(${ptx_target} OBJECT ${ptx_files})
        set_target_properties(${ptx_target} PROPERTIES
            CUDA_PTX_COMPILATION ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
        )
        target_compile_definitions(${ptx_target} PUBLIC ATCG_CUDA_BACKEND _USE_MATH_DEFINES)
        target_compile_options(${ptx_target} PRIVATE -ptx)
        target_link_libraries(${ptx_target} PRIVATE ${target_prefix}_lib)

        set(ptx_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        add_custom_target(${ptx_target}-copy ALL
            DEPENDS ${ptx_target}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_OBJECTS:${ptx_target}> ${ptx_output_dir}
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()