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