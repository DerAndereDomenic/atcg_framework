function(ATCG_add_executable target_name_base target_name_var source)
    set(target_name ${target_name_base})
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    add_executable(${target_name})
    
    target_sources(${target_name} PRIVATE ${source})
    target_link_libraries(${target_name} PRIVATE atcg_lib)
    target_compile_definitions(${target_name} PRIVATE ATCG_TARGET_DIR="${ATCG_PROJECT_ROOT_DIR}/src/${target_name}")

	if(MSVC)
		set_target_properties(${target_name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${ATCG_PROJECT_ROOT_DIR}")
	endif()
endfunction()

function(ATCG_add_library target_name_base target_name_var source)
    set(target_name ${target_name_base})
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    add_library(${target_name} STATIC)
    
    target_sources(${target_name} PRIVATE ${source})
    target_link_libraries(${target_name} PRIVATE atcg_lib)
    target_compile_definitions(${target_name} PRIVATE ATCG_TARGET_DIR="${ATCG_PROJECT_ROOT_DIR}/src/${target_name}")

	if(MSVC)
		set_target_properties(${target_name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${ATCG_PROJECT_ROOT_DIR}")
	endif()
endfunction()