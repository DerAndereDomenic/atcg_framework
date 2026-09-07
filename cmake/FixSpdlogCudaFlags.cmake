# cmake/FixSpdlogCudaFlags.cmake
#
# spdlog unconditionally adds a bare "/Zc:__cplusplus" to its interface
# compile options on MSVC, without scoping it to CXX. This breaks nvcc,
# which chokes on the unwrapped /Zc:__cplusplus flag (see nvcc "single
# input file" error). This patches the *targets* after they're created,
# without modifying the submodule itself.
function(fix_spdlog_cuda_flags)
    if(NOT MSVC)
        return()
    endif()

    foreach(_tgt IN ITEMS spdlog spdlog_header_only)
        if(NOT TARGET ${_tgt})
            continue()
        endif()

        get_target_property(_opts ${_tgt} INTERFACE_COMPILE_OPTIONS)
        if(NOT _opts)
            continue()
        endif()

        list(FIND _opts "/Zc:__cplusplus" _idx)
        if(_idx EQUAL -1)
            message(STATUS "fix_spdlog_cuda_flags: ${_tgt} does not contain a bare /Zc:__cplusplus, skipping (spdlog version may have changed)")
            continue()
        endif()

        list(REMOVE_ITEM _opts "/Zc:__cplusplus")
        set_target_properties(${_tgt} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_opts}")
        target_compile_options(${_tgt} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
        message(STATUS "fix_spdlog_cuda_flags: patched ${_tgt} to scope /Zc:__cplusplus to CXX only")
    endforeach()
endfunction()