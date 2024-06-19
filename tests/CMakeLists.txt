if(ATCG_BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE source "**.h*" "**.c*")

    set(TARGET_NAME atcg_lib_test)

    ATCG_add_executable(${TARGET_NAME} target_name "${source}")

    target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)

    target_include_directories(atcg_lib_test PRIVATE external/googletest/googletest/include)

    include(GoogleTest)
    gtest_discover_tests(${TARGET_NAME})
endif()