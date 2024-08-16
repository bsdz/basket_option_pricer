find_package(GTest REQUIRED)

enable_testing()

# gtest executable
#
set(GTEST_BOP_EXE gtest_bop)

add_executable(
    ${GTEST_BOP_EXE}
    ${CMAKE_SOURCE_DIR}/src_test/basket_options/borovkova_1.cpp
)
target_link_libraries(
    ${GTEST_BOP_EXE}
    ${BOP_LIB_STATIC}
    GTest::gtest_main
)
target_include_directories(
    ${GTEST_BOP_EXE}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src_test
)

# test discovery
#
include(CTest)
include(GoogleTest)
gtest_discover_tests(${GTEST_BOP_EXE})
