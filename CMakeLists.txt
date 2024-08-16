cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/conan_toolchain.cmake)
project(basket_option_pricer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# file(GLOB_RECURSE SOURCES src *.cpp)
# file(GLOB_RECURSE SOURCES_TEST src_test *.cpp)
# message(FOO="${SOURCES}")

# add_compile_options(-Wall -Wextra -Wpedantic)
# add_compile_options(-fvisibility=hidden)

# bop objects
#
set(BOP_OBJS bop_objs)

add_library(
  ${BOP_OBJS}
  OBJECT
  ${CMAKE_SOURCE_DIR}/src/basket_option_pricer/basket_options.cpp
  ${CMAKE_SOURCE_DIR}/src/basket_option_pricer/basket_options_gen.cpp
  ${CMAKE_SOURCE_DIR}/src/basket_option_pricer/standard_normal.cpp
)

# shared libraries will need PIC. for later performance, we can use
# separate shared and static objects but will double compile time
set_property(TARGET ${BOP_OBJS} PROPERTY POSITION_INDEPENDENT_CODE 1)

target_link_libraries(
  ${BOP_OBJS}
  Eigen3::Eigen
  Ceres::ceres
  glog::glog
  pybind11::embed
)

target_include_directories(
  ${BOP_OBJS}
  PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# bop library static & shared
#
set(BOP_LIB_STATIC bop_lib_static)

add_library(
  ${BOP_LIB_STATIC}
  STATIC
  $<TARGET_OBJECTS:${BOP_OBJS}>
)
target_link_libraries(
  ${BOP_LIB_STATIC}
  Eigen3::Eigen
  Ceres::ceres
  glog::glog
  pybind11::embed
)

set(BOP_LIB_SHARED bop_lib_shared)

add_library(
  ${BOP_LIB_SHARED}
  SHARED
  $<TARGET_OBJECTS:${BOP_OBJS}>
)

add_subdirectory(src_test)

add_subdirectory(pybind)
