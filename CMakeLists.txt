cmake_minimum_required(VERSION 3.11)
project(drl_in_action)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

####################################
##             utils              ##
####################################
add_library(utils INTERFACE)
target_include_directories(utils INTERFACE include/utils)

####################################
##             spdlog             ##
####################################
include(FetchContent)
find_package(spdlog QUIET CONFIG)
if(NOT spdlog_FOUND)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.5.0)
  FetchContent_GetProperties(spdlog)
  if(NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
    add_subdirectory(${spdlog_SOURCE_DIR})
  endif()
endif()

####################################
##             grid_world         ##
####################################
FetchContent_Declare(
  libfort
  GIT_REPOSITORY https://github.com/seleznevae/libfort.git
  GIT_TAG v0.4.0)
FetchContent_GetProperties(libfort)
if(NOT libfort_POPULATED)
  FetchContent_Populate(libfort)
endif()
add_library(fort ${libfort_SOURCE_DIR}/lib/fort.c)
target_include_directories(fort PUBLIC ${libfort_SOURCE_DIR}/lib)

add_library(grid_world INTERFACE)
target_include_directories(grid_world INTERFACE include/grid_world)
target_link_libraries(grid_world INTERFACE fort utils)

####################################
##             torch             ##
####################################
set(libtorch_VERSION 1.4.0)
find_package(Torch ${libtorch_VERSION} EXACT QUIET CONFIG)
if(NOT Torch_FOUND)
  message(STATUS "libtorch ${libtorch_VERSION} - not found")
  message(STATUS "Fetching libtorch")
  include(FetchContent)
  FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-${libtorch_VERSION}.zip
    SOURCE_DIR libtorch)
  FetchContent_GetProperties(libtorch)
  if(NOT libtorch_POPULATED)
    unset(FETCHCONTENT_QUIET CACHE)
    FetchContent_Populate(libtorch)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
  endif()
  find_package(Torch ${libtorch_VERSION} EXACT CONFIG REQUIRED)
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

####################################
##             examples           ##
####################################
add_executable(grid_world_example src/grid_world_example.cpp)
target_include_directories(grid_world_example PRIVATE include)
target_link_libraries(grid_world_example grid_world)

add_executable(dq_learning src/dq_learning.cpp)
target_link_libraries(dq_learning ${TORCH_LIBRARIES} grid_world spdlog::spdlog)
