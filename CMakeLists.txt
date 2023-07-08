cmake_minimum_required(VERSION 3.0)
project(optimizer)

find_package( Eigen3 3.3 REQUIRED )
find_package(GTest REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

add_definitions( -DSRCDIR=\"${CMAKE_SOURCE_DIR}/\")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Werror=reorder")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

include_directories( ${CMAKE_SOURCE_DIR} , 
                     ${EIGEN3_INCLUDE_DIR} ,
                     ${GTEST_INCLUDE_DIRS} )

add_executable( ex_var1d_func1d
  examples/var1d_func1d.cpp
  src/Function.hpp
  src/Optimizer.cpp
  src/FactoryOptimizer.cpp
  src/Data.cpp
  src/SimAnnParams.hpp
  src/PartSwarmParams.hpp
)

add_executable( ex_var2d_func1d
  examples/var2d_func1d.cpp
  src/Function.hpp
  src/Optimizer.cpp
  src/FactoryOptimizer.cpp
  src/Data.cpp
  src/SimAnnParams.hpp
  src/PartSwarmParams.hpp
)

add_executable( ex_sim_ann
  examples/sim_ann.cpp
  src/Function.hpp
  src/Optimizer.cpp
  src/FactoryOptimizer.cpp
  src/Data.cpp
  src/SimAnnParams.hpp
  src/PartSwarmParams.hpp
)

add_executable( swarm
  examples/sim_ann.cpp
  src/Function.hpp
  src/Optimizer.cpp
  src/FactoryOptimizer.cpp
  src/Data.cpp
  src/SimAnnParams.hpp
  src/PartSwarmParams.hpp
)

add_executable( testall 
  src/Function.hpp
  src/Optimizer.cpp
  tests/tests.cpp
  tests/function_1d.cpp
  tests/function_2d.cpp
  src/FactoryOptimizer.cpp
  src/Data.cpp
)

target_link_libraries( testall
  Eigen3::Eigen
  pthread
  GTest::gtest 
  GTest::gtest_main
)

target_link_libraries( ex_var1d_func1d
  Eigen3::Eigen
)

target_link_libraries( ex_var2d_func1d
  Eigen3::Eigen
)

enable_testing()

set_target_properties( testall PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_test(
  NAME
  testall
  COMMAND
  testall
)