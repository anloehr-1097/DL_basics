cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DL_basics)
# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(BOOST_ROOT "~/Code/Cpp/DL_basics/boost_lib/")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
# add_compile_options(-fsanitize=address)
# find_package(Boost REQUIRED)
# adding a required dependency
# find_package(Torch REQUIRED)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# add_executable(Autoencoders Autoencoders.cpp)
# add_executable(tests_boost.out test_boost.cpp)

enable_testing()

add_executable(
  test_ops
  tests/test_ops.cpp
)
target_link_libraries(
  test_ops 
  GTest::gtest_main
)

add_executable(
  test_tensor
  tests/test_tensor.cpp
)
target_link_libraries(
  test_tensor 
  GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_ops)

add_executable(main.out main.cpp autodiff.h autodiff.cpp)
add_executable(tests.out tests/tests.cpp autodiff.cpp)
# target_include_directories(DL_basics PRIVATE "~/Code/Cpp/DL_basics/boost_lib/boost/graph/")
# target_include_directories(main PRIVATE "~/Code/Cpp/DL_basics/boost_lib/boost/graph/")
# target_include_directories(tests_boost.out PRIVATE ${BOOST_ROOT})

# set_property(TARGET tests_boost.out PROPERTY CXX_STANDARD 17)
set_property(TARGET main.out PROPERTY CXX_STANDARD 17)
set_property(TARGET tests.out PROPERTY CXX_STANDARD 17)
