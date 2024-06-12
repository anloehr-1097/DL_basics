cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DL_basics)
# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BOOST_ROOT "~/Code/Cpp/DL_basics/boost_lib/")
set(CMAKE_BUILD_TYPE Debug)

# add_compile_options(-fsanitize=address)
find_package(Boost REQUIRED)
# adding a required dependency
# find_package(Torch REQUIRED)

# add_executable(Autoencoders Autoencoders.cpp)
add_executable(tests_boost.out test_boost.cpp)
add_executable(main.out main.cpp autodiff.h autodiff.cpp)
add_executable(tests.out tests/tests.cpp autodiff.cpp)
# target_include_directories(DL_basics PRIVATE "~/Code/Cpp/DL_basics/boost_lib/boost/graph/")
# target_include_directories(main PRIVATE "~/Code/Cpp/DL_basics/boost_lib/boost/graph/")
target_include_directories(tests_boost.out PRIVATE ${BOOST_ROOT})

set_property(TARGET tests_boost.out PROPERTY CXX_STANDARD 17)
set_property(TARGET main.out PROPERTY CXX_STANDARD 17)
set_property(TARGET tests.out PROPERTY CXX_STANDARD 17)
