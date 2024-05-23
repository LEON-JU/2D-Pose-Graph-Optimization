cmake_minimum_required(VERSION 3.12)
project(trajectory_optimization)

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/local/include/eigen3")

# Find Eigen library
find_package(Eigen3 REQUIRED)

add_executable(trajectory_optimization main.cpp)

# Link Eigen library
target_link_libraries(trajectory_optimization Eigen3::Eigen)
