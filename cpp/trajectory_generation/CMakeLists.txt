# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(testing LANGUAGES CXX)   # Define project name and languages
# set(CMAKE_CXX_FLAGS "-O3")

find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)


add_executable(
	min_snap_generator
	src/main.cpp
	src/min_snap_traj.cpp
	src/polynomial.cpp
	)
target_link_libraries(min_snap_generator OsqpEigen::OsqpEigen osqp::osqp)

target_include_directories(min_snap_generator PRIVATE inc)