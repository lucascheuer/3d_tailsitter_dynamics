# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(3d_aircraft_sim LANGUAGES CXX)   # Define project name and languages
# set(CMAKE_CXX_FLAGS "-O3")
add_executable(
	3d_aircraft_sim
	src/main.cpp
	src/aircraft_dynamics.cpp
	src/rkf45.cpp
	src/aircraft_model.cpp
	src/helpers.cpp
	src/aircraft_controller.cpp
	src/aircraft_controller_params.cpp
	src/aircraft_controller_model.cpp
) # Create an executable target from main.cpp
target_include_directories(3d_aircraft_sim PRIVATE inc)