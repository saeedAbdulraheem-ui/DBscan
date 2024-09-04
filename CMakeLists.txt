cmake_minimum_required(VERSION 3.10)
project(DBSCAN)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)

# find required packages
find_package(OpenCV REQUIRED)

# include directories for the libraries
include_directories(${OpenCV_INCLUDE_DIRS})

# specify the path to nanoflann.hpp (replace with the actual path on your system)
include_directories(nanoflann.hpp)

# create a library from dbscan.h
add_library(DBSCAN_LIB dbscan.h)

# set the link language to C++
set_target_properties(DBSCAN_LIB PROPERTIES LINKER_LANGUAGE CXX)

# specify that the library uses these packages
target_link_libraries(DBSCAN_LIB ${OpenCV_LIBS})

# create an executable (replace main.cpp with your main file)
add_executable(DBSCAN main.cpp)

# link the executable with the library
target_link_libraries(DBSCAN DBSCAN_LIB)