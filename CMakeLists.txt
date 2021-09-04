cmake_minimum_required(VERSION 3.7)
set(CMAKE_CXX_STANDARD 17)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS} -std=c++17 -pthread")

project(CatDogClassifier)

find_package(Threads REQUIRED)
find_package(OpenCV 4.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

file(GLOB project_SRCS src/*.cpp)

add_executable(CatDogClassifier ${project_SRCS})
target_link_libraries(CatDogClassifier ${OpenCV_LIBRARIES} -lpthread)

