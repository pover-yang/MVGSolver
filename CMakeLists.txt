cmake_minimum_required(VERSION 3.10)
project(MVGSolver)

set(CMAKE_CXX_STANDARD 14)

# OpenCV
find_package(OpenCV)
# Eigen
find_package(Eigen3)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

file(GLOB SRC_FILES "*.cpp")

add_library(MVGSolver ${SRC_FILES})
