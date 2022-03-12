cmake_minimum_required(VERSION 3.21)
project(Go)

set(CMAKE_CXX_STANDARD 14)

add_executable(Go c_go.cpp c_go.h)
