cmake_minimum_required(VERSION 3.5)
project(HNSW)

set(CMAKE_CXX_STANDARD 26)

add_executable(HNSW main.cpp
        cmd_args.cpp
        cmd_args.h
        distance.h
        hnsw.h
)
