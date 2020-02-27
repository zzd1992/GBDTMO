cmake_minimum_required(VERSION 3.13)
project(GBDTMO)

set(CMAKE_CXX_STANDARD 11)
set(OPT "-fopenmp -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT}")

add_library(GBDT SHARED
        dataStruct.h dataStruct.cpp
        mathFunc.h mathFunc.cpp
        tree.h tree.cpp
        loss.h loss.cpp
        booster.h booster.cpp
        io.h io.cpp
        api.cpp)

add_executable(Test main.cpp)
