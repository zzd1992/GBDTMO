# !/usr/bin/env bash
if [ ! -d "build" ]; then
    mkdir build
fi

cd src
g++ -std=c++11 -fopenmp -O3 -fPIC -shared \
api.cpp \
booster.cpp \
dataStruct.cpp \
io.cpp \
loss.cpp \
mathFunc.cpp \
tree.cpp \
-o ../build/gbdtmo.so
