## About this repository
This repository forked from the [GBDTMO](https://github.com/zzd1992/GBDTMO). To address the main paper, please refer to [paper](https://arxiv.org/abs/1909.04373).
In the forked version, I modified some settings based on my experiments.

## Implementations
GBDT-MO implemented from scratch by C++ with a Python interface. The implementations are similar to LightGBM except the learning mechanisms designed for multiple outputs. Some advanced features are not included, such as GPU training and distributed training. This project is only tested on Linux and used for academic explorations currently.

## Installation
First, clone this project and compile the C++ source code:
```
dos2unix make.sh
bash make.sh
```
The shared library will be generated in:
```
build/gbdtmo.so
```
Then, install the Python package:
```
pip install .
```

## Latest modification
Recent modifications in the forked version are as follows;
<ul>
<li> Update the make.sh  </li>
<li> Added  wrapper </li>
</ul>

### make.sh
Now works for windows users.
Now it builds the gbdtmo.so for you, but you still need to import the library from another OS.

### Wrapper
Built-in estimators of the Sklearn. You may use this wrapper to implement the Sklearn features. 
