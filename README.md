# GBDT-MO: Gradient Boosted Decision Tree for Multiple Outputs
![](figs/tree_example.png)

## Introduction
Gradient boosted decision tree (GBDT) is a popular machine learning algorithm. Current open-sourced GBDT implementations are mainly designed for single output. When there are multiple outputs, they build multiple trees each of which corresponds to an output variable. Such a strategy ignores the correlations between output variables which leads to **worse generalization ability and tree redundancy**.

To address this problem, we propose a general method to learn GBDT for multiple outputs (GBDT-MO). Each leaf of GBDT-MO constructs the predictions of all variables or a subset of automatically selected variables. Experiments show that GBDT-MO has **better generalization ability and faster training speed** than GBDT for single output. 

## Implementations
We implement GBDT-MO from scratch by C++. And we provide a Python interface. Our implementations are similar to LightGBM except the learning mechanisms designed for multiple outputs. Some advanced features are not included, such as GPU training and distributed training. This project is only tested on Linux and used for academic explorations currently.

## Installation
First, clone this project and compile the C++ source code:
```
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

## Results
We show test performance on six real-world datasets. We also show training time averaged by rounds in log scale. Here, GBDT-SO is our own implementation of GBDT for single output.
![](figs/time_all.png)

|  Dataset | MNIST              | Yeast              | Caltech101         | NUS\-WIDE          | MNIST\-inpaining     | Student\-por         |
|:----------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:----------------------:|:----------------------:|
|  Metric    | accuracy           | accuracy           | accuracy           | top\-1 accuracy    | RMSE                 | RMSE                 |
| XGBoost                | 97\.86             | **62\.94** | 56\.52             | 43\.72             | 0\.26088             | 0\.24623             |
| LightGBM              | 98\.03             | 61\.97             | 55\.94             | 43\.99             | 0\.26090             | 0\.24466             |
| GBDT\-sparse       | 96\.41             | 62\.83             | 43\.93             | 44\.05             | \-                   | \-                   |
| GBDT\-SO              | 98\.08             | 61\.97             | 56\.62             | 44\.10             | 0\.26157             | 0\.24408             |
| GBDT\-MO            | **98\.30** | 62\.29             | **57\.49** | **44\.21** | **0\.26025** | **0\.24392** |


## Links

* For algorithm and experiment details, please refer our [preprint paper](https://arxiv.org/abs/1909.04373).
* For examples or reproducing the results, please refer [GBDTMO-EX](https://github.com/zzd1992/GBDTMO-EX).
* For instructions, please refer our [documentation](https://gbdtmo.readthedocs.io).

