# GBDT-MO: Gradient Boosted Decision Tree for Multiple Outputs
![](figs/tree_example.png)
## Introduction
Gradient boosted decision tree (GBDT) is a popular machine learning algorithm. Current open-sourced GBDT implementations are mainly designed for single output. When there are multiple outputs, they build multiple trees each of which corresponds to an output variable. Such a strategy ignores the correlations between output variables which leads to **worse generalization ability and tree redundancy**.

To address this problem, we propose a general method to learn GBDT for multiple outputs (GBDT-MO). Each leaf of GBDT-MO constructs the predictions of all variables or a subset of automatically selected variables. This is achieved by considering the summation of objective gains over all output variables. Experiments show that GBDT-MO has **better generalization ability and faster training speed (for a single round)** than GBDT for single output. 

For algorithm and experiment details, please refer our [preprint paper](https://arxiv.org/abs/1909.04373). If you want to see examples of GBDT-MO or reproduce the experiments in our paper, please refer [GBDTMO-EX](https://github.com/zzd1992/GBDTMO-EX).

## Implementations
We implement GBDT-MO from scratch by C++. And we provide a Python interface. Our implementations are similar to LightGBM except the learning mechanisms designed for multiple outputs. Some advanced features are not included, such as GPU training and distributed training. This project is used for academic explorations currently.

## Results
We show test performance on six real-world datasets. We also show training speed in log scale.  Here, GBDT-SO is our own implementation of GBDT for single output. GBDT-MO has better performance and faster training speed.
![](figs/time_all.png)

|  Dataset | MNIST              | Yeast              | Caltech101         | NUS\-WIDE          | MNIST\-inpaining     | Student\-por         |
|:----------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:----------------------:|:----------------------:|
|  Metric    | accuracy           | accuracy           | accuracy           | top\-1 accuracy    | RMSE                 | RMSE                 |
| XGBoost                | 97\.86             | **62\.94** | 56\.52             | 43\.72             | 0\.26088             | 0\.24623             |
| LightGBM              | 98\.03             | 61\.97             | 55\.94             | 43\.99             | 0\.26090             | 0\.24466             |
| GBDT\-sparse       | 96\.41             | 62\.83             | 43\.93             | 44\.05             | \-                   | \-                   |
| GBDT\-SO              | 98\.08             | 61\.97             | 56\.62             | 44\.10             | 0\.26157             | 0\.24408             |
| GBDT\-MO            | **98\.30** | 62\.29             | **57\.49** | **44\.21** | **0\.26025** | **0\.24392** |

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
Our library is only tested on Linux.

## Python API
`gbdtmo` provides two classes:
```
GBDTMulti(lib, out_dim=1, params={})
GBDTSingle(lib, out_dim=1, params={})
　lib: class, a Python warper of the shared library (.so file)
　out_dim: int, dimension of output
　params: dict, hyper-parameters
```
which correspond to GBDT-MO and GBDT-SO resprectively.

Shared methods for those two classes:
```
set_data(train_set=None, eval_set=None):
  train_set: a tuple of features and labels, i.e. (x_train, y_train)
  eval_set: atuple of features and labels, i.e. (x_eval, y_eval)
  Features must be a 2D float64 numpy array. 
  For GBDTSingle, labels must be a 1D float64 or int32 numpy array. 
  For GBDTMulti, for multi-class classification, labels must be a 1D int32 numpy array. 
  Others, labels must be a 2D float64 or int32 numpy array.
```
```
train(num):
  num: number of boost rounds.
```
```
predict(self, x, num_trees=0):
  x: input features
  num_trees: number of trees used for prediction. If 0, all trees are used.
```
```
dump(path):
  path: text file to dump the model. 
  Must be binary coding, e.g. b'model.txt'
```
```
load(path):
  path: text file to load the model. 
  Must be binary coding, e.g. b'model.txt'
```
An extra method for `GBDTSingle`:
```
train_multi(num):
  num: number of boost rounds.
  Only used for multi-class classification.
```
## Parameters
- `max_depth`: default = 4.
- `max_leaves`: default = 32.
- `max_bins`: default = 32.
- `topk`: default = 0. Sparse factors for sparse split finding. If 0, non-sparse split finding is used.
- `seed`: default = 0. Random seed. **No effect currently**.
- `num_threads`: default = 2. Number of threads for training.
- `min_samples`: default = 20. Minimum samples of each leaf.
- `subsample`: defualt = 1.0. Column sample rate. **No effect currently**.
- `lr`: default = 0.2. Learning rate.
- `base_score`: default = 0.0. 
- `reg_l1`: default = 0.0.
- `reg_l2`: default = 1.0.
- `gamma`: default = 1e-3.
- `loss`:  default = b'mse'. Must be  binary coding.
           Must be one of `mse`(mean square error), `ce`(cross entropy), 
           `bce`(binary cross entropy) and `ce_column`(cross entropy for `GBDTSingle`).
- `early_stop`: default = 0. Early stop rounds. If 0, early stop is not used.
- `one_side`: default = True. Algorithms for sparsesplit finding. If True, restricted one is used.
- `verbose`: default = True.
- `hist_cache`: default = 16. Maximum number of histogram cache. When the number of cached histograms equals to `hist_cache`,
                tree growth policy is not the exact best first search.
