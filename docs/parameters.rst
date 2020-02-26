.. Parameter documentation master file.

Parameters
==========

This page contains descriptions of all parameters in GBDTMO.

Meta
----

- ``verbose``: default = True, type = bool
    - If True, print loss information every round. Otherwise, print nothing.

- ``seed``: default = 0, type = int.
    - Random seed. **No effect currently**.

- ``num_threads``: default = 2, type = int.
    - Number of threads for training.

- ``hist_cache``: default = 16, type = int.
    - Maximum number of histogram cache

- ``topk``: default = 0. 
    - Sparse factors for sparse split finding. 
    - If 0, non-sparse split finding is used.
  
- ``one_side``: default = True, type = bool. 
    - Algorithm type for sparse split finding. 
    - If True, the restricted one is used.
    - Only used when `topk` not equal to 0.

- ``max_bins``: default = 32, type = int.
    - Maximum number of bins for each input variable.

Tree
----

- ``max_depth``: default = 4, type = int.
    - Maximum depth of trees, at least 1.
  
- ``max_leaves``: default = 32, type = int.
    - Maximum leaves of each tree.

- ``min_samples``: default = 20, type = int. 
    - Minimum number of samples of each leaf.
    - Stop growth if current number of samples smaller than this value.

- ``early_stop``: default = 0, type = int.
    - Number of rounds for early stop. 
    - If 0, early stop is not used.

Learning
--------

- ``base_score``: default = 0.0, type = double.
    - Initial value of prediction.

- ``subsample``: default = 1.0, type = double. 
    - Column sample rate. **No effect currently**.
  
- ``lr``: default = 0.2, type = double.
    - Learning rate.
  
- ``reg_l1``: default = 0.0, type = double.
    - L1 regularization.
    - Not used for sparse split finding currently.
  
- ``reg_l2``: default = 1.0, type = double.
    - L2 regularization.
  
- ``gamma``: default = 1e-3, type = double.
    - Minimum objective gain to split.
  
- ``loss``:  default = 'mse', type = string.
    - **Must be binary coding**. For example, b'mse' in Python.
    - Must be one of 'mse' (mean square error), 'bce' (binary cross entropy), 'ce' (cross entropy), and 'ce_column' ( only for ``GBDTSingle``).
