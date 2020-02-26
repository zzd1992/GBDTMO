.. Python API documentation master file.

Python API
==========

load_lib
--------

:load_lib(path):

  Parameters    
    - path (string): path of gbdtmo.so
  Return        
    Python warper of gbdtmo.so

create_graph
------------

:create_graph(file_name, tree_index=0, value_list=[]):

  This function generate a ``Digraph`` instance of ``graphviz``. You can render it by yourself.
  
  Parameters  
    - file_name (string): path of the dumped tree.
    - tree_index (int): the index (start from 0) of tree to be plotted.
    - value_list (list): list of index of output variables to be plotted. Only for **GBDTMO**. When set to ``[]``, all outputs variables will be considered.
  Return      
    a ``Digraph`` instance of a learned tree.
    
GBDTMulti
---------

:GBDTMulti(lib, out_dim=1, params={}):

  Create an instance of GBDTMO model.
  
  :__init__(lib, out_dim, params={}):
    
    Parameters  
      - lib: a Python warper of library by ``load_lib``.
      - out_dim(int): dimension of output.
      - params(dict): a set of parameters. If a parameter is not contained here, it is set to its default value.
 
  
  :set_data(train_set=(), eval_set=()):
      
    Set training and eval datasets. eval_set can be missing. Histograms will be constructed and predictions will be initialized. 
      
    Parameters
      - train_set(tuple): a tuple of numpy array (x_data, x_label). x_data must be `double` and 2D array. If you don't set label, x_label should be `None`. Otherwise, x_label must be `double` or `int32`.
      - eval_set(tuple, default=None): the same as train_set.
       
  :_set_gh(self, g, h):
      
    Set gradient and hessian for growth next tree. *Only used for user-defined loss*.
      
    Parameters  
      - g(numpy.array): gradient
      - h(numpy.array): hessian
                   
  :_set_label(x, is_train):
      
    Reset label. Sometimes it avoids the re-construction of histogram.
      
    Parameters
      - x(numpy.array): labels.
      - is_train(bool): if true, set labels for train_set else for eval_set.
    
  :boost():
    
    Growth a new tree after running ``_set_gh``.

  :train(num):
      
    training the model from scratch.
      
    Parameters
      - num(int): number of boost round.
      
  :dump(path):
    
    dump the model into a text file which has the following structure::
      
      Booster[i]:
        decision node M
        ...
        decision node 1
          leaf node 1
          ...
          leaf node N
      Booster[i+1]:
        ...

    For a decision node::

      node index, parent, left, right, split column, split value

    For a leaf node::

      leaf index, w_0, w_1, ..., w_n 
      
    Parameters  
      - path(string): **must be binary coding**. For example, b'tree.txt.

  :load(path):
    
    load the model from a text file.
      
    Parameters
      - path(string): **must be binary coding**. For example, b'tree.txt.
      
  :predict(x, num_trees=0):
    
    Parameters
      - x(numpy.array): input features
      - num_trees(int): number of trees used to compute the prediction. If 0, all trees will be used.
    Return
      prediction of x.
      
GBDTSingle
----------

GBDTSO is our own implementation of GBDT for single output. It is used to compare the training speed and accuracy with GBDTMO.

:GBDTSingle(lib, out_dim, params={}):

  Create an instance of GBDTSO model. Most of method is shared with GBDTSO. Here we only list the specific methods of GBDTSO.
                  
  :train_multi(num):
      
    training the model from scratch.
      
    Parameters
      - num(int): number of boost round. In each round, ``out_dim`` of trees will be constructed. They correspond to output variables in order.
      
  :reset():
    
    clear the learned trees and re-initialize the predictions to ``base_score``.
