.. Examples documentation master file.

Examples
========

Plotting tree
-------------

Suppose the model is dumped into ``gbdtmo.txt``, plot 5th tree by:
  
  >>> from gbdtmo import create_graph
  >>> from graphviz import Digraph
  >>> graph = create_graph("gbdtmo.txt", 5, [0, 3])
  >>> graph.render("tree_5", format='pdf')

Then ``tree_5.pdf`` will be generated.

Using GBDTMO
------------

First import ``gbdtmo``

  >>> from gbdtmo import GBDTMulti, load_lib
  
Load from ``gbdtmo.so``

  >>> LIB = load_lib("path to gbdtmo.so")
  
Build an instance of GBDTMO. Here the ``out_dim`` is set to 10 and MSE loss is used.

  >>> out_dim = 10
  >>> params = {"max_depth": 5, "lr": 0.1, 'loss': b"mse"}
  >>> booster = GBDTMulti(LIB, out_dim=out_dim, params=params)
  
Set the training and eval datasets.

  >>> x_train, y_train = np.random.rand(10000, out_dim), np.random.rand(10000)
  >>> x_valid, y_valid = np.random.rand(10000, out_dim), np.random.rand(10000)
  >>> booster.set_data((x_train, y_train), (x_valid, y_valid))
  
Training with 30 rounds and dump it into text file.

  >>> booster.train(30)
  >>> booster.dump("tree.txt")

Custom loss
-----------

We show how to train GBDTMO via custom loss. Here is an example of MSE.

::

  def MSE(x, y):
    g = x - y
    h = np.ones_like(x)
    return g, h

>>> g, h = MSE(booster.preds_train.copy(), booster.label.copy())
>>> booster._set_gh(g, h)
>>> booster.boost()

In this way, a new tree is constructed and the predictions are updated.
