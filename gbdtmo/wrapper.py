import os
import numpy as np
from time import process_time
from .gbdtmo import GBDTSingle, GBDTMulti
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error


class GBDTMO_M(BaseEstimator):
    def __init__(self,
                 max_depth=5,
                 learning_rate=0.1,
                 random_state=1,
                 num_boosters=30,
                 lib=None,
                 out_dim=5,
                 inp_dim=10,
                 loss=b"mse",
                 subsample=1.0,
                 ):

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.lib = lib
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.loss = loss
        self.num_boosters = num_boosters
        self.subsample = subsample

    def _fit(self, x_train, x_test, y_train, y_test):
        LIB = load_lib(self.lib)
        params = {"max_depth": self.max_depth,
                  "lr": self.learning_rate,
                  'loss': b"mse",
                  "seed": self.random_state,
                  "subsample": self.subsample}
        self.booster = GBDTMulti(LIB, out_dim=self.out_dim, params=params)
        self.booster.set_data((x_train, y_train), (x_test, y_test))
        return self.booster.train(self.num_boosters)

    def fit(self, X, y):
        # self._fit()
        return self.booster.train(self.num_boosters)

    def predict(self, X):
        return self.predict(X)


class GBDTMO_classifier(GBDTMO_M):

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)


class GBDTMO_regression(GBDTMO_M):

    def score(self, X, y):
        pred = self.predict(X)
        output_errors = np.average((y - pred) ** 2, axis=0)

        return np.sqrt(output_errors)
