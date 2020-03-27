import numpy as np
import numpy.ctypeslib as npct
import ctypes
from .histogram import get_bins_maps
from .lib_utils import *


class BoostUtils:
    def __init__(self, lib):
        self.lib = lib
        self._boostnode = None

    def _set_gh(self, g, h):
        self.lib.SetGH(self._boostnode, g, h)

    def _set_bin(self, bins):
        num, value = [], []
        for i, _ in enumerate(bins):
            num.append(len(_))
        num = np.array(num, np.uint16)
        value = np.concatenate(bins, axis=0)
        self.lib.SetBin(self._boostnode, num, value)

    def _set_label(self, x: np.array, is_train: bool):
        if x.dtype == np.float64:
            if x.ndim == 1:
                self.lib.SetLabelDouble.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_bool]
            elif x.ndim == 2:
                self.lib.SetLabelDouble.argtypes = [ctypes.c_void_p, array_2d_double, ctypes.c_bool]
            else:
                assert False, "label must be 1D or 2D array"
            self.lib.SetLabelDouble(self._boostnode, x, is_train)
        elif x.dtype == np.int32:
            if x.ndim == 1:
                self.lib.SetLabelInt.argtypes = [ctypes.c_void_p, array_1d_int, ctypes.c_bool]
            elif x.ndim == 2:
                self.lib.SetLabelInt.argtypes = [ctypes.c_void_p, array_2d_int, ctypes.c_bool]
            else:
                assert False, "label must be 1D or 2D array"
            self.lib.SetLabelInt(self._boostnode, x, is_train)
        else:
            assert False, "dtype of label must be float64 or int32"

    def boost(self):
        self.lib.Boost(self._boostnode)

    def dump(self, path):
        self.lib.Dump(self._boostnode, path)

    def load(self, path):
        self.lib.Load(self._boostnode, path)

    def train(self, num):
        self.lib.Train(self._boostnode, num)

class GBDTSingle(BoostUtils):
    def __init__(self, lib, out_dim=1, params={}):
        super(BoostUtils, self).__init__()
        BoostUtils.__init__(self, lib)
        self.out_dim = out_dim
        self.params = default_params()
        self.params.update(params)
        self.__dict__.update(self.params)

    def set_booster(self, inp_dim):
        self._boostnode = self.lib.SingleNew(inp_dim,
                                             self.params['loss'],
                                             self.params['max_depth'],
                                             self.params['max_leaves'],
                                             self.params['seed'],
                                             self.params['min_samples'],
                                             self.params['num_threads'],
                                             self.params['lr'],
                                             self.params['reg_l1'],
                                             self.params['reg_l2'],
                                             self.params['gamma'],
                                             self.params['base_score'],
                                             self.params['early_stop'],
                                             self.params['verbose'],
                                             self.params['hist_cache'])

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):
        if train_set is not None:
            self.data, self.label = train_set
            self.set_booster(self.data.shape[-1])
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins, self.num_threads)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.full(len(self.data) * self.out_dim, self.base_score, dtype='float64')

            self.lib.SetData.argtypes = [ctypes.c_void_p, array_2d_uint16, array_2d_double,
                                         array_1d_double, ctypes.c_int, ctypes.c_bool]
            self.lib.SetData(self._boostnode, self.maps, self.data,
                             self.preds_train, len(self.data), True)
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = eval_set
            self.preds_eval = np.full(len(self.data_eval) * self.out_dim, self.base_score, dtype='float64')
            maps = np.zeros((1, 1), 'uint16')
            self.lib.SetData(self._boostnode, maps, self.data_eval,
                             self.preds_eval, len(self.data_eval), False)
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def train_multi(self, num):
        '''
        only used for multi-classification
        '''
        assert self.out_dim>1, "out_dim must bigger than 1"
        self.lib.TrainMulti(self._boostnode, num, self.out_dim)

    def predict(self, x, num_trees=0):
        preds = np.full(len(x) * self.out_dim, self.base_score, dtype='float64')

        if self.out_dim == 1:
            self.lib.Predict.argtypes = [ctypes.c_void_p, array_2d_double, array_1d_double,
                                    ctypes.c_int, ctypes.c_int]
            self.lib.Predict(self._boostnode, x, preds, len(x), num_trees)
            return preds
        else:
            self.lib.PredictMulti(self._boostnode, x, preds, len(x), self.out_dim, num_trees)
            preds = np.reshape(preds, (self.out_dim, len(x)))
            return np.transpose(preds)

    def reset(self):
        self.lib.Reset(self._boostnode)


class GBDTMulti(BoostUtils):
    def __init__(self, lib, out_dim=1, params={}):
        super(BoostUtils, self).__init__()
        BoostUtils.__init__(self, lib)
        self.out_dim = out_dim
        self.params = default_params()
        self.params.update(params)
        self.__dict__.update(self.params)		

    def set_booster(self, inp_dim, out_dim):
        self._boostnode = self.lib.MultiNew(inp_dim,
                                            self.out_dim,
                                            self.params['topk'],
                                            self.params['loss'],
                                            self.params['max_depth'],
                                            self.params['max_leaves'],
                                            self.params['seed'],
                                            self.params['min_samples'],
                                            self.params['num_threads'],
                                            self.params['lr'],
                                            self.params['reg_l1'],
                                            self.params['reg_l2'],
                                            self.params['gamma'],
                                            self.params['base_score'],
                                            self.params['early_stop'],
                                            self.params['one_side'],
                                            self.params['verbose'],
                                            self.params['hist_cache'])

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):
        if train_set is not None:
            self.data, self.label = train_set
            self.set_booster(self.data.shape[-1], self.out_dim)
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins, self.num_threads)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.full((len(self.data), self.out_dim), self.base_score, dtype='float64')
            self.lib.SetData.argtypes = [ctypes.c_void_p, array_2d_uint16, array_2d_double,
                                         array_2d_double, ctypes.c_int, ctypes.c_bool]
            self.lib.SetData(self._boostnode, self.maps, self.data,
                             self.preds_train, len(self.data), True)
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = eval_set
            self.preds_eval = np.full((len(self.data_eval), self.out_dim), self.base_score, dtype='float64')
            maps  = np.zeros((1, 1), 'uint16')
            self.lib.SetData(self._boostnode, maps, self.data_eval,
                             self.preds_eval, len(self.data_eval), False)
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def predict(self, x, num_trees=0):
        preds = np.full((len(x), self.out_dim), self.base_score, dtype='float64')
        self.lib.Predict.argtypes = [ctypes.c_void_p, array_2d_double, array_2d_double,
                                     ctypes.c_int, ctypes.c_int]
        self.lib.Predict(self._boostnode, x, preds, len(x), num_trees)
        return preds


