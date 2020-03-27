import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import *

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags='CONTIGUOUS')
array_1d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=1, flags='CONTIGUOUS')
array_2d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=2, flags='CONTIGUOUS')


def load_lib(path):
    lib = npct.load_library(path, '.')

    lib.SetData.argtypes = [c_void_p, array_2d_uint16, array_2d_double, array_2d_double, c_int, c_bool]
    lib.SetBin.argtypes = [c_void_p, array_1d_uint16, array_1d_double]
    lib.SetGH.argtypes = [c_void_p, array_2d_double, array_2d_double]

    lib.Boost.argtypes = [c_void_p]
    lib.Train.argtypes = [c_void_p, c_int]
    lib.Dump.argtypes = [c_void_p, c_char_p]
    lib.Load.argtypes = [c_void_p, c_char_p]

    lib.MultiNew.argtypes = [c_int, c_int, c_int, c_char_p,
                             c_int, c_int, c_int, c_int, c_int,
                             c_double, c_double, c_double, c_double, c_double, c_int,
                             c_bool, c_bool, c_int]

    lib.SingleNew.argtypes = [c_int, c_char_p,
                              c_int, c_int, c_int, c_int, c_int,
                              c_double, c_double, c_double, c_double, c_double, c_int,
                              c_bool, c_int]

    lib.TrainMulti.argtypes = [c_void_p, c_int, c_int]
    lib.PredictMulti.argtypes = [c_void_p, array_2d_double, array_1d_double, c_int, c_int, c_int]
    lib.Reset.argtypes = [c_void_p]

    return lib


def default_params():
    p = {'max_depth': 4,
         'max_leaves': 32,
         'max_bins': 32,
         'topk': 0,
         'seed': 0,
         'num_threads': 2,
         'min_samples': 20,
         'subsample': 1.0,
         'lr': 0.2,
         'base_score': 0.0,
         'reg_l1': 0.0,
         'reg_l2': 1.0,
         'gamma': 1e-3,
         'loss': b"mse",
         'early_stop': 0,
         'one_side': True,
         'verbose': True,
         'hist_cache': 16
         }

    return p
