import argparse
import numpy as np
from gbdtmo import GBDTMulti, load_lib

parser = argparse.ArgumentParser()
parser.add_argument("-lr", default=0.2, type=float)
parser.add_argument("-depth", default=5, type=int)
args = parser.parse_args()

LIB = load_lib("../build/gbdtmo.so")


def regression():
    inp_dim, out_dim = 10, 5
    params = {"max_depth": args.depth, "lr": args.lr, 'loss': b"mse"}
    booster = GBDTMulti(LIB, out_dim=out_dim, params=params)
    x_train, y_train = np.random.rand(10000, inp_dim), np.random.rand(10000, out_dim)
    x_valid, y_valid = np.random.rand(10000, inp_dim), np.random.rand(10000, out_dim)
    booster.set_data((x_train, y_train), (x_valid, y_valid))
    booster.train(20)
    booster.dump(b"regression.txt")


def classification():
    inp_dim, out_dim = 10, 5
    params = {"max_depth": args.depth, "lr": args.lr, 'loss': b"ce"}
    booster = GBDTMulti(LIB, out_dim=out_dim, params=params)
    x_train = np.random.rand(10000, inp_dim)
    y_train = np.random.randint(0, out_dim, size=(10000, )).astype("int32")
    x_valid = np.random.rand(10000, inp_dim)
    y_valid = np.random.randint(0, out_dim, size=(10000, )).astype("int32")
    booster.set_data((x_train, y_train), (x_valid, y_valid))
    booster.train(20)
    booster.dump(b"classification.txt")


if __name__ == '__main__':
    regression()
    # classification()
