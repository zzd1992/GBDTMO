import numpy as np
from numba import jit
from multiprocessing import Pool
from functools import partial


@jit(forceobj=True)
def construct_bin_column(x: np.array, max_bins: int) -> np.array:
    x, cnt = np.unique(x, return_counts=True)
    sum_cnt = np.sum(cnt)
    if len(x) == 1:
        return np.array([], 'float64')
    elif len(x) == 2:
        bins = (x[0]*cnt[0] + x[1]*cnt[1]) / sum_cnt
        return np.array([bins], 'float64')
    elif len(x) <= max_bins:
        bins = np.zeros(len(x)-1, 'float64')
        for i in range(len(x)-1):
            bins[i] = (x[i] + x[i+1]) / 2.0
        return bins
    elif len(x) > max_bins:
        cnt = np.cumsum(cnt)
        t, p = 0, len(x) / float(max_bins)
        bins = np.zeros(max_bins-1, 'float64')
        for i in range(len(x)):
            if cnt[i] >= p:
                bins[t] = x[i]
                t += 1
                p = cnt[i] + (sum_cnt - cnt[i]) / float(max_bins-t)
            if t == max_bins-1: 
                return bins
        return bins


def map_bin_column(x, bins):
    bins = np.insert(bins, 0, -np.inf)
    bins = np.insert(bins, len(bins), np.inf)

    return np.searchsorted(bins, x, side='left').astype('uint16') - 1


def _get_bins_maps(x_column: np.array, max_bins: int) -> tuple:
    bins = construct_bin_column(x_column, max_bins)
    maps = map_bin_column(x_column, bins)

    return (bins, maps)


def get_bins_maps(x: np.array, max_bins: int, threads: int =1) -> (list, np.array):
    out = []
    if threads==1:
        for i in range(x.shape[-1]):
            out.append(_get_bins_maps(x[:, i], max_bins))
    else:
        x = list(np.transpose(x))
        pool = Pool(threads)
        f = partial(_get_bins_maps, max_bins=max_bins)
        out = pool.map(f, x)
        pool.close()

    bins, maps = [], []
    while out:
        _bin, _map = out.pop(0)
        bins.append(_bin)
        maps.append(_map)
    return bins, np.stack(maps, axis=1)


if __name__ == '__main__':
    x = np.random.rand(10000, 10)
    bins, maps = get_bins_maps(x, 8, 2)
    bin = bins[0]
    print(bin)
