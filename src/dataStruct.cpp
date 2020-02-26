#include "dataStruct.h"

void histogram_single(vector<int32_t> &order, Histogram &Hist, uint16_t *maps, double *G, double *H) {
    uint16_t bin;
    for (auto i : order) {
        bin = maps[i];
        ++Hist.cnt[bin];
        Hist.g[bin] += G[i];
        Hist.h[bin] += H[i];
    }
    // integration
    for (int i = 1; i < Hist.cnt.size(); ++i) {
        Hist.cnt[i] += Hist.cnt[i - 1];
        Hist.g[i] += Hist.g[i - 1];
        Hist.h[i] += Hist.h[i - 1];
    }
}

void histogram_multi(vector<int32_t> &order, Histogram &Hist, uint16_t *maps, double *G, double *H, int out_dim) {
    int j, ind, bin;
    for (auto i : order) {
        bin = maps[i] * out_dim;
        ind = i * out_dim;
        ++Hist.cnt[maps[i]];
        for (j = 0; j < out_dim; ++j) {
            Hist.g[bin] += G[ind];
            Hist.h[bin++] += H[ind++];
        }
    }
    // integration
    ind = 0;
    for (int i = 1; i < Hist.cnt.size(); ++i) {
        Hist.cnt[i] += Hist.cnt[i - 1];
        for (j = 0; j < out_dim; ++j) {
            Hist.g[ind + out_dim] += Hist.g[ind];
            Hist.h[ind + out_dim] += Hist.h[ind];
            ++ind;
        }
    }
}