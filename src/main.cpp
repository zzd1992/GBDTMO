#include <iostream>
#include "booster.h"

using namespace std;

void test(int inp_dim, int out_dim, int N, const char *mode) {
    double data[N * inp_dim];
    double preds[N * out_dim];
    int32_t lables[N];
    uint16_t maps[N * inp_dim];

    double data_eval[N * inp_dim];
    double preds_eval[N * out_dim];
    int32_t lables_eval[N];
    uint16_t maps_eval[N * inp_dim];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < inp_dim; ++j) {
            double t = static_cast<double> (rand() % 1000) / 1000;
            data[inp_dim * i + j] = t;
            data_eval[inp_dim * i + j] = t;
            maps[j * N + i] = static_cast<uint16_t> (t * 20);
            maps_eval[j * N + i] = static_cast<uint16_t> (t * 20);
        }
    }

    int t_in = 0, t_out = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            preds[t_out + j] = 0.0f;
            preds_eval[t_out + j] = 0.0f;
        }
        if (data[t_in] + data[t_in + 1] > 1.0) {
            lables[i] = 1;
            lables_eval[i] = 1;
        } else {
            lables[i] = 0;
            lables_eval[i] = 0;
        }
        t_in += inp_dim;
        t_out += out_dim;
    }

    uint16_t bins[inp_dim];
    double values[inp_dim * 19];
    for (int i = 0; i < inp_dim; i++) {
        bins[i] = 19;
        for (int j = 0; j < 19; j++) {
            values[i * 19 + j] = (j + 1) / 20.0f;
        }
    }

    if (mode == "single") {
        BoosterSingle m = BoosterSingle(inp_dim, "ce_column", 3, 32, 0, 2, 2, 0.5, 0.0, 1.0, 1e-3, 0.0, 0, true, 16);
        m.set_data(maps, data, preds, N, true);
        m.set_data(maps_eval, data_eval, preds_eval, N, false);
        m.set_bin(bins, values);
        m.set_label(lables, true);
        m.set_label(lables_eval, false);
        m.train_multi(30, out_dim);
        m.dump("single.txt");
    }
    if (mode == "multi") {
        BoosterMulti m = BoosterMulti(inp_dim, out_dim, 0, "ce", 3, 32, 0, 2, 2, 0.5, 0.0, 1.0, 1e-3, 0.0, 0, true,
                                      true, 16);
        m.set_data(maps, data, preds, N, true);
        m.set_data(maps_eval, data_eval, preds_eval, N, false);
        m.set_bin(bins, values);
        m.set_label(lables, true);
        m.set_label(lables_eval, false);
        m.train(30);
        m.dump("multi.txt");
    }
    if (mode == "sparse") {
        BoosterMulti m = BoosterMulti(inp_dim, out_dim, out_dim / 2, "ce", 3, 32, 0, 2, 2, 0.5, 0.0, 1.0, 1e-3, 0.0, 0,
                                      true, true, 16);
        m.set_data(maps, data, preds, N, true);
        m.set_data(maps_eval, data_eval, preds_eval, N, false);
        m.set_bin(bins, values);
        m.set_label(lables, true);
        m.set_label(lables_eval, false);
        m.train(30);
        m.dump("sparse.txt");
    }
}

int main() {
    cout << "single:" << endl;
    test(5, 5, 1000, "single");
    cout << "multi:" << endl;
    test(5, 5, 1000, "multi");
    cout << "sparse:" << endl;
    test(5, 5, 1000, "sparse");
    return 0;
}

