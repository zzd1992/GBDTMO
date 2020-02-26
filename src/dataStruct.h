#ifndef MGBDT_DATASTRUCTURE_H
#define MGBDT_DATASTRUCTURE_H

#include <math.h>
#include <queue>
#include <iostream>
#include <random>
#include <utility>
#include <omp.h>
#include <limits.h>

using namespace std;

struct Histogram {
    Histogram(int n = 1, int d = 1) : cnt(n, 0), g(n * d, 0), h(n * d, 0) {}

    vector<int> cnt;
    vector<double> g;
    vector<double> h;

    inline void operator-(const Histogram &x) {
        for (int i = 0; i < x.cnt.size(); i++) {
            cnt[i] -= x.cnt[i];
            g[i] -= x.g[i];
            h[i] -= x.h[i];
        }
    }
};

void histogram_single(vector<int32_t> &, Histogram &, uint16_t *, double *, double *);

void histogram_multi(vector<int32_t> &, Histogram &, uint16_t *, double *, double *, int);

struct Dataset {
    int num = 0;
    double *Features;
    uint16_t *Maps;
    double *Preds;
    std::vector<int32_t> Orders;
    std::vector<int32_t> LeafIndex;

    double *Label_double;
    int32_t *Label_int32;
};

struct HyperParameter {
    bool hist = true;
    bool one_side = true;
    bool best_first = true;
    bool verbose = true;
    const char *loss = "mse";
    int max_depth = 5;
    int max_leaves = int(pow(2, max_depth));
    int inp_dim;
    int out_dim = 1;
    int min_samples = 2;
    int num_threads = 1;
    double lr = 0.2;
    double reg_l1 = 0.0;
    double reg_l2 = 1.0;
    double gamma = 1e-3;
    double max_delta = 10.0;
    unsigned int seed = 0;
    double base_score = 0.0f;
    int topk = 0;
    int Max_caches = 16;
    int early_stop = 0;
};

// access to all top-k elements.
template<class T>
class TopkDeque {
private:
    int max_size;

public:
    deque<T> data;

    TopkDeque(int n = 8) : max_size(n) {};

    //binary search
    inline int search(T const &x) {
        int l = 0, r = data.size();
        while (l < r) {
            int m = (r + l - 1) / 2;
            if (x > data[m]) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }

    inline void push(T const &x) {
        if (data.size() == max_size) {
            if (x > data[max_size - 1]) {
                int i = search(x);
                data.insert(data.begin() + i, x);
                data.pop_back();
            }
        } else {
            int i = search(x);
            data.insert(data.begin() + i, x);
        }
    }

};


// access only the largest element.
template<class T>
class TopkPriority {
private:
    int k = 8;

public:
    priority_queue<T, vector<T>, greater<T>> data;

    TopkPriority(int n) : k(n) {};

    inline void push(T const &x) {
        data.push(x);
        if (data.size() > k) {
            data.pop();
        }
    }
};

class EarlyStoper {
private:
    int k = 10;
    bool larger_better = true;
    pair<double, int> info;

public:
    bool is_continue = true;

    EarlyStoper() {};

    EarlyStoper(int n, bool state) : k(n), larger_better(state) {
        if (state) {
            info = make_pair(-1e10, 0);
        } else {
            info = make_pair(1e10, 0);
        }
    };

    inline pair<double, int> best() {
        cout << "Best score " << get<0>(info) << " at round " << get<1>(info) << endl;
        return info;
    }

    inline void push(const pair<double, int> &x) {
        if (larger_better) {
            if (x > info) {
                info = x;
            }
        } else {
            if (x < info) {
                info = x;
            }
        }
        if (get<1>(x) < get<1>(info) + k) {
            is_continue = true;
        } else {
            is_continue = false;
        }
    }
};


#endif //MGBDT_DATASTRUCTURE_H
