#include "loss.h"

void mse_grad(Dataset &data, double *g, double *h, int n, int out_dim = 1) {
    auto preds = data.Preds;
    auto labels = data.Label_double;
    int N = n * out_dim;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        g[i] = preds[i] - labels[i];
    }
}

double mse_score(Dataset &data, int n, int out_dim = 1) {
    auto preds = data.Preds;
    auto labels = data.Label_double;
    int N = n * out_dim;
    double s = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:s)
    for (int i = 0; i < N; ++i) {
        s += Sqr(preds[i] - labels[i]);
    }
    return sqrt(s / N);
}

void bce_grad(Dataset &data, double *g, double *h, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int N = n * out_dim;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const double t = 1.0f / (1.0f + exp(-preds[i]));
        g[i] = t - labels[i];
        h[i] = t * (1 - t);
    }
}

double bce_score(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int N = n * out_dim;
    double score = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:score)
    for (int i = 0; i < N; ++i) {
        double t = log(1 + exp(-preds[i]));
        if (labels[i] == 1) { score += t; }
        else { score += t + preds[i]; }
    }
    return score / N;
}

void ce_grad_thread(Dataset &data, double *g, double *h, int n, int out_dim, int start) {
    auto preds = data.Preds + start * out_dim;
    auto labels = data.Label_int32 + start;
    vector<double> rec(out_dim);
    int i, j, idx = 0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < out_dim; ++j) { rec[j] = preds[idx + j]; }
        Softmax(rec);
        for (j = 0; j < out_dim; ++j) {
            g[idx + j] = rec[j];
            h[idx + j] = rec[j] * (1.0f - rec[j]);
        }
        g[idx + labels[i]] -= 1.0f;
        idx += out_dim;
    }
}

void ce_grad(Dataset &data, double *g, double *h, int n, int out_dim) {
    if (n < 32) {
        ce_grad_thread(data, g, h, n, out_dim, 0);
    } else {
#pragma omp parallel
        {
            int threads = omp_get_num_threads();
            int id = omp_get_thread_num();
            int start = n / threads * id;
            int sample;
            if (id == threads - 1) {
                sample = n - n / threads * id;
            } else {
                sample = n / threads;
            }
            ce_grad_thread(data, g + start * out_dim, h + start * out_dim, sample, out_dim, start);
        };
    }
}

double ce_score_thread(Dataset &data, int n, int out_dim, int start) {
    auto preds = data.Preds + start * out_dim;
    auto labels = data.Label_int32 + start;
    int i, j, idx = 0;
    double score_sum = 0.0f;
    vector<double> rec(out_dim);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < out_dim; ++j) { rec[j] = preds[idx + j]; }
        score_sum += Log_sum_exp(rec) - preds[idx + labels[i]];
        idx += out_dim;
    }
    return score_sum;
}

double ce_score(Dataset &data, int n, int out_dim) {
    if (n < 32) {
        return ce_score_thread(data, n, out_dim, 0) / n;
    } else {
        double s = 0.0f;
#pragma omp parallel reduction(+:s)
        {
            int threads = omp_get_num_threads();
            int id = omp_get_thread_num();
            int start = n / threads * id;
            int sample;
            if (id == threads - 1) {
                sample = n - n / threads * id;
            } else {
                sample = n / threads;
            }
            s += ce_score_thread(data, sample, out_dim, start);
        };
        return s / n;
    }
}

void ce_grad_column_thread(Dataset &data, double *g, double *h, int n, int out_dim, int threads, int thread) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int i, j;
    vector<int> idx(out_dim);
    vector<double> rec(out_dim);
    for (i = thread; i < n; i += threads) {
        idx[0] = i;
        rec[0] = preds[i];
        for (j = 1; j < out_dim; ++j) {
            idx[j] = idx[j - 1] + n;
            rec[j] = preds[idx[j]];
        }
        Softmax(rec);
        for (j = 0; j < out_dim; ++j) {
            g[idx[j]] = rec[j];
            h[idx[j]] = rec[j] * (1.0f - rec[j]);
        }
        g[idx[labels[i]]] -= 1.0f;
    }
}

void ce_grad_column(Dataset &data, double *g, double *h, int n, int out_dim) {
    if (n < 32) {
        ce_grad_column_thread(data, g, h, n, out_dim, 1, 0);
    } else {
#pragma omp parallel
        {
            int threads = omp_get_num_threads();
            int id = omp_get_thread_num();
            ce_grad_column_thread(data, g, h, n, out_dim, threads, id);
        };
    }
}

double ce_score_column_thread(Dataset &data, int n, int out_dim, int threads, int thread) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int i, j;
    vector<int> idx(out_dim);
    vector<double> rec(out_dim);
    double score_sum = 0.0f;
    for (i = thread; i < n; i += threads) {
        idx[0] = i;
        rec[0] = preds[i];
        for (j = 1; j < out_dim; ++j) {
            idx[j] = idx[j - 1] + n;
            rec[j] = preds[idx[j]];
        }
        score_sum += Log_sum_exp(rec) - preds[idx[labels[i]]];

    }
    return score_sum;
}

double ce_score_column(Dataset &data, int n, int out_dim) {
    if (n < 32) {
        return ce_score_column_thread(data, n, out_dim, 1, 0);
    } else {
        int threads, id;
        double s = 0.0f;
#pragma omp parallel reduction(+:s)
        {
            threads = omp_get_num_threads();
            id = omp_get_thread_num();
            s += ce_score_column_thread(data, n, out_dim, threads, id);
        };
        return s / n;
    }
}

double acc_multiclass(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < n; ++i) {
        int idx = i * out_dim;
        double score = preds[idx];
        int ind = 0;
        for (int j = 1; j < out_dim; j++) {
            ++idx;
            if (preds[idx] > score) {
                score = preds[idx];
                ind = j;
            }
        }
        if (ind == labels[i]) { ++acc; }
    }
    return static_cast<double> (acc) / n;
}

double acc_multiclass_column(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < n; ++i) {
        int idx = i;
        double score = preds[idx];
        int ind = 0;
        for (int j = 1; j < out_dim; j++) {
            idx += n;
            if (preds[idx] > score) {
                score = preds[idx];
                ind = j;
            }
        }
        if (ind == labels[i]) { ++acc; }
    }
    return static_cast<double> (acc) / n;
}

double acc_binary(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0, N = n * out_dim;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < N; ++i) {
        if (labels[i] == 1) {
            if (preds[i] >= 0.0f) { ++acc; }
        } else {
            if (preds[i] < 0.0f) { ++acc; }
        }
    }
    return static_cast<double> (acc) / N;
}


