#ifndef MGBDT_MATH_H
#define MGBDT_MATH_H

#include <vector>
#include <math.h>
#include <unordered_set>

using namespace std;

template<typename T>
inline static T Sum(vector<T> x) {
    T s = 0;
    for (auto &i : x) { s += i; }
    return s;
}

template<typename T>
inline static T Sqr(T x) { return x * x; }

inline static double ThresholdL1(double w, double reg_l1) {
    if (w > +reg_l1) return w - reg_l1;
    if (w < -reg_l1) return w + reg_l1;
    return 0.0;
}

inline static double CalWeight(double g_sum, double h_sum, double reg_l1, double reg_l2) {
    if (reg_l1 == 0) { return -g_sum / (h_sum + reg_l2); }
    else { return -ThresholdL1(g_sum, reg_l1) / (h_sum + reg_l2); }
}

inline static void CalWeight(vector<double> &value, double *g_sum, double *h_sum, double reg_l1, double reg_l2) {
    if (reg_l1 == 0) {
        for (int i = 0; i < value.size(); ++i) { value[i] = -g_sum[i] / (h_sum[i] + reg_l2); }
    } else {
        for (int i = 0; i < value.size(); ++i) { value[i] = -ThresholdL1(g_sum[i], reg_l1) / (h_sum[i] + reg_l2); }
    }
}

inline static double CalScore(double g_sum, double h_sum, double reg_l1, double reg_l2) {
    if (reg_l1 == 0) { return Sqr(g_sum) / (h_sum + reg_l2); }
    return Sqr(ThresholdL1(g_sum, reg_l1)) / (h_sum + reg_l2);
}

inline static void CalScore(vector<double> &value, double *g_sum, double *h_sum, double reg_l1, double reg_l2) {
    if (reg_l1 == 0) {
        for (int i = 0; i < value.size(); ++i) { value[i] = Sqr(g_sum[i]) / (h_sum[i] + reg_l2); }
    } else {
        for (int i = 0; i < value.size(); ++i) { value[i] = Sqr(ThresholdL1(g_sum[i], reg_l1)) / (h_sum[i] + reg_l2); }
    }
}

inline static void Softmax(vector<double> &rec) {
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0f;
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] = exp(rec[i] - wmax);
        wsum += rec[i];
    }
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] /= wsum;
    }
}

inline static double Log_sum_exp(vector<double> &rec) {
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0f;
    for (size_t i = 0; i < rec.size(); ++i) {
        wsum += exp(rec[i] - wmax);
    }
    return log(wsum) + wmax;
}

#endif //MGBDT_MATH_H
