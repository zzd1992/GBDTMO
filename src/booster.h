#ifndef MGBDT_BOOSTER_H
#define MGBDT_BOOSTER_H

#include "tree.h"
#include "mathFunc.h"
#include "dataStruct.h"
#include "loss.h"
#include "io.h"
#include <algorithm>

struct CacheInfo {
    int node;
    int depth;
    SplitInfo split;
    vector<int32_t> order;
    vector<Histogram> hist;

    CacheInfo(int n, int d, SplitInfo &x, vector<int32_t> &y, vector<Histogram> &z) :
            node(n), depth(d), split(x), order(y), hist(z) {};

    bool operator>(const CacheInfo &x) const { return split.gain > x.split.gain; }
};


class BoosterUtils {
public:
    void set_bin(uint16_t *, double *);

    void set_gh(double *, double *);

    void set_data(uint16_t *, double *, double *, int, bool);

    void set_label(double *, bool);

    void set_label(int32_t *, bool);

    void rebuild_order(vector<int32_t> &, vector<int32_t> &, vector<int32_t> &, uint16_t *, uint16_t);

    void showloss(double, double, int);

    void showloss(double, int);

    void load(const char *path) { LoadTrees(trees, path); }

    void dump(const char *path) { DumpTrees(trees, path); }

    virtual void update() = 0;

    virtual void growth() = 0;

    virtual void train(int) = 0;

    virtual void predict(double *, double *, int, int) = 0;

protected:
    Tree tree;
    vector<Tree> trees;
    std::vector<uint16_t> bin_nums;
    std::vector<std::vector<double>> bin_values;
    SplitInfo meta;
    Dataset Train;
    Dataset Eval;
    double *G;
    double *H;
    HyperParameter hp;
    TopkDeque<CacheInfo> cache;
    Objective obj;
};


class BoosterSingle : public BoosterUtils {
public:
    BoosterSingle(int, const char *,
                  int, int, int, int, int,
                  double, double, double, double, double,
                  int, bool, int);

    void update() override;

    void growth() override;

    void train(int) override;

    void predict(double *, double *, int, int) override;

    void reset();

    void train_multi(int, int);

    void predict_multi(double *, double *, int, int, int);

private:
    double Score_sum, Opt;

    void get_score_opt(Histogram &, double &, double &);

    void hist_all(vector<int32_t> &, vector<Histogram> &);

    void boost_column(Histogram &, int);

    void boost_all(vector<Histogram> &);

    void build_tree_best();
};


class BoosterMulti : public BoosterUtils {
public:
    BoosterMulti(int, int, int, const char *,
                 int, int, int, int, int,
                 double, double, double, double, double,
                 int, bool, bool, int);

    void update() override;

    void growth() override;

    void train(int) override;

    void predict(double *, double *, int, int) override;

private:
    double Score_sum;
    vector<double> Score;
    vector<double> Opt;
    vector<pair<double, int>> OptPair;

    void get_score_opt(Histogram &, vector<double> &, vector<double> &, double &);

    void get_score_opt(Histogram &, vector<pair<double, int>> &, vector<double> &, double &);

    void hist_all(vector<int32_t> &, vector<Histogram> &);

    void boost_column_full(Histogram &, int);

    void boost_column_topk_two_side(Histogram &, int);

    void boost_column_topk_one_side(Histogram &, int);

    void boost_all(vector<Histogram> &);

    void build_tree_best();
};

#endif //MGBDT_BOOSTER_H