#ifndef MGBDT_TREE_H
#define MGBDT_TREE_H

#include "mathFunc.h"
#include "dataStruct.h"
#include <vector>
#include <map>
#include <iomanip>
#include <omp.h>

struct SplitInfo {
    double gain = -1e8;
    int column = -1;
    int bin = -1;
    double threshold = 0.0f;

    inline void reset() { gain = -1e8, column = -1; }

    inline void update(double gain_, int column_, int bin_, double threshold_) {
        gain = gain_;
        column = column_;
        bin = bin_;
        threshold = threshold_;
    }
};

struct NonLeafNode {
    int parent = 0, left = 0, right = 0;
    int column = -1;
    int bin = 0;
    double threshold = 0.0f;

    NonLeafNode() {};

    NonLeafNode(int parent_, int column_, int bin_, double threshold_) :
            parent(parent_), column(column_), bin(bin_), threshold(threshold_) {};
};

struct LeafNode {
    LeafNode(int n = 1) : values(n, 0) {};
    int parent;
    vector<double> values;

    inline void Update(int parent_, double value_) {
        parent = parent_;
        values[0] = value_;
    }

    inline void Update(int parent_, vector<double> &values_) {
        parent = parent_;
        values.assign(values_.begin(), values_.end());
    }

    inline void Update(int parent_, vector<pair<double, int>> &values_) {
        parent = parent_;
        for (auto &it : values_) {
            values[it.second] = it.first;
        }
    }
};


struct Tree {
    Tree(bool is_sparse = false) : sparse(is_sparse) {};
    bool sparse;
    int leaf_num = 0, nonleaf_num = 0;
    map<int, LeafNode> leaf;
    map<int, NonLeafNode> nonleaf;

    inline void clear() {
        leaf_num = 0;
        nonleaf_num = 0;
        leaf.clear();
        nonleaf.clear();
    }

    inline void add_leaf(const LeafNode &node, bool left) {
        ++leaf_num;
        leaf.emplace(leaf_num, node);
        if (left) {
            nonleaf[node.parent].left = leaf_num;
        } else {
            nonleaf[node.parent].right = leaf_num;
        }
    }

    inline void add_nonleaf(const NonLeafNode &node, bool left) {
        --nonleaf_num;
        nonleaf.emplace(nonleaf_num, node);
        if (left) {
            nonleaf[node.parent].left = nonleaf_num;
        } else {
            nonleaf[node.parent].right = nonleaf_num;
        }
    }

    inline void shrinkage(double lr) {
#pragma omp parallel for schedule(static) if (leaf.size() >= 256)
        for (int i = 1; i < leaf.size() + 1; ++i) {
            for (auto &p : leaf[i].values) { p *= lr; }
        }
    }

    // predict by original features
    // predict for each group (used for multi-core computation)
    void pred_value_single_(double *, double *, HyperParameter &, int);

    void pred_value_multi_(double *, double *, HyperParameter &, int);

    // predict all groups
    void pred_value_single(double *, double *, HyperParameter &, int);

    void pred_value_multi(double *, double *, HyperParameter &, int);

    // predict by bin maps
    void pred_value_single(uint16_t *, double *, HyperParameter &, int);

    void pred_value_multi(uint16_t *, double *, HyperParameter &, int);

};

#endif //MGBDT_TREE_H
