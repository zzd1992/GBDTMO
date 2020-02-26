#ifndef MGBDT_LOSS_H
#define MGBDT_LOSS_H

#include <iostream>
#include <omp.h>
#include <exception>
#include <stdexcept>
#include <cstring>
#include "dataStruct.h"
#include "mathFunc.h"

//TODO: make it easier to implement new loss objective

// grad function: dataset, g, h, rows, columns
// score function: dataset, rows, columns
// function with "column": only used for multiple predictions one by one

typedef void (*func_grad)(Dataset &, double *, double *, int, int);

typedef double (*func_score)(Dataset &, int, int);

typedef double (*func_metric)(Dataset &, int, int);


void mse_grad(Dataset &, double *, double *, int, int);

double mse_score(Dataset &, int, int);

void bce_grad(Dataset &, double *, double *, int, int);

double bce_score(Dataset &, int, int);

void ce_grad_thread(Dataset &, double *, double *, int, int, int);

void ce_grad(Dataset &, double *, double *, int, int);

double ce_score_thread(Dataset &, int, int, int);

double ce_score(Dataset &, int, int);

void ce_grad_column_thread(Dataset &, double *, double *, int, int, int, int);

void ce_grad_column(Dataset &, double *, double *, int, int);

double ce_score_column_thread(Dataset &, int, int, int, int);

double ce_score_column(Dataset &, int, int);

double acc_binary(Dataset &, int, int);

double acc_multiclass(Dataset &, int, int);

double acc_multiclass_column(Dataset &, int, int);


struct Objective {
    bool constHessian = true;
    double hessian = 1.0f;
    double largerBetter = false;
    func_grad f_grad;
    func_score f_score;
    func_metric f_metric;

    Objective() {};

    Objective(const char *name) {
        if (strcmp(name, "mse") == 0) {
            f_grad = mse_grad;
            f_score = mse_score;
            f_metric = mse_score;
        } else if (strcmp(name, "ce") == 0) {
            constHessian = false;
            largerBetter = true;
            f_grad = ce_grad;
            f_score = ce_score;
            f_metric = acc_multiclass;
        } else if (strcmp(name, "ce_column") == 0) {
            constHessian = false;
            largerBetter = true;
            f_grad = ce_grad_column;
            f_score = ce_score_column;
            f_metric = acc_multiclass_column;
        } else if (strcmp(name, "bce") == 0) {
            constHessian = false;
            largerBetter = true;
            f_grad = bce_grad;
            f_score = bce_score;
            f_metric = acc_binary;
        } else {
            string s = "Objective type name must be in [mse, bce, ce, ce_column]";
            cout << s << endl;
            throw std::runtime_error(s);
        }
    }
};

#endif //MGBDT_LOSS_H
