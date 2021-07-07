
//#pragma once
//这个代码包含所有基本的操作，包括一些内积操作和读文件函数

#ifndef SDCA_UTILS_H
#define SDCA_UTILS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdlib.h>//atoi
#include <math.h>//ceil,pow
#include "libsvm_data.h"

using namespace std;

double min(double a, double b);
double max(double a, double b);

double dot_dense(const vector<double>& x); //w.dot(w)
double dot_dense(const vector<double>& x, const vector<double>& y);//w1.dot(w2)
double norm_2_sparse(const Data& train_data, int k); //sqrt(Xk.dot(Xk))
double dot_sparse(const Data& train_data, int k, const vector<double>& w);//Xi.dot(w)

void normalize_data(Data& train_data);

inline double stringToNum(const string& str);// only used in readLibsvm()function

											 //读数据的时候就扩展特征维度,在每个sample的第一个位置添加
											 //另外，index中，最后一个维度，index[n_sample+1]=X.size(),为了方便编写循环
void read_libsvm(const string filename, Data &train_data);
void read_libsvm_raw(const string filename, Data &train_data);
void add_bias(Data& raw_data, Data& train_data);


#endif // !SDCA_UTILS_H

