
//#pragma once
//�������������л����Ĳ���������һЩ�ڻ������Ͷ��ļ�����

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

											 //�����ݵ�ʱ�����չ����ά��,��ÿ��sample�ĵ�һ��λ�����
											 //���⣬index�У����һ��ά�ȣ�index[n_sample+1]=X.size(),Ϊ�˷����дѭ��
void read_libsvm(const string filename, Data &train_data);
void read_libsvm_raw(const string filename, Data &train_data);
void add_bias(Data& raw_data, Data& train_data);


#endif // !SDCA_UTILS_H

