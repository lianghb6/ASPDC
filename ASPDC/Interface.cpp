
#include <iostream>
#include "utils.h"
#include "dual_svm.h"
#include <chrono>
#include <ctime>
#include <string.h>
#include "mex.h"
using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	double lambda, tol;
	int n_epoch;
	bool verbose;
	std::string train_file, loss;

	//------------------------��matlab�˻�ò���
	char* file = new char[mxGetN(prhs[0]) + 1];////get file name
	mxGetString(prhs[0], file, mxGetN(prhs[0]) + 1);
	train_file = "";
	for (int i = 0; i < mxGetN(prhs[0]); i++) {////�����mxGetN(prhs[0])�����Ƕ����أ�
		train_file += file[i];
	}

	char* in_loss = new char[mxGetN(prhs[1]) + 1];////get loss
	mxGetString(prhs[1], in_loss, mxGetN(prhs[1]) + 1);
	loss = "";
	for (int i = 0; i < mxGetN(prhs[1]); i++) {
		loss += in_loss[i];
	}

	char* algorithm = new char[mxGetN(prhs[2]) + 1];////get algorithm name 
	mxGetString(prhs[2], algorithm, mxGetN(prhs[2]) + 1);

	lambda = mxGetScalar(prhs[3]);
	tol = mxGetScalar(prhs[4]);
	n_epoch = mxGetScalar(prhs[5]);
	verbose = mxGetScalar(prhs[6]);

	//------------------------C++�˼�����
	Data train_data;
	read_libsvm(train_file, train_data);
	//read_libsvm_raw(train_file, raw_data);//������������չ
	//normalize_data(raw_data);////���Ҫ�������ݶ�������򻯵�ʱ�䣬�Ͱ�����Ķ�ɾ����
	//add_bias(raw_data, train_data);
	normalize_data(train_data);

	dual_svm clf(loss, lambda, tol, n_epoch, verbose);

	if (strcmp(algorithm, "SDCA") == 0) {
		clf.fit_SDCA(train_data);
	}
	else if (strcmp(algorithm, "ASDCA") == 0) {
		clf.fit_ASDCA_v1(train_data);
	}
	else if (strcmp(algorithm, "SPDC") == 0) {
		clf.fit_SPDC(train_data);
	}
	else if (strcmp(algorithm, "ASPDC") == 0) {
		clf.fit_ASPDC(train_data);
	}
	else if (strcmp(algorithm, "ASPDCi") == 0) {
		clf.fit_ASPDCi(train_data);
	}
	else if (strcmp(algorithm, "SVRG") == 0) {
		clf.fit_SVRG(train_data);
	}
	else {
		std::cout << "algorithm  is not available in Interface.cpp" << std::endl;
	}

	//------------------------��C++�˴������
	plhs[0] = mxCreateDoubleMatrix(clf.dual_gap_array.size(), 1, mxREAL);//�����һ��������dual_gap_array
	double* dual_gap = mxGetPr(plhs[0]);
	for (int i = 0; i < clf.dual_gap_array.size(); i++) {
		dual_gap[i] = clf.dual_gap_array[i];
	}

	plhs[1] = mxCreateDoubleMatrix(clf.primal_val_array.size(), 1, mxREAL);//����ڶ���������primal_val_array
	double* primal_val = mxGetPr(plhs[1]);
	for (int i = 0; i < clf.primal_val_array.size(); i++) {
		primal_val[i] = clf.primal_val_array[i];
	}

	plhs[2] = mxCreateDoubleMatrix(clf.dual_val_array.size(), 1, mxREAL);//�������������dual_val_array,ע��SPDC�Ķ�ż��ʽ��SDCA����ϸ΢���
	double* dual_val = mxGetPr(plhs[2]);
	for (int i = 0; i < clf.dual_val_array.size(); ++i) {
		dual_val[i] = clf.dual_val_array[i];
	}

}
