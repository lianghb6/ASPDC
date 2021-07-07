
#ifndef DUAL_SVM_H
#define DUAL_SVM_H

#include<string>
#include<vector>
#include<cmath>//ceil,sqrt
#include<random>
#include<iomanip>//output format
#include"libsvm_data.h"
#include"utils.h"
#include <chrono>//int
#include <algorithm>//random_shuffle
//#include<omp.h>

#define NOW std::chrono::system_clock::now() 

class dual_svm {
public:
	dual_svm(std::string _loss = "L2_svm", double _lambda = 0.0001, double _tol = 1e-15, int _n_epoch = 24, bool _verbose = true) :
		loss(_loss), lambda(_lambda), tol(_tol), n_epoch(_n_epoch), verbose(_verbose) {}

	void fit_SDCA(Data& train_data);
	void fit_ASDCA_v1(Data& train_data);////ASDCA simplied version
	void fit_ASDCA(Data& train_data);
	void fit_SPDC(Data& train_data);
	void fit_ASPDCi(Data& train_data);
	void fit_ASPDC(Data& train_data);
	void fit_Catalyst(Data& train_data);
	void fit_SPD1(Data& train_data);
	void fit_SVRG(Data& train_data);

	//for svrg
	
	double calculate_primal(Data& train_data)const;////calculate primal value

	double calculate_dual(Data& train_data)const;////calculate dual value, for SDCA and ASDCA only
	double calculate_dual_SPDC(Data& train_data)const;////claculate dual value for SPDC only
	double calculate_dual_ASPDC(Data& train_data)const;////calculate dual value for ASPDC only

	std::vector<double> primal_val_array;
	std::vector<double> dual_val_array;
	std::vector<double> dual_gap_array;

private:
	string loss;////L1-svm(not suit for SPDC),L2-svm,smooth hinge...
	double lambda;////penalty parameter
	double tol;////stop condition
	int n_epoch;
	bool verbose;

	std::vector<double> w;
	std::vector<double> alpha;

	//-------------------------for SPDC only
	std::vector<double> u;//auxiliary variable only used in SPDC
};

#endif // !DUAL_SVM_H

