#include"libsvm_data.h"
#include"utils.h"
#include"dual_svm.h"

int main() {
	////basic parameter

	string loss = "smooth_hinge";
	double lambda = 0.0000001;
	int n_epoch = 30;
	double tol = 1e-16;
	bool verbose = true;

	//set dataset
	//string file_name = "ijcnn1";
	//string file_name = "a9a";
	//string file_name = "w8a.t";
	string file_name = "covtype";
	//string file_name = "news20.binary";

	//set algorithm
	//string algo = "SDCA";
	string algo = "SPDC";
	//string algo = "ASDCA";
	//string algo = "ASPDC";
	//string algo = "ASPDCi";
	//string algo = "SPD1";
	//string algo = "SVRG";

	std::cout << "loss = " << loss << std::endl
		<< "algorithm = " << algo << std::endl;

	std::cout << "strat reading data of file " << file_name << "..." << std::endl;
	Data train_data;
	read_libsvm(file_name, train_data);

	cout << "strat normalize data of " << file_name << "..." << endl;
	normalize_data(train_data);

	dual_svm clf(loss, lambda, tol, n_epoch, verbose);
	if (algo == "SDCA") {
		clf.fit_SDCA(train_data);
	}
	else if (algo == "SPDC") {
		clf.fit_SPDC(train_data);
	}
	else if (algo == "ASDCA") {
		clf.fit_ASDCA_v1(train_data);
	}
	else if (algo == "ASPDC") {
		clf.fit_ASPDC(train_data);
	}
	else if (algo == "ASPDCi") {
		clf.fit_ASPDCi(train_data);
	}
	else if (algo == "SPD1") {
		clf.fit_SPD1(train_data);
	}
	else if (algo == "SVRG") {
		clf.fit_SVRG(train_data);
	}
	else {
		std::cout << "Error in main.cpp: Not available algorithms!" << std::endl;
	}

	return 0;
}

