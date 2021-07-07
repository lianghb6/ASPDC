#include"dual_svm.h"
#include<float.h>//DBL_MAX DBL_MIN

////在计算dual value 上，SDCA和ASDCA公用一个，而SPDC和FSPDC公用另一个
double dual_svm::calculate_primal(Data& train_data) const {
	double regularizer = 0.5*lambda*dot_dense(w);

	double sum = 0.0;
	if (loss == "hinge") {
		double temp = 0.0;
		for (int i = 0; i < train_data.n_sample; ++i) {
			temp = 1 - train_data.Y[i] * dot_sparse(train_data, i, w);
			sum += max(0.0, temp);//+=
		}
	}
	else if (loss == "L2_svm") {
		double temp = 0.0;
		for (int i = 0; i < train_data.n_sample; ++i) {
			temp = 1 - train_data.Y[i] * dot_sparse(train_data, i, w);
			sum += max(0.0, temp)*max(0.0, temp);//+=
		}
	}
	else if (loss == "smooth_hinge") {
		//SPDC demo loss type
		double temp = 0.0;
		for (int i = 0; i < train_data.n_sample; ++i) {
			temp = train_data.Y[i] * dot_sparse(train_data, i, w);
			if (temp >= 1.0) {
				sum += 0.0;//+=
			}
			else if (temp <= 0.0) {
				sum += 0.5 - temp;//+=
			}
			else if (temp > 0.0&&temp < 1.0) {
				sum += 0.5*(1 - temp)*(1 - temp);//+=
			}
		}
	}
	else {
		std::cout << "Error in in dual_svm.cpp/calculate_primal(): Not available loss type !" << std::endl;
	}

	return (sum / (double)train_data.n_sample + regularizer);
}

double dual_svm::calculate_dual(Data& train_data) const {
	//for SDCA and ASDCA
	//double regularizer = 0.5*lambda*dot_dense(w);//修改为用alpha计算会有什么影响么，除了时间

	vector<double> regularizer(train_data.n_feature, 0.0);
	for (int i = 0; i < train_data.n_sample; ++i) {//遍历一次所有数据
		for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//遍历一个数据,嵌套的时候小心别混淆索引
			regularizer[train_data.col[j]] += alpha[i] * train_data.X[j] / ((double)train_data.n_sample*lambda);
			////复合赋值操作，小心别写成直接赋值了.另外一定要小心,与SPDC,ASPDC有细微差别，是+=，而不是-=
		}
	}

	double sum = 0.0;
	if (loss == "hinge") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			sum += alpha[i] * train_data.Y[i];//+=，//注意与SPDC的和ASPDC的不同
		}
	}
	else if (loss == "L2_svm") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			sum += alpha[i] * train_data.Y[i] - alpha[i] * alpha[i] / 4.0;//+=
		}
	}
	else if (loss == "smooth_hinge") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			sum += train_data.Y[i] * alpha[i] - 0.5*alpha[i] * alpha[i];//+=
		}
	}
	else {
		std::cout << "Error in dual_svm.cpp/calculate_dual(): Not available loss type !" << std::endl;
	}

	return (sum / (double)train_data.n_sample - 0.5*lambda*dot_dense(regularizer));
	//return (sum / (double)train_data.n_sample - regularizer);
}

double dual_svm::calculate_dual_SPDC(Data& train_data) const {
	//for SPDC	
	double regularizer = 0.5*dot_dense(u) / lambda;

	double sum = 0.0;
	if (loss == "L2_svm") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			sum = sum - alpha[i] * train_data.Y[i] - alpha[i] * alpha[i] / 4.0;//注意这里和SDCA和ASDCA的不一样
		}
	}
	else if (loss == "smooth_hinge") {//for SPDC
		for (int i = 0; i < train_data.n_sample; ++i) {
			sum = sum - train_data.Y[i] * alpha[i] - 0.5*alpha[i] * alpha[i];
		}
	}
	else {
		std::cout << "Not available loss type in dual_svm.cpp//calculate_dual_SPDC()!!" << std::endl;
	}

	return (sum / (double)train_data.n_sample - regularizer);

	//for debug
	/*double regularizer = 0.5*dot_dense(u) ;
	regularizer = regularizer / lambda;
	double sum = 0.0;
	if (loss == "L2_svm") {
	for (int i = 0; i < train_data.n_sample; ++i) {
	sum = sum - alpha[i] * train_data.Y[i] - alpha[i] * alpha[i] / 4.0;
	}
	}
	else {
	std::cout << "Not available loss type!!" << std::endl;
	}

	int cnt = 0;
	for (int i = 0; i < train_data.n_sample; i++) {
	if (train_data.Y[i] * alpha[i] > 0) {
	++cnt;
	}
	}
	std::cout << "there are totaly " << cnt/(double)train_data.n_sample<< "ratio of alpha[i] that volate the subjection (Yi*alpha[i]<0)" << std::endl;
	double res = sum / (double)train_data.n_sample;
	res = res - regularizer;
	return res;*/
}

double dual_svm::calculate_dual_ASPDC(Data& train_data) const {
	////for SPDC_r7, SPDC_r8 and ASPDC
	////在ASDCA的lemma3证明中，D(alpha)并不是这个形式，这里和ASDCA算法中的保持一致。
	////为什么要另外设置一个计算对偶函数值的函数呢？因为SPDC_r7有些特殊，它又像SDCA，又像SPDC.所以，只好给他写一个对偶函数的计算式子。

	//double regularizer = 0.5*lambda*dot_dense(w);
	//要理解为什么要修改，因为在每一次epoch中，我们都用AccSPDC去求解一个近似的primal-dual问题。w_prox作为primal变量，alpha作为dual变量。
	//而w是作为外层epoch的primal变量（也恰恰是primal原问题的变量）
	//而这里需要求的，不是作为近似primal-dual 的dual value，而是原始dual value的值。所以一定要分清楚。
	//所以其实，这和SPDC的dual value的计算函数值形式是一样的，不过在SPDC中，可以利用U直接求出来。

	vector<double> regularizer(train_data.n_feature, 0.0);
	for (int i = 0; i < train_data.n_sample; ++i) {//遍历一次所有数据
		for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//遍历一个数据,嵌套的时候小心别混淆索引
			regularizer[train_data.col[j]] += -1.0*alpha[i] * train_data.X[j] / ((double)train_data.n_sample*lambda);
			////复合赋值操作，小心别写成直接赋值了.另外一定要小心,与SDCA，ASDCA有细微差别，是-=，而不是+=
		}
	}

	double res = 0.0;
	for (int i = 0; i < train_data.n_sample; ++i) {
		if (loss == "L2_svm") {
			res += -train_data.Y[i] * alpha[i] - alpha[i] * alpha[i] / 4.0;//+=
		}
		else if (loss == "smooth_hinge") {//for SPDC
			res += -train_data.Y[i] * alpha[i] - 0.5*alpha[i] * alpha[i];//+=
		}
		else {
			std::cout << "Unavailable loss type in dual_svm.cpp//calculate_dual_ASPDC" << std::endl;
		}
	}
	return (res / (double)train_data.n_sample - 0.5*lambda*dot_dense(regularizer));
}

//****************************************************************************************SDCA算法

void dual_svm::fit_SDCA(Data& train_data) {
	std::cout << " invoking SDCA algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample  = " << n_sample << std::endl
		<< "  n_feature + 1 = " << n_feature << endl
		<< " lambda =  " << lambda << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	//初始化变量w,alpha
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);

	//normalize_data(train_data);

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);//std::uniform_int_distribution<size_t> distribution(0, n_sample - 1)貌似有问题

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;
		for (int idx = 0; idx < n_sample; ++idx)//idx只在本行出现
		{
			int rand_id = distribution(generator);
			double d = 0.0;
			if (loss == "hinge") {
				double temp = 1 - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w);
				temp = temp*lambda*n_sample + alpha[rand_id] * train_data.Y[rand_id];
				d = train_data.Y[rand_id] * max(0.0, min(1.0, temp)) - alpha[rand_id];
			}
			else if (loss == "L2_svm") {
				double temp = 1 - 0.5*alpha[rand_id] * train_data.Y[rand_id] - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w);
				temp = temp*lambda*n_sample * 2.0 / (2.0 + lambda*n_sample);
				d = train_data.Y[rand_id] * max(-1.0 * train_data.Y[rand_id] * alpha[rand_id], temp);
			}
			else if (loss == "smooth_hinge") {
				double temp = 1 - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w) - train_data.Y[rand_id] * alpha[rand_id];
				temp = temp*lambda*n_sample / (1.0 + lambda*n_sample) + alpha[rand_id] * train_data.Y[rand_id];
				d = train_data.Y[rand_id] * max(0.0, min(1.0, temp)) - alpha[rand_id];
			}
			else {
				std::cout << "Error in dual_svm.cpp/fit_SDCA(): Not available loss type!!" << std::endl;
			}

			alpha[rand_id] += d;
			//w的更新,注意是+=,这与SPDC的不一样。
			for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
				//遍历一个sample rand_id
				w[train_data.col[k]] += d* train_data.X[k] / (lambda*n_sample);//+=
			}

		}//end for-idx

		 //calculate the dual_gap
		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual(train_data);
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(dual_gap);

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}//end if-dual_gap
	}//end for-epoch

}

//****************************************************************************************ASDCA_v1算法

////ASDCA的简化版本,简化了终止条件

void dual_svm::fit_ASDCA_v1(Data& train_data) {
	std::cout << " invoking ASDCA_v1 algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample  = " << n_sample << "  n_feature + 1 = " << n_feature << endl;
	std::cout << " lambda = " << lambda << std::endl;

	double gamma = 1.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;//wrong
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Error in dual_svm.cpp/ASDCA(): Not available loss type!! " << std::endl;
	}

	////ASCDA基本参数
	double kappa = 1.0 / (gamma*n_sample) - lambda;

	double mu = lambda / 2.0;
	double rho = mu + kappa;
	double eta = sqrt(mu / rho);
	double beta = (1.0 - eta) / (1.0 + eta);

	std::cout << " kappa = " << kappa << std::endl
		<< "mu = " << mu << std::endl
		<< "rho = " << rho << std::endl
		<< "eta = " << eta << std::endl
		<< "beta = " << beta << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	if (1.0 / (gamma*lambda) < 10 * n_sample) {
		std::cout << "1.0 / (gamma*lambda) < 10 * n_sample" << std::endl;
		return;////ASDCA条件不满足,当lambda足够小的时候才调用ASDCAss
	}

	//初始化变量w,alpha
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	vector<double> w_prox(n_feature, 0.0);//在每个epoch用这个变量作为迭代变量

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;

		////每个epoch之前初始化w_prox
		for (int j = 0; j < n_feature; ++j) {
			w_prox[j] = kappa*w[j] / (lambda + kappa);//赋值初始化
		}
		for (int i = 0; i < n_sample; ++i) {//遍历一次所有数据
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//遍历一个数据,嵌套的时候小心别混淆索引
				w_prox[train_data.col[j]] += alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));//符合赋值操作，小心别写成直接赋值了，另外和ASPDC也有点区别。
			}
		}
		////初始化结束

		for (int idx = 0; idx < n_sample * 2; ++idx) {
			////简化了ASDCA的退出条件,在这个for循环内，w_prox(而不是w)是迭代变量
			int rand_id = distribution(generator);
			double d = 0.0;
			////alpha的更新和SDCA相似,只是需要修改lambda参数。
			if (loss == "hinge") {
				std::cout << "Warning in dual_svm.cpp/ASDCA: ASDCA may be not suited to hinge loss." << std::endl;
				//double temp = 1 - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox);
				//temp = temp*(lambda + kappa)*n_sample + alpha[rand_id] * train_data.Y[rand_id];
				//d = train_data.Y[rand_id] * max(0.0, min(1.0, temp)) - alpha[rand_id];
			}
			else if (loss == "L2_svm") {
				double temp = 1 - 0.5*alpha[rand_id] * train_data.Y[rand_id] - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox);
				temp = temp*(lambda + kappa)*n_sample * 2.0 / (2.0 + (lambda + kappa)*n_sample);
				d = train_data.Y[rand_id] * max(-1.0* train_data.Y[rand_id] * alpha[rand_id], temp);
			}
			else if (loss == "smooth_hinge") {
				double temp = 1 - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox) - train_data.Y[rand_id] * alpha[rand_id];
				temp = temp*(lambda + kappa)*n_sample / (1.0 + (lambda + kappa)*n_sample) + alpha[rand_id] * train_data.Y[rand_id];
				d = train_data.Y[rand_id] * max(0.0, min(1.0, temp)) - alpha[rand_id];
			}
			else {
				std::cout << "Error in dual_svm.cpp/ASDCA(): Not available loss type!!" << std::endl;
			}

			alpha[rand_id] += d;
			//w_prox的更新
			for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
				//遍历一个sample rand_id，注意下面的更新是+=，这和SPDC的不一样
				w_prox[train_data.col[k]] += d* train_data.X[k] / ((double)n_sample*(lambda + kappa));
			}
		}////end for-idx

		 // //momentum
		for (int j = 0; j < n_feature; ++j) {
			w[j] = w_prox[j] + beta*(w_prox[j] - w[j]);
		}

		//////no momentum,实验表明，ASDCA中没有momentum更加快
		//for (int j = 0; j < n_feature; ++j) {
		//	w[j] = w_prox[j];
		//}

		//calculate the dual_gap
		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual(train_data);
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(abs(dual_gap));

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}

	}////end for-epoch

}

//****************************************************************************************ASDCA算法

void dual_svm::fit_ASDCA(Data& train_data) {
	////这是完整的ASDCA，综合考虑了每个epoch的停止条件以及整个算法的两个停止条件。
	std::cout << " invoking ASDCA algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample  = " << n_sample << "  n_feature + 1 = " << n_feature << endl;
	std::cout << " lambda = " << lambda << std::endl;

	double gamma = 1.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;//wrong
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Error in dual_svm.cpp/ASDCA(): Not available loss type!! " << std::endl;
	}

	////初始化变量w,alpha
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	vector<double> w_prox(n_feature, 0.0);////在每个epoch内用这个变量作为迭代变量

	vector<double> delta_wy(n_feature, 0.0);////only use in stopping condition 2

											////ASCDA基本参数
	double kappa = 1.0 / (gamma*n_sample) - lambda;
	double mu = lambda / 2.0;
	double rho = mu + kappa;
	double eta = sqrt(mu / rho);
	double beta = (1.0 - eta) / (1.0 + eta);
	double _xi = (1 + 1.0 / (eta*eta))*(calculate_primal(train_data) - calculate_dual(train_data));
	////_xi关乎整个算法停止条件
	std::cout << "P(0)-D(0) = " << (calculate_primal(train_data) - calculate_dual(train_data)) << std::endl;//debug, =0.5

	std::cout << " kappa = " << kappa << std::endl
		<< "mu = " << mu << std::endl
		<< "rho = " << rho << std::endl
		<< "eta = " << eta << std::endl
		<< "beta = " << beta << std::endl
		<< "_xi = " << _xi << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	std::cout << "The lambda should be less than " << 1.0 / (gamma * 10 * n_sample) << "  Otherwise call SDCA/prox_SDCA" << std::endl;
	if (1.0 / (gamma*lambda) < 10 * n_sample) {
		std::cout << "1.0 / (gamma*lambda) < 10 * n_sample" << std::endl;
		return;////ASDCA条件不满足,当lambda足够小的时候才调用ASDCA
	}

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	////stopping condition 1
	double max_epoch = 1 + log(_xi / tol)*0.5 / eta;
	std::cout << "Stopping condition 1: ASDCA need at least : " << max_epoch << " epoches to aproach the given precision" << std::endl;

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;

		//---------------------------------------------------------------每一个while都是解决一个point proximal 子问题
		////每个epoch之前初始化w_prox
		for (int j = 0; j < n_feature; ++j) {
			w_prox[j] = kappa*w[j] / (lambda + kappa);//赋值初始化
		}
		for (int i = 0; i < n_sample; ++i) {//遍历一次所有数据
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//遍历一个数据,嵌套的时候小心别混淆索引
				w_prox[train_data.col[j]] += alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));//符合赋值操作，小心别写成直接赋值了，另外和ASPDC也有点区别。
			}
		}
		////初始化结束
		////epoch之前，先决定epsilon的值
		double epsilon = _xi*0.5*eta / (1 + 1.0 / (eta*eta));
		std::cout << "epsilon = " << epsilon << std::endl;
		double eps = 0.0;////用来存储proximal problem 最后一次proximal dual gap,for stopping condition 2 only.

		for (int cnt = 0; cnt < 10; ++cnt) {
			////解决一个proximal point问题，如果提前达到指定精度就停止了
			////在这个循环内，w是常量，w_prox才是proximal point 变量
			////该循环有cnt*n的复杂度
			double prox_primal = 0.0;
			double prox_dual = 0.0;
			double prox_dual_gap = 0.0;
			////计算proximal primal value
			double sum = 0.0;
			if (loss == "hinge") {
				double temp = 0.0;
				for (int i = 0; i < train_data.n_sample; ++i) {
					temp = 1 - train_data.Y[i] * dot_sparse(train_data, i, w_prox);
					sum += max(0.0, temp);//+=
				}
			}
			else if (loss == "L2_svm") {
				double temp = 0.0;
				for (int i = 0; i < train_data.n_sample; ++i) {
					temp = 1 - train_data.Y[i] * dot_sparse(train_data, i, w_prox);
					sum += max(0.0, temp)*max(0.0, temp);//+=
				}
			}
			else if (loss == "smooth_hinge") {
				//SPDC demo loss type
				double temp = 0.0;
				for (int i = 0; i < train_data.n_sample; ++i) {
					temp = train_data.Y[i] * dot_sparse(train_data, i, w_prox);
					if (temp >= 1.0) {
						sum += 0.0;//+=
					}
					else if (temp <= 0.0) {
						sum += 0.5 - temp;//+=
					}
					else if (temp > 0.0&&temp < 1.0) {
						sum += 0.5*(1 - temp)*(1 - temp);//+=
					}
				}
			}
			else {
				std::cout << "Error in in dual_svm.cpp: Not available loss type !" << std::endl;
			}
			prox_primal = sum / double(n_sample) + 0.5*(lambda + kappa)*dot_dense(w_prox) - kappa*dot_dense(w, w_prox);

			////计算proximal dual value
			sum = 0.0;////reset
			if (loss == "hinge") {
				for (int i = 0; i < train_data.n_sample; ++i) {
					sum += alpha[i] * train_data.Y[i];//+=，//注意与SPDC的和ASPDC的不同
				}
			}
			else if (loss == "L2_svm") {
				for (int i = 0; i < train_data.n_sample; ++i) {
					sum += alpha[i] * train_data.Y[i] - alpha[i] * alpha[i] / 4.0;//+=
				}
			}
			else if (loss == "smooth_hinge") {
				for (int i = 0; i < train_data.n_sample; ++i) {
					sum += train_data.Y[i] * alpha[i] - 0.5*alpha[i] * alpha[i];//+=
				}
			}
			else {
				std::cout << "Error in dual_svm.cpp: Not available loss type !" << std::endl;
			}

			prox_dual = sum / double(n_sample) - 0.5*(lambda + kappa)*dot_dense(w_prox);
			prox_dual_gap = prox_primal - prox_dual;

			cout << "counter" << ": " << cnt
				<< "--proximal primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << prox_primal
				<< "--proximal dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << prox_dual
				<< "--proximal dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << prox_dual_gap << endl;

			if (prox_dual_gap < epsilon) {
				eps = prox_dual_gap;
				std::cout << "Proximal dual gap have approached the given epsilon!!" << std::endl;
				break;////如果达到了给定精度，就停止
			}

			////如果没达到给定精度，就继续执行n次计算,针对proximal problem而言的。
			for (int idx = 0; idx < n_sample; ++idx) {
				int rand_id = distribution(generator);
				double d = 0.0;
				////alpha的更新和SDCA相似,只是需要修改lambda参数。
				if (loss == "hinge") {
					std::cout << "Warning in dual_svm.cpp/ASDCA: ASDCA may be not suited to hinge loss." << std::endl;
					//double temp = 1 - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox);
					//temp = temp*(lambda + kappa)*n_sample + alpha[rand_id] * train_data.Y[rand_id];
					//d = train_data.Y[rand_id] * max(0.0, min(1.0, temp)) - alpha[rand_id];
				}
				else if (loss == "L2_svm") {
					double temp = 1 - 0.5*alpha[rand_id] * train_data.Y[rand_id] - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox);
					temp = temp*(lambda + kappa)*n_sample * 2.0 / (2.0 + (lambda + kappa)*n_sample);
					d = train_data.Y[rand_id] * max(-1.0* train_data.Y[rand_id] * alpha[rand_id], temp);
				}
				else if (loss == "smooth_hinge") {
					double temp = 1 - train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox) - train_data.Y[rand_id] * alpha[rand_id];
					temp = temp*(lambda + kappa)*n_sample / (1.0 + (lambda + kappa)*n_sample) + alpha[rand_id] * train_data.Y[rand_id];
					d = train_data.Y[rand_id] * max(0.0, min(1.0, temp)) - alpha[rand_id];
				}
				else {
					std::cout << "Error in dual_svm.cpp/ASDCA(): Not available loss type!!" << std::endl;
				}

				alpha[rand_id] += d;
				//w_prox的更新
				for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
					//遍历一个sample rand_id，注意下面的更新是+=，这和SPDC的不一样
					w_prox[train_data.col[k]] += d* train_data.X[k] / ((double)n_sample*(lambda + kappa));
				}

			}////end for-idx

		}////end for(cnt)

		 //---------------------------------------------------------------------------------proximal 问题求解结束

		 ////stopping condition 2,为了方便，就在这更新之前计算了
		for (int j = 0; j < n_feature; ++j) {
			delta_wy[j] = w_prox[j] - w[j];
		}
		double cond_num_2 = (1 + rho / mu)*eps + dot_dense(delta_wy)*rho*kappa / (2.0*mu);
		std::cout << "Precision given by stopping condition 2: " << cond_num_2 << endl;
		if (cond_num_2 < tol) {
			std::cout << "Stopping condition 2 holds" << std::endl;
			break;
		}

		// //momentum
		for (int j = 0; j < n_feature; ++j) {
			w[j] = w_prox[j] + beta*(w_prox[j] - w[j]);
		}

		//////no momentum,实验表明，ASDCA中没有momentum更加快
		//for (int j = 0; j < n_feature; ++j) {
		//	w[j] = w_prox[j];
		//}

		_xi = _xi*(1 - 0.5*eta);

		//calculate the dual_gap
		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual(train_data);
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(abs(dual_gap));



		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " Dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl
				<< "--------------------------------------------------------------------------------------------" << std::endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}


	}////end for-epoch

}

//****************************************************************************************SPDC算法

void dual_svm::fit_SPDC(Data& train_data) {
	std::cout << "invoking SPDC algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample  " << n_sample << "  n_feature + 1 " << n_feature << endl
		<< " lambda" << lambda << std::endl;

	//初始化变量w,alpha，u
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	u = vector<double>(n_feature, 0.0);//为了计算对偶函数值，把它作为私有成员了

	vector<double> w_bar(n_feature, 0.0);//式子(8)中的momentum加速变量
	vector<double> w_old(n_feature, 0.0);//式子（8）的实现需要设置w的上一个迭代值

	double gamma = 0.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Error in dual_svm.cpp/fit_SPDC(): Not available loss type!!" << std::endl;
	}

	double _tau = 0.5*sqrt(gamma / (double)(n_sample*lambda));
	double _sigma = 0.5*sqrt(n_sample*lambda / gamma);
	double _theta = 1.0 - 1.0 / (n_sample + 2 * sqrt(n_sample / (lambda*gamma)));
	std::cout << "_tau = " << _tau << " _sigma = " << _sigma << " _theta = " << _theta << std::endl;
	//normalize_data(train_data);

	std::random_device rd;//random_device 类定义的函数对象可以生成用来作为种子的随机的无符号整数值。每一个 rd() 调用都会返回不同的值
	std::default_random_engine generator(rd());//创建随机数生成器，其中rd()是随机种子。
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);//创建分布对象

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;
		for (int idx = 0; idx < n_sample; ++idx)
		{
			int rand_id = distribution(generator);//产生随机数
			//更新alpha[rand_id]
			double res = 0.0;
			if (loss == "L2_svm") {
				res = alpha[rand_id] + _sigma*(dot_sparse(train_data, rand_id, w_bar) - train_data.Y[rand_id]);//用w_bar的值去更新y
				res = res / (1.0 + 0.5*_sigma);
				if (train_data.Y[rand_id] > 0) {
					res = min(0.0, res);//res<0.0
				}
				else {
					res = max(0.0, res);//res>0
				}
			}
			else if (loss == "smooth_hinge") {
				res = alpha[rand_id] + _sigma*(dot_sparse(train_data, rand_id, w_bar) - train_data.Y[rand_id]);
				res = res*train_data.Y[rand_id] / (1.0 + _sigma);
				res = train_data.Y[rand_id] * max(-1.0, min(0.0, res));
			}
			else {
				std::cout << "Error in dual_svm.cpp/fit_SPDC(): Not available loss type!!" << std::endl;
			}
			double delta = res - alpha[rand_id];
			alpha[rand_id] = res;

			//更新w
			for (int i = 0; i < n_feature; i++) {
				w_old[i] = w[i];//用在momentum实现中。
			}

			vector<double> u_temp(n_feature, 0.0);
			for (int k = 0; k < n_feature; ++k) {
				u_temp[k] = u[k];//相当于每个epoch也重新初始化
			}
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {
				u_temp[train_data.col[j]] += delta*train_data.X[j];
			}

			for (int i = 0; i < n_feature; i++) {
				//计算式子（6）的中间量
				w[i] = (w[i] - _tau*u_temp[i]) / (1.0 + lambda*_tau);//更新w的时候不需要用w_bar，直接用w的就行
			}

			//更新u
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {
				u[train_data.col[j]] += delta*train_data.X[j] / (double)n_sample;
			}

			//extraction
			for (int i = 0; i < n_feature; i++) {
				w_bar[i] = w[i] + _theta*(w[i] - w_old[i]);//w_bar的更新
			}
			//一次SPDC迭代结束

		}//end for-idx

		 //注意每一个epoch输出的是w而不是w_bar
		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual_SPDC(train_data);
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(dual_gap);

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}

	}//end for-epoch
}
//****************************************************************************************SPD1

//void dual_svm::fit_SPD1(Data& train_data) {
//
//	std::cout << "invoking SPD1 algorithm..." << std::endl;
//	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
//	std::cout << "n_sample  " << n_sample << "  n_feature + 1 " << n_feature << std::endl
//		<< " lambda" << lambda << std::endl;
//
//	//初始化变量w,alpha，u
//	alpha = vector<double>(n_sample, 0.0);//输出，是每次迭代值的平均
//	w = vector<double>(n_feature, 0.0);//输出，是每次迭代的平均值
//
//	vector<double> w_hat(n_feature, 0.0);//迭代变量
//	vector<double> alpha_hat(n_sample, 0.0);//迭代变量
//
//	vector<double> w_sum(n_feature, 0.0);//累积的迭代变量
//	vector<double> alpha_sum(n_sample, 0.0);//累积的迭迭代变量
//
//
//	double gamma = 0.0;
//	if (loss == "L2_svm") {
//		//gamma = 2.0;
//		gamma = 0.5;
//	}
//	else if (loss == "smooth_hinge") {
//		gamma = 1.0;
//	}
//	else {
//		std::cout << "Error in dual_svm.cpp/fit_SPDC(): Not available loss type!!" << std::endl;
//	}
//
//	std::random_device rd;//生成随机数种子
//
//	std::default_random_engine generator_dual(rd());//随机数生成器
//	std::uniform_int_distribution<int> distribution_dual(0, n_sample - 1);//分布对象
//
//	std::default_random_engine generator_primal(rd());//随机数生成器
//	std::uniform_int_distribution<int> distribution_primal(0, n_feature - 1);//分布对象
//
//	for (int t = 0; t < n_sample*n_feature*(n_epoch); ++t) 
//	{
//		//随机选择两个不同的i,j
//		int rand_i = distribution_dual(generator_dual);//随机生成[0,n-1]的整数
//		int rand_j = distribution_primal(generator_primal);//随机生成[0,d-1]的整数
//
//		double tau = 2.0 / (lambda * (double)(4 + t));
//		double eta = 2.0 * n_sample*n_feature / (gamma * (double)(t + 4));
//
//		//找到Xij元素
//		double Xij = 0.0;//有可能Xij=0的啊
//		for (int k = train_data.index[rand_i]; k < train_data.index[rand_i + 1]; ++k) {
//			if (train_data.col[k] == rand_j) {
//				Xij = train_data.X[k];
//				break;
//			}
//		}
//
//		if (Xij == 0)
//			continue;
//
//		if (loss == "smooth_hinge") {
//			//更新w_hat
//			w_hat[rand_j] = (w_hat[rand_j] - eta*Xij*alpha_hat[rand_i]) / (1.0 + lambda*eta);
//			//alpha
//			double temp = train_data.Y[rand_i] * (alpha_hat[rand_i] + tau*Xij*w_hat[rand_j] - train_data.Y[rand_i] * tau / (double)n_feature);
//			temp = temp / (1 + tau / (double)n_feature);
//			alpha_hat[rand_i] = train_data.Y[rand_i] * max(min(temp, 0), -1);
//		}
//		else {
//			std::cout << "Error in dual_svm.cpp/fit_SPD1(): Not available loss type!!" << std::endl;
//		}
//
//		w_sum[rand_j] += w_hat[rand_j];
//		alpha_sum[rand_i] += alpha_hat[rand_i];
//
//		if ((t + 1) % n_sample == 0) 
//		{
//			//注意是赋值，而不是+=
//			for (int k = 0; k < n_feature; ++k) {
//				w[k] = w_sum[k] / t;
//			}
//
//			for (int k = 0; k < n_sample; ++k) {
//				alpha[k] = alpha_sum[k] / t;
//			}
//
//			double primal_val = calculate_primal(train_data);
//			primal_val_array.push_back(primal_val);
//
//			if (verbose) {
//				cout << "epoch " << ": " << t/n_sample
//					<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val << std::endl;
//			}
//		}
//
//	}
//
//}


void dual_svm::fit_SPD1(Data& train_data) {
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "invoking SPD1 algorithm..." << std::endl
		<< "n_sample  " << n_sample << "  n_feature + 1 " << n_feature << std::endl
		<< " lambda" << lambda << std::endl;

	//初始化变量w,alpha，u
	alpha = vector<double>(n_sample, 0.0);//输出变量，是每次迭代值的平均
	w = vector<double>(n_feature, 0.0);//输出变量，是每次迭代的平均值
	
	vector<double> w_hat(n_feature, 0.0);//迭代变量
	vector<double> alpha_hat(n_sample, 0.0);//迭代变量
	for (int k = 0; k < n_sample; ++k) {
		if (train_data.Y[k] == 1)
			alpha_hat[k] = -1.0;
		else if (train_data.Y[k] == -1)
			alpha_hat[k] = 1.0;
		else
			std::cout << "Error in SPD1: wrong label." << std::endl;
	}

	vector<double> w_sum(n_feature, 0.0);//累积的迭代变量
	vector<double> alpha_sum(n_sample, 0.0);//累积的迭迭代变量


	double gamma = 0.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Error in dual_svm.cpp/fit_SPDC(): Not available loss type!!" << std::endl;
	}

	std::random_device rd;//生成随机数种子

	std::default_random_engine generator_dual(rd());//随机数生成器
	std::uniform_int_distribution<int> distribution_dual(0, n_sample - 1);//分布对象

	std::default_random_engine generator_primal(rd());//随机数生成器
	std::uniform_int_distribution<int> distribution_primal(0, n_feature - 1);//分布对象

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		
		for (int idx = 0; idx < n_sample*n_feature; ++idx)
		{
			//对于SPD1来说。一个epoch遍历一遍数据集，相当于n*d次更新
			//确定参数
			int t = n_sample*n_feature*epoch + idx;//当前迭代次数
			double eta = 2.0 / (lambda * (double)(4 + t));
			double tau = 2.0 * n_sample*n_feature / (gamma * (double)(t + 4));

			//随机选择两个不同的i,j
			int rand_i = distribution_dual(generator_dual);//随机生成[0,n-1]的整数
			int rand_j = distribution_primal(generator_primal);//随机生成[0,d-1]的整数

			//更新primal迭代变量w_hat,以及//更新dual变脸alpha_hat
			//找到Xij元素
			double Xij = 0.0;//有可能Xij=0的啊
			for (int k = train_data.index[rand_i]; k < train_data.index[rand_i + 1]; ++k) {
				if (train_data.col[k] == rand_j) {
					Xij = train_data.X[k];
					break;
				}
			}

			//if (Xij == 0)
			//	continue;//xij=0时候也需要更新的

			if (loss == "smooth_hinge") {
				//更新w_hat
				w_hat[rand_j] = (w_hat[rand_j] - eta*Xij*alpha_hat[rand_i]) / (1.0 + lambda*eta);
				//alpha
				double temp = train_data.Y[rand_i] * (alpha_hat[rand_i] + tau*Xij*w_hat[rand_j] - train_data.Y[rand_i] * tau / (double)n_feature);//算法SPD1框架公式
				temp = temp / (1 + tau / (double)n_feature);
				alpha_hat[rand_i] = temp;
				//alpha_hat[rand_i] = train_data.Y[rand_i] * max(min(temp, 0.0), -1.0);//按理说应该是需要加这个约束的啊,但是加上之后一直在在上升
			}
			else {
				std::cout << "Error in dual_svm.cpp/fit_SPD1(): Not available loss type!!" << std::endl;
			}
	
			//一次SPD1迭代结束

			//计算累积的w_sum和alpha_sum
			w_sum[rand_j] += w_hat[rand_j];
			alpha_sum[rand_i] += alpha_hat[rand_i];

		}//end for-idx


		////注意是赋值，而不是+=
		double t = (epoch+1)*n_sample*n_feature;
		for (int k = 0; k < n_feature; ++k) {
			w[k] = w_sum[k]/t;
		}

		for (int k = 0; k < n_sample; ++k) {
			alpha[k] = alpha_sum[k]/t;
		}

		//取最后一次迭代是不收敛的。
		//for (int k = 0; k < n_feature; ++k) {
		//	w[k] = w_hat[k] ;
		//}

		//for (int k = 0; k < n_sample; ++k) {
		//	alpha[k] = alpha_hat[k];
		//}

		 //注意每一个epoch输出的是w而不是w_hat
		double primal_val = calculate_primal(train_data);
		primal_val_array.push_back(primal_val);

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val<<std::endl;
		}


	}//end for-epoch

}

//****************************************************************************************SVRG

void dual_svm::fit_SVRG(Data& train_data) {

	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << " invoking SVRG algorithm..." << std::endl
		<< "n_sample  = " << n_sample << std::endl
		<< "  n_feature + 1 = " << n_feature << endl
		<< " lambda =  " << lambda << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	//初始化变量w
	w = vector<double>(n_feature, 0.0); //外层变量，同时也是输出变量
	vector<double> w_hat(n_feature, 0.0);//内循环迭代变量

	//SVRG的参数
	double gamma = 0.0;//smooth参数
	if (loss == "L2_svm") {
		//gamma = 2.0;
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Error in dual_svm.cpp/fit_SVRG(): Not available loss type!!" << std::endl;
	}

	double eta = 0.1/ gamma;//步长参数
	//double eta = 0.01;

	std::cout << eta << std::endl;
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);//std::uniform_int_distribution<size_t> distribution(0, n_sample - 1)貌似有问题

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		//计算在w处的梯度值mu
		vector<double> mu(n_feature, 0.0);//snapshot的梯度值
		for (int j = 0; j < n_feature; ++j) {
			mu[j] = lambda* w[j];
		}
		//遍历一遍样本
		for (int i = 0; i < n_sample; ++i) {
			double linearPreditor = train_data.Y[i] * dot_sparse(train_data, i, w);
			double nabla = 0.0;
			if (linearPreditor >= 1.0) {
				nabla = 0.0;
			}
			else if (linearPreditor <= 0.0) {
				nabla = -train_data.Y[i];
			}
			else {
				nabla = linearPreditor - 1.0;
			}
			//nabla*Xi/n
			for (int k = train_data.index[i]; k < train_data.index[i + 1]; ++k) {
				mu[train_data.col[k]] += nabla*train_data.X[k] / (double)n_sample;
			}

		}

		//开始内循环
		//初始化内循环的迭代变量w_hat
		
		//for (int j = 0; j < n_feature; ++j) {
		//	w_hat[j] = w[j];
		//}

		for (int idx = 0; idx < n_sample*2; ++idx)
		{

			int rand_i = distribution(generator);
			//计算相关的梯度估计
			vector<double> gradienEstimator(n_feature, 0.0);
			for (int j = 0; j < n_feature; ++j) {
				gradienEstimator[j] = mu[j] + lambda*(w_hat[j] - w[j]);
			}

			if (loss == "smooth_hinge") {
				//计算\phi在w_hat的导数
				double nabla_w_hat = 0.0;
				double linearPreditor = train_data.Y[rand_i] * dot_sparse(train_data, rand_i, w_hat);
				if (linearPreditor >= 1.0) {
					nabla_w_hat = 0.0;
				}
				else if (linearPreditor <= 0.0) {
					nabla_w_hat = -train_data.Y[rand_i];
				}
				else {
					nabla_w_hat = linearPreditor - 1.0;
				}

				//关于w计算\phi的导数
				double nabla_w = 0.0;
				double linearPreditor_w = train_data.Y[rand_i] * dot_sparse(train_data, rand_i, w);
				if (linearPreditor_w >= 1.0) {
					nabla_w = 0.0;
				}
				else if (linearPreditor_w <= 0.0) {
					nabla_w = -train_data.Y[rand_i];
				}
				else {
					nabla_w = linearPreditor_w - 1.0;
				}

				//更新梯度估计
				for (int k = train_data.index[rand_i]; k < train_data.index[rand_i + 1]; ++k) {
					gradienEstimator[train_data.col[k]] += (nabla_w_hat - nabla_w)*train_data.X[k];
				}

			}
			else {
				std::cout << "Error in dual_svm.cpp/fit_SVRG(): Not available loss type!!" << std::endl;
			}

			//更新w_hat的值
			for (int j = 0; j < n_feature; ++j) {
				w_hat[j] = w_hat[j] - eta*gradienEstimator[j];
			}

		}//end for-idx
		
		//更新每个epoch的输出值
		for (int j = 0; j < n_feature; ++j) {
			w[j] = w_hat[j];
		}
		 
		double primal_val = calculate_primal(train_data);
		primal_val_array.push_back(primal_val);

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val << std::endl;
		}
	}//end for-epoch
}

//****************************************************************************************

//****************************************************************************************ASPDC

void dual_svm::fit_ASPDC(Data& train_data) {
	std::cout << "invoking ASPDC algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample = " << n_sample << std::endl
		<< "  n_feature + 1 = " << n_feature << std::endl
		<< " lambda = " << lambda << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	//初始化变量w,alpha，u
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	//vector<double> w_old(n_feature, 0.0);////for momentum加速

	//double _theta = 0.9;
	//std::cout << "theta: " << _theta << std::endl;

	double gamma = 1.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;//wrong
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Not available loss type!! in dual_svm.cpp/SPDC_r7" << std::endl;
	}

	std::cout << "lambda*n_sample = " << lambda*n_sample << std::endl;

	if (lambda < 4 / (n_sample*gamma)) {
		////是否满足调用条件
		std::cout << "Error in dual_svm.cpp/SPDC_r7(): the lambda is not large enough!" << std::endl;
		return;
	}

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		////use to momentum 
		//for (int j = 0; j < n_feature; ++j) {
		//	w_old[j] = w[j];
		//}

		double dual_gap = 0.0;
		for (int idx = 0; idx < n_sample; ++idx)
		{//遍历一轮数据
			int rand_id = distribution(generator);
			////更新alpha[rand_id]
			double delta = 0.0;
			if (loss == "L2_svm") {
				//精确求解
				double temp = 2.0*train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w) - train_data.Y[rand_id] * alpha[rand_id] - 2;
				delta = train_data.Y[rand_id] * min(-train_data.Y[rand_id] * alpha[rand_id], temp);
			}
			else if (loss == "smooth_hinge") {
				double temp = train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w) - 1;
				delta = train_data.Y[rand_id] * max(-1.0, min(0.0, temp)) - alpha[rand_id];
			}
			else {
				std::cout << "Not available loss type!!" << std::endl;
			}
			alpha[rand_id] = alpha[rand_id] + delta;

			//for (int j = 0; j < n_feature; ++j) {
			//	w_old[j] = w[j];
			//}

			////更新w
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {
				w[train_data.col[j]] -= delta*train_data.X[j] / (double)(lambda*n_sample);
				////for debug
				//double temp = delta*train_data.X[j] / (double)(lambda*n_sample);
				//w[train_data.col[j]] -= temp;
				////for debug
			}

			//for (int j = 0; j < n_feature; ++j) {
			//	w[j] = w[j] + _theta*(w[j] - w_old[j]);
			//}

		}////for-idx

		 ////momentum
		 //for (int j = 0; j < n_feature; ++j) {
		 //	w[j] = w[j] + _theta*(w[j] - w_old[j]);
		 //}

		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual_ASPDC(train_data);
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(dual_gap);

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(15) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}


	}// for-epoch

}

//****************************************************************************************ASPDCi
////SPDC_r8与SPDC_r7的对偶函数值计算方式是一样的

void dual_svm::fit_ASPDCi(Data& train_data) {
	std::cout << "invoking ASPDC-s algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample = " << n_sample << "  n_feature + 1 = " << n_feature << endl;
	std::cout << " lambda = " << lambda << std::endl;

	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);////外层迭代变量
	vector<double> w_prox(n_feature, 0.0);////epoch内的迭代变量

	double gamma = 1.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;//wrong
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Not available loss type!! in dual_svm.cpp/SPDC_r8" << std::endl;
	}

	////基本参数
	double kappa = 4.0 / (n_sample*gamma) - lambda;////理论分析的下界值
	double mu = lambda / 2.0;
	double rho = mu + kappa;
	double eta = sqrt(mu / rho);
	double beta = (1.0 - eta) / (1.0 + eta);

	std::cout << " kappa = " << kappa << std::endl
		<< "mu = " << mu << std::endl
		<< "rho = " << rho << std::endl
		<< "eta = " << eta << std::endl
		<< "beta = " << beta << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl
		<< "(lambda+kappa)*n_sample*gamma = " << (lambda + kappa)*n_sample*gamma << std::endl;

	if (lambda > 4 / (n_sample*gamma)) {
		////是否满足调用条件
		std::cout << "Error in dual_svm.cpp/ASPDC(): the lambda is not small enough!" << std::endl;
		return;
	}

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;

		////initialize w at the begin of every stage(epoch)
		for (int j = 0; j < n_feature; ++j) {
			w_prox[j] = w[j] * kappa / (lambda + kappa);////相当于一次初始化
		}
		for (int i = 0; i < n_sample; ++i) {//遍历一次所有数据
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//遍历一个数据,嵌套的时候小心别混淆索引
				w_prox[train_data.col[j]] -= alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));
				////复合赋值操作，小心别写成直接赋值了.另外一定要小心,与SDCA，ASDCA有细微差别，是-=，而不是+=
			}
		}

		//begin stage(epoch)
		for (int idx = 0; idx < n_sample * 2; ++idx)
		{ ////m=5n
			int rand_id = distribution(generator);
			////更新alpha[rand_id]
			////alpha的更新其实和SPDC_r1，SPDC_r0是一样的
			double delta = 0.0;
			if (loss == "L2_svm") {
				//精确求解
				double temp = 2.0*train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox) - train_data.Y[rand_id] * alpha[rand_id] - 2;
				delta = train_data.Y[rand_id] * min(-train_data.Y[rand_id] * alpha[rand_id], temp);
			}
			else if (loss == "smooth_hinge") {
				double temp = train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox) - 1;
				delta = train_data.Y[rand_id] * max(-1.0, min(0.0, temp)) - alpha[rand_id];
			}
			else {
				std::cout << "Not available loss type!!" << std::endl;
			}
			alpha[rand_id] = alpha[rand_id] + delta;

			////更新w
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {////遍历一个样本
				w_prox[train_data.col[j]] -= delta*train_data.X[j] / (double)((lambda + kappa)*n_sample);
			}

		}//for-idx

		 //////momentum
		 //for (int j = 0; j < n_feature; ++j) {
		 //	w[j] = w_prox[j] + beta*(w_prox[j] - w[j]);
		 //}

		 // ////no momentum
		for (int j = 0; j < n_feature; ++j) {
			w[j] = w_prox[j];
		}

		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual_ASPDC(train_data);////应该调用哪一个函数呢
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(abs(dual_gap));

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(20) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}


	}// for-epoch

}

//****************************************************************************************Catalyst

void dual_svm::fit_Catalyst(Data& train_data) {
	////Catalyst + SDCA
	std::cout << "invoking Catalyst algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//特征维度已经扩展
	std::cout << "n_sample = " << n_sample << "  n_feature + 1 = " << n_feature << endl;
	std::cout << " lambda = " << lambda << std::endl;

	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);////外层迭代变量
	vector<double> w_prox(n_feature, 0.0);////epoch内的迭代变量

	double gamma = 1.0;
	if (loss == "L2_svm") {
		//gamma = 2.0;//wrong
		gamma = 0.5;
	}
	else if (loss == "smooth_hinge") {
		gamma = 1.0;
	}
	else {
		std::cout << "Not available loss type!! in dual_svm.cpp/SPDC_r8" << std::endl;
	}

	////基本参数
	double kappa = 4.0 / (n_sample*gamma) - lambda;////理论分析的下界值
	double q = lambda / (lambda + kappa);
	double beta = 1 - 2.0*sqrt(q) / (1 + sqrt(q));

	std::cout << " kappa = " << kappa << std::endl
		<< "q = " << q << std::endl
		<< "beta = " << beta << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl
		<< "(lambda+kappa)*n_sample*gamma = " << (lambda + kappa)*n_sample*gamma << std::endl;

	if (lambda > 4 / (n_sample*gamma)) {
		////是否满足调用条件
		std::cout << "Error in dual_svm.cpp/ASPDC(): the lambda is not small enough!" << std::endl;
		return;
	}

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;

		////initialize w at the begin of every stage(epoch)
		for (int j = 0; j < n_feature; ++j) {
			w_prox[j] = w[j] * kappa / (lambda + kappa);////相当于一次初始化
		}
		for (int i = 0; i < n_sample; ++i) {//遍历一次所有数据
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//遍历一个数据,嵌套的时候小心别混淆索引
				w_prox[train_data.col[j]] -= alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));
				////复合赋值操作，小心别写成直接赋值了.另外一定要小心,与SDCA，ASDCA有细微差别，是-=，而不是+=
			}
		}

		//begin stage(epoch)
		for (int idx = 0; idx < n_sample * 5 * 2; ++idx)
		{ ////m=5n
			int rand_id = distribution(generator);
			////更新alpha[rand_id]
			////alpha的更新其实和SPDC_r1，SPDC_r0是一样的
			double delta = 0.0;
			if (loss == "L2_svm") {
				//精确求解
				double temp = 2.0*train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox) - train_data.Y[rand_id] * alpha[rand_id] - 2;
				delta = train_data.Y[rand_id] * min(-train_data.Y[rand_id] * alpha[rand_id], temp);
			}
			else if (loss == "smooth_hinge") {
				double temp = train_data.Y[rand_id] * dot_sparse(train_data, rand_id, w_prox) - 1;
				delta = train_data.Y[rand_id] * max(-1.0, min(0.0, temp)) - alpha[rand_id];
			}
			else {
				std::cout << "Not available loss type!!" << std::endl;
			}
			alpha[rand_id] = alpha[rand_id] + delta;

			////更新w
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {////遍历一个样本
				w_prox[train_data.col[j]] -= delta*train_data.X[j] / (double)((lambda + kappa)*n_sample);
			}

		}//for-idx

		 //////momentum
		for (int j = 0; j < n_feature; ++j) {
			w[j] = w_prox[j] + beta*(w_prox[j] - w[j]);
		}

		// ////no momentum
		//for (int j = 0; j < n_feature; ++j) {
		//	w[j] = w_prox[j];
		//}

		double primal_val = calculate_primal(train_data);
		double dual_val = calculate_dual_ASPDC(train_data);////应该调用哪一个函数呢？
		dual_gap = primal_val - dual_val;

		primal_val_array.push_back(primal_val);
		dual_val_array.push_back(dual_val);
		dual_gap_array.push_back(abs(dual_gap));

		if (verbose) {
			cout << "epoch " << ": " << epoch
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(20) << primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

	}// for-epoch

}

//*********************************************************************************************

