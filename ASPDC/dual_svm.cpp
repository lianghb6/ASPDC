#include"dual_svm.h"
#include<float.h>//DBL_MAX DBL_MIN

////�ڼ���dual value �ϣ�SDCA��ASDCA����һ������SPDC��FSPDC������һ��
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
	//double regularizer = 0.5*lambda*dot_dense(w);//�޸�Ϊ��alpha�������ʲôӰ��ô������ʱ��

	vector<double> regularizer(train_data.n_feature, 0.0);
	for (int i = 0; i < train_data.n_sample; ++i) {//����һ����������
		for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//����һ������,Ƕ�׵�ʱ��С�ı��������
			regularizer[train_data.col[j]] += alpha[i] * train_data.X[j] / ((double)train_data.n_sample*lambda);
			////���ϸ�ֵ������С�ı�д��ֱ�Ӹ�ֵ��.����һ��ҪС��,��SPDC,ASPDC��ϸ΢�����+=��������-=
		}
	}

	double sum = 0.0;
	if (loss == "hinge") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			sum += alpha[i] * train_data.Y[i];//+=��//ע����SPDC�ĺ�ASPDC�Ĳ�ͬ
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
			sum = sum - alpha[i] * train_data.Y[i] - alpha[i] * alpha[i] / 4.0;//ע�������SDCA��ASDCA�Ĳ�һ��
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
	////��ASDCA��lemma3֤���У�D(alpha)�����������ʽ�������ASDCA�㷨�еı���һ�¡�
	////ΪʲôҪ��������һ�������ż����ֵ�ĺ����أ���ΪSPDC_r7��Щ���⣬������SDCA������SPDC.���ԣ�ֻ�ø���дһ����ż�����ļ���ʽ�ӡ�

	//double regularizer = 0.5*lambda*dot_dense(w);
	//Ҫ���ΪʲôҪ�޸ģ���Ϊ��ÿһ��epoch�У����Ƕ���AccSPDCȥ���һ�����Ƶ�primal-dual���⡣w_prox��Ϊprimal������alpha��Ϊdual������
	//��w����Ϊ���epoch��primal������Ҳǡǡ��primalԭ����ı�����
	//��������Ҫ��ģ�������Ϊ����primal-dual ��dual value������ԭʼdual value��ֵ������һ��Ҫ�������
	//������ʵ�����SPDC��dual value�ļ��㺯��ֵ��ʽ��һ���ģ�������SPDC�У���������Uֱ���������

	vector<double> regularizer(train_data.n_feature, 0.0);
	for (int i = 0; i < train_data.n_sample; ++i) {//����һ����������
		for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//����һ������,Ƕ�׵�ʱ��С�ı��������
			regularizer[train_data.col[j]] += -1.0*alpha[i] * train_data.X[j] / ((double)train_data.n_sample*lambda);
			////���ϸ�ֵ������С�ı�д��ֱ�Ӹ�ֵ��.����һ��ҪС��,��SDCA��ASDCA��ϸ΢�����-=��������+=
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

//****************************************************************************************SDCA�㷨

void dual_svm::fit_SDCA(Data& train_data) {
	std::cout << " invoking SDCA algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << "n_sample  = " << n_sample << std::endl
		<< "  n_feature + 1 = " << n_feature << endl
		<< " lambda =  " << lambda << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	//��ʼ������w,alpha
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);

	//normalize_data(train_data);

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);//std::uniform_int_distribution<size_t> distribution(0, n_sample - 1)ò��������

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;
		for (int idx = 0; idx < n_sample; ++idx)//idxֻ�ڱ��г���
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
			//w�ĸ���,ע����+=,����SPDC�Ĳ�һ����
			for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
				//����һ��sample rand_id
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

//****************************************************************************************ASDCA_v1�㷨

////ASDCA�ļ򻯰汾,������ֹ����

void dual_svm::fit_ASDCA_v1(Data& train_data) {
	std::cout << " invoking ASDCA_v1 algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
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

	////ASCDA��������
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
		return;////ASDCA����������,��lambda�㹻С��ʱ��ŵ���ASDCAss
	}

	//��ʼ������w,alpha
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	vector<double> w_prox(n_feature, 0.0);//��ÿ��epoch�����������Ϊ��������

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;

		////ÿ��epoch֮ǰ��ʼ��w_prox
		for (int j = 0; j < n_feature; ++j) {
			w_prox[j] = kappa*w[j] / (lambda + kappa);//��ֵ��ʼ��
		}
		for (int i = 0; i < n_sample; ++i) {//����һ����������
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//����һ������,Ƕ�׵�ʱ��С�ı��������
				w_prox[train_data.col[j]] += alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));//���ϸ�ֵ������С�ı�д��ֱ�Ӹ�ֵ�ˣ������ASPDCҲ�е�����
			}
		}
		////��ʼ������

		for (int idx = 0; idx < n_sample * 2; ++idx) {
			////����ASDCA���˳�����,�����forѭ���ڣ�w_prox(������w)�ǵ�������
			int rand_id = distribution(generator);
			double d = 0.0;
			////alpha�ĸ��º�SDCA����,ֻ����Ҫ�޸�lambda������
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
			//w_prox�ĸ���
			for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
				//����һ��sample rand_id��ע������ĸ�����+=�����SPDC�Ĳ�һ��
				w_prox[train_data.col[k]] += d* train_data.X[k] / ((double)n_sample*(lambda + kappa));
			}
		}////end for-idx

		 // //momentum
		for (int j = 0; j < n_feature; ++j) {
			w[j] = w_prox[j] + beta*(w_prox[j] - w[j]);
		}

		//////no momentum,ʵ�������ASDCA��û��momentum���ӿ�
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

//****************************************************************************************ASDCA�㷨

void dual_svm::fit_ASDCA(Data& train_data) {
	////����������ASDCA���ۺϿ�����ÿ��epoch��ֹͣ�����Լ������㷨������ֹͣ������
	std::cout << " invoking ASDCA algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
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

	////��ʼ������w,alpha
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	vector<double> w_prox(n_feature, 0.0);////��ÿ��epoch�������������Ϊ��������

	vector<double> delta_wy(n_feature, 0.0);////only use in stopping condition 2

											////ASCDA��������
	double kappa = 1.0 / (gamma*n_sample) - lambda;
	double mu = lambda / 2.0;
	double rho = mu + kappa;
	double eta = sqrt(mu / rho);
	double beta = (1.0 - eta) / (1.0 + eta);
	double _xi = (1 + 1.0 / (eta*eta))*(calculate_primal(train_data) - calculate_dual(train_data));
	////_xi�غ������㷨ֹͣ����
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
		return;////ASDCA����������,��lambda�㹻С��ʱ��ŵ���ASDCA
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

		//---------------------------------------------------------------ÿһ��while���ǽ��һ��point proximal ������
		////ÿ��epoch֮ǰ��ʼ��w_prox
		for (int j = 0; j < n_feature; ++j) {
			w_prox[j] = kappa*w[j] / (lambda + kappa);//��ֵ��ʼ��
		}
		for (int i = 0; i < n_sample; ++i) {//����һ����������
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//����һ������,Ƕ�׵�ʱ��С�ı��������
				w_prox[train_data.col[j]] += alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));//���ϸ�ֵ������С�ı�д��ֱ�Ӹ�ֵ�ˣ������ASPDCҲ�е�����
			}
		}
		////��ʼ������
		////epoch֮ǰ���Ⱦ���epsilon��ֵ
		double epsilon = _xi*0.5*eta / (1 + 1.0 / (eta*eta));
		std::cout << "epsilon = " << epsilon << std::endl;
		double eps = 0.0;////�����洢proximal problem ���һ��proximal dual gap,for stopping condition 2 only.

		for (int cnt = 0; cnt < 10; ++cnt) {
			////���һ��proximal point���⣬�����ǰ�ﵽָ�����Ⱦ�ֹͣ��
			////�����ѭ���ڣ�w�ǳ�����w_prox����proximal point ����
			////��ѭ����cnt*n�ĸ��Ӷ�
			double prox_primal = 0.0;
			double prox_dual = 0.0;
			double prox_dual_gap = 0.0;
			////����proximal primal value
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

			////����proximal dual value
			sum = 0.0;////reset
			if (loss == "hinge") {
				for (int i = 0; i < train_data.n_sample; ++i) {
					sum += alpha[i] * train_data.Y[i];//+=��//ע����SPDC�ĺ�ASPDC�Ĳ�ͬ
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
				break;////����ﵽ�˸������ȣ���ֹͣ
			}

			////���û�ﵽ�������ȣ��ͼ���ִ��n�μ���,���proximal problem���Եġ�
			for (int idx = 0; idx < n_sample; ++idx) {
				int rand_id = distribution(generator);
				double d = 0.0;
				////alpha�ĸ��º�SDCA����,ֻ����Ҫ�޸�lambda������
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
				//w_prox�ĸ���
				for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
					//����һ��sample rand_id��ע������ĸ�����+=�����SPDC�Ĳ�һ��
					w_prox[train_data.col[k]] += d* train_data.X[k] / ((double)n_sample*(lambda + kappa));
				}

			}////end for-idx

		}////end for(cnt)

		 //---------------------------------------------------------------------------------proximal ����������

		 ////stopping condition 2,Ϊ�˷��㣬���������֮ǰ������
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

		//////no momentum,ʵ�������ASDCA��û��momentum���ӿ�
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

//****************************************************************************************SPDC�㷨

void dual_svm::fit_SPDC(Data& train_data) {
	std::cout << "invoking SPDC algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << "n_sample  " << n_sample << "  n_feature + 1 " << n_feature << endl
		<< " lambda" << lambda << std::endl;

	//��ʼ������w,alpha��u
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	u = vector<double>(n_feature, 0.0);//Ϊ�˼����ż����ֵ��������Ϊ˽�г�Ա��

	vector<double> w_bar(n_feature, 0.0);//ʽ��(8)�е�momentum���ٱ���
	vector<double> w_old(n_feature, 0.0);//ʽ�ӣ�8����ʵ����Ҫ����w����һ������ֵ

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

	std::random_device rd;//random_device �ඨ��ĺ��������������������Ϊ���ӵ�������޷�������ֵ��ÿһ�� rd() ���ö��᷵�ز�ͬ��ֵ
	std::default_random_engine generator(rd());//���������������������rd()��������ӡ�
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);//�����ֲ�����

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;
		for (int idx = 0; idx < n_sample; ++idx)
		{
			int rand_id = distribution(generator);//���������
			//����alpha[rand_id]
			double res = 0.0;
			if (loss == "L2_svm") {
				res = alpha[rand_id] + _sigma*(dot_sparse(train_data, rand_id, w_bar) - train_data.Y[rand_id]);//��w_bar��ֵȥ����y
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

			//����w
			for (int i = 0; i < n_feature; i++) {
				w_old[i] = w[i];//����momentumʵ���С�
			}

			vector<double> u_temp(n_feature, 0.0);
			for (int k = 0; k < n_feature; ++k) {
				u_temp[k] = u[k];//�൱��ÿ��epochҲ���³�ʼ��
			}
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {
				u_temp[train_data.col[j]] += delta*train_data.X[j];
			}

			for (int i = 0; i < n_feature; i++) {
				//����ʽ�ӣ�6�����м���
				w[i] = (w[i] - _tau*u_temp[i]) / (1.0 + lambda*_tau);//����w��ʱ����Ҫ��w_bar��ֱ����w�ľ���
			}

			//����u
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {
				u[train_data.col[j]] += delta*train_data.X[j] / (double)n_sample;
			}

			//extraction
			for (int i = 0; i < n_feature; i++) {
				w_bar[i] = w[i] + _theta*(w[i] - w_old[i]);//w_bar�ĸ���
			}
			//һ��SPDC��������

		}//end for-idx

		 //ע��ÿһ��epoch�������w������w_bar
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
//	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
//	std::cout << "n_sample  " << n_sample << "  n_feature + 1 " << n_feature << std::endl
//		<< " lambda" << lambda << std::endl;
//
//	//��ʼ������w,alpha��u
//	alpha = vector<double>(n_sample, 0.0);//�������ÿ�ε���ֵ��ƽ��
//	w = vector<double>(n_feature, 0.0);//�������ÿ�ε�����ƽ��ֵ
//
//	vector<double> w_hat(n_feature, 0.0);//��������
//	vector<double> alpha_hat(n_sample, 0.0);//��������
//
//	vector<double> w_sum(n_feature, 0.0);//�ۻ��ĵ�������
//	vector<double> alpha_sum(n_sample, 0.0);//�ۻ��ĵ���������
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
//	std::random_device rd;//�������������
//
//	std::default_random_engine generator_dual(rd());//�����������
//	std::uniform_int_distribution<int> distribution_dual(0, n_sample - 1);//�ֲ�����
//
//	std::default_random_engine generator_primal(rd());//�����������
//	std::uniform_int_distribution<int> distribution_primal(0, n_feature - 1);//�ֲ�����
//
//	for (int t = 0; t < n_sample*n_feature*(n_epoch); ++t) 
//	{
//		//���ѡ��������ͬ��i,j
//		int rand_i = distribution_dual(generator_dual);//�������[0,n-1]������
//		int rand_j = distribution_primal(generator_primal);//�������[0,d-1]������
//
//		double tau = 2.0 / (lambda * (double)(4 + t));
//		double eta = 2.0 * n_sample*n_feature / (gamma * (double)(t + 4));
//
//		//�ҵ�XijԪ��
//		double Xij = 0.0;//�п���Xij=0�İ�
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
//			//����w_hat
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
//			//ע���Ǹ�ֵ��������+=
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
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << "invoking SPD1 algorithm..." << std::endl
		<< "n_sample  " << n_sample << "  n_feature + 1 " << n_feature << std::endl
		<< " lambda" << lambda << std::endl;

	//��ʼ������w,alpha��u
	alpha = vector<double>(n_sample, 0.0);//�����������ÿ�ε���ֵ��ƽ��
	w = vector<double>(n_feature, 0.0);//�����������ÿ�ε�����ƽ��ֵ
	
	vector<double> w_hat(n_feature, 0.0);//��������
	vector<double> alpha_hat(n_sample, 0.0);//��������
	for (int k = 0; k < n_sample; ++k) {
		if (train_data.Y[k] == 1)
			alpha_hat[k] = -1.0;
		else if (train_data.Y[k] == -1)
			alpha_hat[k] = 1.0;
		else
			std::cout << "Error in SPD1: wrong label." << std::endl;
	}

	vector<double> w_sum(n_feature, 0.0);//�ۻ��ĵ�������
	vector<double> alpha_sum(n_sample, 0.0);//�ۻ��ĵ���������


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

	std::random_device rd;//�������������

	std::default_random_engine generator_dual(rd());//�����������
	std::uniform_int_distribution<int> distribution_dual(0, n_sample - 1);//�ֲ�����

	std::default_random_engine generator_primal(rd());//�����������
	std::uniform_int_distribution<int> distribution_primal(0, n_feature - 1);//�ֲ�����

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		
		for (int idx = 0; idx < n_sample*n_feature; ++idx)
		{
			//����SPD1��˵��һ��epoch����һ�����ݼ����൱��n*d�θ���
			//ȷ������
			int t = n_sample*n_feature*epoch + idx;//��ǰ��������
			double eta = 2.0 / (lambda * (double)(4 + t));
			double tau = 2.0 * n_sample*n_feature / (gamma * (double)(t + 4));

			//���ѡ��������ͬ��i,j
			int rand_i = distribution_dual(generator_dual);//�������[0,n-1]������
			int rand_j = distribution_primal(generator_primal);//�������[0,d-1]������

			//����primal��������w_hat,�Լ�//����dual����alpha_hat
			//�ҵ�XijԪ��
			double Xij = 0.0;//�п���Xij=0�İ�
			for (int k = train_data.index[rand_i]; k < train_data.index[rand_i + 1]; ++k) {
				if (train_data.col[k] == rand_j) {
					Xij = train_data.X[k];
					break;
				}
			}

			//if (Xij == 0)
			//	continue;//xij=0ʱ��Ҳ��Ҫ���µ�

			if (loss == "smooth_hinge") {
				//����w_hat
				w_hat[rand_j] = (w_hat[rand_j] - eta*Xij*alpha_hat[rand_i]) / (1.0 + lambda*eta);
				//alpha
				double temp = train_data.Y[rand_i] * (alpha_hat[rand_i] + tau*Xij*w_hat[rand_j] - train_data.Y[rand_i] * tau / (double)n_feature);//�㷨SPD1��ܹ�ʽ
				temp = temp / (1 + tau / (double)n_feature);
				alpha_hat[rand_i] = temp;
				//alpha_hat[rand_i] = train_data.Y[rand_i] * max(min(temp, 0.0), -1.0);//����˵Ӧ������Ҫ�����Լ���İ�,���Ǽ���֮��һֱ��������
			}
			else {
				std::cout << "Error in dual_svm.cpp/fit_SPD1(): Not available loss type!!" << std::endl;
			}
	
			//һ��SPD1��������

			//�����ۻ���w_sum��alpha_sum
			w_sum[rand_j] += w_hat[rand_j];
			alpha_sum[rand_i] += alpha_hat[rand_i];

		}//end for-idx


		////ע���Ǹ�ֵ��������+=
		double t = (epoch+1)*n_sample*n_feature;
		for (int k = 0; k < n_feature; ++k) {
			w[k] = w_sum[k]/t;
		}

		for (int k = 0; k < n_sample; ++k) {
			alpha[k] = alpha_sum[k]/t;
		}

		//ȡ���һ�ε����ǲ������ġ�
		//for (int k = 0; k < n_feature; ++k) {
		//	w[k] = w_hat[k] ;
		//}

		//for (int k = 0; k < n_sample; ++k) {
		//	alpha[k] = alpha_hat[k];
		//}

		 //ע��ÿһ��epoch�������w������w_hat
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

	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << " invoking SVRG algorithm..." << std::endl
		<< "n_sample  = " << n_sample << std::endl
		<< "  n_feature + 1 = " << n_feature << endl
		<< " lambda =  " << lambda << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	//��ʼ������w
	w = vector<double>(n_feature, 0.0); //��������ͬʱҲ���������
	vector<double> w_hat(n_feature, 0.0);//��ѭ����������

	//SVRG�Ĳ���
	double gamma = 0.0;//smooth����
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

	double eta = 0.1/ gamma;//��������
	//double eta = 0.01;

	std::cout << eta << std::endl;
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);//std::uniform_int_distribution<size_t> distribution(0, n_sample - 1)ò��������

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		//������w�����ݶ�ֵmu
		vector<double> mu(n_feature, 0.0);//snapshot���ݶ�ֵ
		for (int j = 0; j < n_feature; ++j) {
			mu[j] = lambda* w[j];
		}
		//����һ������
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

		//��ʼ��ѭ��
		//��ʼ����ѭ���ĵ�������w_hat
		
		//for (int j = 0; j < n_feature; ++j) {
		//	w_hat[j] = w[j];
		//}

		for (int idx = 0; idx < n_sample*2; ++idx)
		{

			int rand_i = distribution(generator);
			//������ص��ݶȹ���
			vector<double> gradienEstimator(n_feature, 0.0);
			for (int j = 0; j < n_feature; ++j) {
				gradienEstimator[j] = mu[j] + lambda*(w_hat[j] - w[j]);
			}

			if (loss == "smooth_hinge") {
				//����\phi��w_hat�ĵ���
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

				//����w����\phi�ĵ���
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

				//�����ݶȹ���
				for (int k = train_data.index[rand_i]; k < train_data.index[rand_i + 1]; ++k) {
					gradienEstimator[train_data.col[k]] += (nabla_w_hat - nabla_w)*train_data.X[k];
				}

			}
			else {
				std::cout << "Error in dual_svm.cpp/fit_SVRG(): Not available loss type!!" << std::endl;
			}

			//����w_hat��ֵ
			for (int j = 0; j < n_feature; ++j) {
				w_hat[j] = w_hat[j] - eta*gradienEstimator[j];
			}

		}//end for-idx
		
		//����ÿ��epoch�����ֵ
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
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << "n_sample = " << n_sample << std::endl
		<< "  n_feature + 1 = " << n_feature << std::endl
		<< " lambda = " << lambda << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl;

	//��ʼ������w,alpha��u
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	//vector<double> w_old(n_feature, 0.0);////for momentum����

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
		////�Ƿ������������
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
		{//����һ������
			int rand_id = distribution(generator);
			////����alpha[rand_id]
			double delta = 0.0;
			if (loss == "L2_svm") {
				//��ȷ���
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

			////����w
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
////SPDC_r8��SPDC_r7�Ķ�ż����ֵ���㷽ʽ��һ����

void dual_svm::fit_ASPDCi(Data& train_data) {
	std::cout << "invoking ASPDC-s algorithm..." << std::endl;
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << "n_sample = " << n_sample << "  n_feature + 1 = " << n_feature << endl;
	std::cout << " lambda = " << lambda << std::endl;

	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);////����������
	vector<double> w_prox(n_feature, 0.0);////epoch�ڵĵ�������

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

	////��������
	double kappa = 4.0 / (n_sample*gamma) - lambda;////���۷������½�ֵ
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
		////�Ƿ������������
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
			w_prox[j] = w[j] * kappa / (lambda + kappa);////�൱��һ�γ�ʼ��
		}
		for (int i = 0; i < n_sample; ++i) {//����һ����������
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//����һ������,Ƕ�׵�ʱ��С�ı��������
				w_prox[train_data.col[j]] -= alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));
				////���ϸ�ֵ������С�ı�д��ֱ�Ӹ�ֵ��.����һ��ҪС��,��SDCA��ASDCA��ϸ΢�����-=��������+=
			}
		}

		//begin stage(epoch)
		for (int idx = 0; idx < n_sample * 2; ++idx)
		{ ////m=5n
			int rand_id = distribution(generator);
			////����alpha[rand_id]
			////alpha�ĸ�����ʵ��SPDC_r1��SPDC_r0��һ����
			double delta = 0.0;
			if (loss == "L2_svm") {
				//��ȷ���
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

			////����w
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {////����һ������
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
		double dual_val = calculate_dual_ASPDC(train_data);////Ӧ�õ�����һ��������
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
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;//����ά���Ѿ���չ
	std::cout << "n_sample = " << n_sample << "  n_feature + 1 = " << n_feature << endl;
	std::cout << " lambda = " << lambda << std::endl;

	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);////����������
	vector<double> w_prox(n_feature, 0.0);////epoch�ڵĵ�������

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

	////��������
	double kappa = 4.0 / (n_sample*gamma) - lambda;////���۷������½�ֵ
	double q = lambda / (lambda + kappa);
	double beta = 1 - 2.0*sqrt(q) / (1 + sqrt(q));

	std::cout << " kappa = " << kappa << std::endl
		<< "q = " << q << std::endl
		<< "beta = " << beta << std::endl
		<< "lambda*n_sample = " << lambda*n_sample << std::endl
		<< "(lambda+kappa)*n_sample*gamma = " << (lambda + kappa)*n_sample*gamma << std::endl;

	if (lambda > 4 / (n_sample*gamma)) {
		////�Ƿ������������
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
			w_prox[j] = w[j] * kappa / (lambda + kappa);////�൱��һ�γ�ʼ��
		}
		for (int i = 0; i < n_sample; ++i) {//����һ����������
			for (int j = train_data.index[i]; j < train_data.index[i + 1]; ++j) {//����һ������,Ƕ�׵�ʱ��С�ı��������
				w_prox[train_data.col[j]] -= alpha[i] * train_data.X[j] / ((double)n_sample*(lambda + kappa));
				////���ϸ�ֵ������С�ı�д��ֱ�Ӹ�ֵ��.����һ��ҪС��,��SDCA��ASDCA��ϸ΢�����-=��������+=
			}
		}

		//begin stage(epoch)
		for (int idx = 0; idx < n_sample * 5 * 2; ++idx)
		{ ////m=5n
			int rand_id = distribution(generator);
			////����alpha[rand_id]
			////alpha�ĸ�����ʵ��SPDC_r1��SPDC_r0��һ����
			double delta = 0.0;
			if (loss == "L2_svm") {
				//��ȷ���
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

			////����w
			for (int j = train_data.index[rand_id]; j < train_data.index[rand_id + 1]; j++) {////����һ������
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
		double dual_val = calculate_dual_ASPDC(train_data);////Ӧ�õ�����һ�������أ�
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

