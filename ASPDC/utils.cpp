#include"utils.h"

double dot_dense(const vector<double>& x)
{//w.dot(w)
	double ret = 0;
	for (int i = 0; i < x.size(); i++) {
		ret += x[i] * x[i];
	}
	return (ret);
}

double dot_dense(const vector<double>& x, const vector<double>& y)
{ // w1.dot(w2)
	if (x.size() != y.size()) {
		std::cout << "Error in utils.cpp/dot_dense(): dot operation must perform on vectors with the same size!" << std::endl;
		return 0.0;
	}
	double ret = 0;
	for (int i = 0; i < x.size(); i++) {
		ret += x[i] * y[i];
	}
	return (ret);
}

double norm_2_sparse(const Data& train_data, int k)
{	//sqrt(Xi.dot(Xi)),used in normalize
	double ret = 0;
	if (k >= train_data.index.size() - 1) {//����߽���
		std::cout << "Error in utils.cpp/norm_2_square: the index is out of range of train_data.index.size()!!!" << std::endl;
		return 0.0;
	}
	for (int i = train_data.index[k]; i < train_data.index[k + 1]; i++) {
		ret += train_data.X[i] * train_data.X[i];
	}
	return sqrt(ret);
}

double dot_sparse(const Data& train_data, const int k, const vector<double>& w)
{//Xk.dot(w)
	if (k >= train_data.index.size() - 1) {//����߽���
		std::cout << "Error in utils.cpp/dot_sparse: the index is out of range of train_data.index.size()!!!" << std::endl;
		return 0.0;
	}
	double ret = 0;
	for (int i = train_data.index[k]; i < train_data.index[k + 1]; i++) {
		ret += train_data.X[i] * w[train_data.col[i]];
	}
	return (ret);
}

void normalize_data(Data& train_data)
{	//��main�����ж������ݺ��Ƚ���nomalize,Ȼ����ѵ�� 
	int n_sample = train_data.n_sample;
	vector<double> norm(n_sample, 0.0);
	for (int i = 0; i < n_sample; ++i) {
		norm[i] = norm_2_sparse(train_data, i);
	}
	for (int j = 0; j < n_sample; ++j) {//��ÿһ��sample��������
		for (int k = train_data.index[j]; k < train_data.index[j + 1]; ++k) {//����һ��sample j
			train_data.X[k] /= norm[j];
		}
	}
}

inline double stringToNum(const string& str)
{	// only used in readLibsvm()function
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

//read data from libsvm format����չ����ά�ȣ�Ҳ���ڱ������У�
//���⣬index�У����һ��ά�ȣ�����index[n_sample+1]=X.size(),��Ϊ�˷����дѭ��
void read_libsvm(const string filename, Data &train_data) {
	//std::cout << "strat to read data of file: " << filename << std::endl;
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {//ÿһ�еĿ�ͷ,Ҳ����ÿһ��sample�Ŀ�ʼ
			num_sample++;
			train_data.X.push_back(1.0);//ÿ�У�ÿ��sample���Ŀ�ʼ��ͬʱҲ����һ��(��һ��sample)�Ľ�����ֱ����ÿһ�еĿ�ͷ������չ��1
			train_data.col.push_back(0);//Ҳ����˵X[sample_i][0]=1������ÿһ�������Ŀ�ʼ�ط�����һ��1
			train_data.index.push_back(train_data.X.size() - 1);//���������i�׸�Ԫ����X�е�λ��Ϊ��ʱ��X.size()-1,��ʵ����ָ������չֵ,ע���������X.size()=1ʱ��
			train_data.Y.push_back(1);
		}
		else if (read_string == "-1" || read_string == "0") {////covtype������1,2,|| read_string == "2"
			num_sample++;
			train_data.X.push_back(1.0);
			train_data.col.push_back(0);
			train_data.index.push_back(train_data.X.size() - 1);
			train_data.Y.push_back(-1);
		}
		else {//ÿһ���е������ֶ�
			int colon = read_string.find(":");
			if (colon != -1) {
				int idx = atoi(read_string.substr(0, colon).c_str());//atoi��stdlib.h��
				double feature = stringToNum(read_string.substr(colon + 1));
				if (idx > num_feature) {
					num_feature = idx;
				}
				train_data.X.push_back(feature);
				train_data.col.push_back(idx);//part1��õĽ���Ǵ�1��ʼ�����ģ�ע����ÿһ��sample(ÿһ�еĵ�һ��λ�ò�����1)����������Ͳ����ټ�1�ˡ�			
			}
		}
	}//end while(fin)

	fin.close();
	train_data.index.push_back(train_data.X.size());//index�ĵ�һ��Ԫ����0�����һ��Ԫ����X.size��������һ��sample���һ��Ԫ�ص���һ��λ�á�
	train_data.n_sample = num_sample;
	train_data.n_feature = num_feature + 1;//������չ

//////////////////////////////////////////////////////////////������
/*for (int i = 0; i < train_data.n_sample; i++) {
cout << " ";
if (train_data.Y[i] > 0) {
cout << "+";
}
cout << train_data.Y[i];
for (int j = train_data.index[i]; j < train_data.index[i + 1]; j++) {
cout << " " << train_data.col[j] + 1 << ":" << train_data.X[j];
}
}*/
}

double min(double a, double b) {
	return (a > b) ? b : a;
}
double max(double a, double b) {
	return (a > b) ? a : b;
}

//------------------------------------------------2018/7/19-------------------------------//
void read_libsvm_raw(const string filename, Data &train_data) {
	////ֱ�Ӷ���ԭʼ���ݣ�������������չ
	//std::cout << "strat to read data of file: " << filename << std::endl;
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {//ÿһ�еĿ�ͷ,Ҳ����ÿһ��sample�Ŀ�ʼ
			num_sample++;
			train_data.index.push_back(train_data.X.size());
			train_data.Y.push_back(1.0);
		}
		else if (read_string == "-1" || read_string == "0" || read_string == "2") {////covtype������1,2
			num_sample++;
			train_data.index.push_back(train_data.X.size());
			train_data.Y.push_back(-1.0);
		}
		else {//ÿһ���е������ֶ�
			int colon = read_string.find(":");
			if (colon != -1) {
				int idx = atoi(read_string.substr(0, colon).c_str());//atoi��stdlib.h��
				double feature = stringToNum(read_string.substr(colon + 1));
				if (idx > num_feature) {
					num_feature = idx;
				}
				train_data.X.push_back(feature);//��1��ʼ����
				train_data.col.push_back(idx);//part1��õĽ���Ǵ�1��ʼ�����ģ�ע����ÿһ��sample(ÿһ�еĵ�һ��λ�ò�����1)����������Ͳ����ټ�1�ˡ�			
			}
		}
	}//end while(fin)

	fin.close();
	train_data.index.push_back(train_data.X.size());//index�ĵ�һ��Ԫ����0�����һ��Ԫ����X.size��������һ��sample���һ��Ԫ�ص���һ��λ�á�
	train_data.n_sample = num_sample;
	train_data.n_feature = num_feature;//��������չ

									   //////////////////////////////////////////////////////////////������
									   /*for (int i = 0; i < train_data.n_sample; i++) {
									   cout << " ";
									   if (train_data.Y[i] > 0) {
									   cout << "+";
									   }
									   cout << train_data.Y[i];
									   for (int j = train_data.index[i]; j < train_data.index[i + 1]; j++) {
									   cout << " " << train_data.col[j] + 1 << ":" << train_data.X[j];
									   }
									   }*/
}

void add_bias(Data& raw_data, Data& train_data) {
	int index = 0;////index����raw_data.X[],�Լ�col[],Ҳ���Ǳ������з���Ԫ��
	for (int i = 0; i < raw_data.n_sample; ++i) {
		train_data.index.push_back(train_data.X.size());
		train_data.X.push_back(1.0);////�ڵ�һ�н���ά����չ��
		train_data.col.push_back(0);
		for (int j = raw_data.index[i]; j < raw_data.index[i + 1]; ++j) {
			train_data.X.push_back(raw_data.X[index]);
			train_data.col.push_back(raw_data.col[index]);
			++index;
		}
	}
	train_data.index.push_back(train_data.X.size());
	train_data.n_feature = raw_data.n_feature + 1;
	train_data.n_sample = raw_data.n_sample;
	for (int k = 0; k < raw_data.n_sample; ++k) {
		train_data.Y.push_back(raw_data.Y[k]);
	}

}
