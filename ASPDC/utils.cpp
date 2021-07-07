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
	if (k >= train_data.index.size() - 1) {//数组边界检查
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
	if (k >= train_data.index.size() - 1) {//数组边界检查
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
{	//在main函数中读完数据后，先进行nomalize,然后再训练 
	int n_sample = train_data.n_sample;
	vector<double> norm(n_sample, 0.0);
	for (int i = 0; i < n_sample; ++i) {
		norm[i] = norm_2_sparse(train_data, i);
	}
	for (int j = 0; j < n_sample; ++j) {//对每一个sample进行正则化
		for (int k = train_data.index[j]; k < train_data.index[j + 1]; ++k) {//遍历一个sample j
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

//read data from libsvm format，扩展特征维度，也放在本函数中；
//另外，index中，最后一个维度，设置index[n_sample+1]=X.size(),是为了方便编写循环
void read_libsvm(const string filename, Data &train_data) {
	//std::cout << "strat to read data of file: " << filename << std::endl;
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {//每一行的开头,也就是每一个sample的开始
			num_sample++;
			train_data.X.push_back(1.0);//每行（每个sample）的开始，同时也是上一行(上一个sample)的结束，直接在每一行的开头插入扩展的1
			train_data.col.push_back(0);//也就是说X[sample_i][0]=1，即在每一个样本的开始地方插入一个1
			train_data.index.push_back(train_data.X.size() - 1);//标记在样本i首个元素在X中的位置为此时的X.size()-1,其实就是指向新扩展值,注意这包含了X.size()=1时候
			train_data.Y.push_back(1);
		}
		else if (read_string == "-1" || read_string == "0") {////covtype中类是1,2,|| read_string == "2"
			num_sample++;
			train_data.X.push_back(1.0);
			train_data.col.push_back(0);
			train_data.index.push_back(train_data.X.size() - 1);
			train_data.Y.push_back(-1);
		}
		else {//每一行中的其他字段
			int colon = read_string.find(":");
			if (colon != -1) {
				int idx = atoi(read_string.substr(0, colon).c_str());//atoi在stdlib.h中
				double feature = stringToNum(read_string.substr(colon + 1));
				if (idx > num_feature) {
					num_feature = idx;
				}
				train_data.X.push_back(feature);
				train_data.col.push_back(idx);//part1获得的结果是从1开始计数的，注意在每一个sample(每一行的第一个位置插入了1)，所以这里就不用再减1了。			
			}
		}
	}//end while(fin)

	fin.close();
	train_data.index.push_back(train_data.X.size());//index的第一个元素是0，最后一个元素是X.size，标记最后一个sample最后一个元素的下一个位置。
	train_data.n_sample = num_sample;
	train_data.n_feature = num_feature + 1;//特征扩展

//////////////////////////////////////////////////////////////测试用
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
	////直接读入原始数据，不进行特征扩展
	//std::cout << "strat to read data of file: " << filename << std::endl;
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {//每一行的开头,也就是每一个sample的开始
			num_sample++;
			train_data.index.push_back(train_data.X.size());
			train_data.Y.push_back(1.0);
		}
		else if (read_string == "-1" || read_string == "0" || read_string == "2") {////covtype中类是1,2
			num_sample++;
			train_data.index.push_back(train_data.X.size());
			train_data.Y.push_back(-1.0);
		}
		else {//每一行中的其他字段
			int colon = read_string.find(":");
			if (colon != -1) {
				int idx = atoi(read_string.substr(0, colon).c_str());//atoi在stdlib.h中
				double feature = stringToNum(read_string.substr(colon + 1));
				if (idx > num_feature) {
					num_feature = idx;
				}
				train_data.X.push_back(feature);//从1开始计数
				train_data.col.push_back(idx);//part1获得的结果是从1开始计数的，注意在每一个sample(每一行的第一个位置插入了1)，所以这里就不用再减1了。			
			}
		}
	}//end while(fin)

	fin.close();
	train_data.index.push_back(train_data.X.size());//index的第一个元素是0，最后一个元素是X.size，标记最后一个sample最后一个元素的下一个位置。
	train_data.n_sample = num_sample;
	train_data.n_feature = num_feature;//特征不扩展

									   //////////////////////////////////////////////////////////////测试用
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
	int index = 0;////index遍历raw_data.X[],以及col[],也就是遍历所有非零元素
	for (int i = 0; i < raw_data.n_sample; ++i) {
		train_data.index.push_back(train_data.X.size());
		train_data.X.push_back(1.0);////在第一列进行维度扩展。
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
