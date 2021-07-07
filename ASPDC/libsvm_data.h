#ifndef  LIBSVM_DATA_H
#define  LIBSVM_DATA_H
#include <vector>

using namespace std;

class Data {
public:
	vector<double> X;//只存储非零值，按行存储
	vector<double> Y;
	vector<int> index;//维度等于n_sample+1,[idex[i],index[i+1])这个左闭右开区间表示sample Xi在数组X中的索引，注意index的最后一个维度是X的大小
	vector<int> col;//维度和X的相同，为非零个数，与X的元素一一对应，表明X中对应元素的列坐标。
	int n_sample;//n_samples
	int n_feature;//这个特征维度在read_libsvm()就已经扩展了
	Data() :n_sample(0), n_feature(0) {}
};

#endif // ! LIBSVM_DATA_H


