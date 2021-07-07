#ifndef  LIBSVM_DATA_H
#define  LIBSVM_DATA_H
#include <vector>

using namespace std;

class Data {
public:
	vector<double> X;//ֻ�洢����ֵ�����д洢
	vector<double> Y;
	vector<int> index;//ά�ȵ���n_sample+1,[idex[i],index[i+1])�������ҿ������ʾsample Xi������X�е�������ע��index�����һ��ά����X�Ĵ�С
	vector<int> col;//ά�Ⱥ�X����ͬ��Ϊ�����������X��Ԫ��һһ��Ӧ������X�ж�ӦԪ�ص������ꡣ
	int n_sample;//n_samples
	int n_feature;//�������ά����read_libsvm()���Ѿ���չ��
	Data() :n_sample(0), n_feature(0) {}
};

#endif // ! LIBSVM_DATA_H


