#ifndef NEURAL_H
#define NEURAL_H
#include<vector>
using namespace std;

#define INPUT_COUNT 3
#define HIDE_LAYERS_COUNT 1
#define OUTPUT_COUNT 1
#define E 2.71828
#define MAXTIMES 500
typedef double(*func)(double);

struct hidelayers{
	int neuron_amount;
	double b;
	double berror;
	vector<vector<double>> weight;
	vector<vector<double>> deltaweight;
	vector<double> error;
	vector<double> output;
	hidelayers(int amount) :neuron_amount(amount),berror(0){
		output = *(new vector<double>(neuron_amount+1));
		error = *(new vector<double>(neuron_amount+1));
	}
};


static double sigmond(double wx){
	return 1/(1+pow(E,-wx));
}
static double deltasigmond(double wx){
	double a = sigmond(wx);
	return a*(1 - a);
}
static double tanh(double wx){
	return 0.0;
}


class NeuralNet{
private:
	
	vector<vector<double>> input;                                                                                            //输入层x
	vector<vector<double>> output;                                                                                           //输出层y
	vector<hidelayers> hide;                                                                                         //隐藏层
	//vector<vector<double>> outputweight;                                                                             //最后一层权重
	//vector<double> b;                                                                                                //每层偏移b
	//vector<int> neurals;                                                                                             //隐藏层每层神经元数
	int input_count;                                                                                                 //输入单元数
	int hide_layers_count;                                                                                           //隐藏层数
	int output_count;                                                                                                //输出单元数
	
	void forhead(vector<double>&,int);
	void bp(int);
	void initializeWeight();
	func f;
	double J(double, double,int);
public:
	NeuralNet(vector<unsigned int>&neuralnum, func fun = sigmond, int inputcount = INPUT_COUNT, int hidelayer = HIDE_LAYERS_COUNT, int outputcount = OUTPUT_COUNT);
	void train(const vector<vector<double>> &, const vector<vector<double>> &);
	void test(const vector<vector<double>> &, vector<vector<double>> &,double);
	void weightvisual();
	~NeuralNet();
};


#endif
