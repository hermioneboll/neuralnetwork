#include"stdafx.h"
#include"neural.h"
#include<stdlib.h>
#include<time.h>
#include<sstream>
#include<iostream>

void NeuralNet::forhead(vector<double> &output,int no){
	int count = hide[0].neuron_amount;
	double tempresult = 0;
	int size = input_count;
	for (int j = 0; j < input_count; j++){
		hide[0].output[j] = input[no][j];
	}
	hide[0].output[input_count] = hide[0].b;

	for (int i = 1; i <= hide_layers_count; i++){
		count = hide[i].neuron_amount;
		tempresult = 0;
		size = hide[i - 1].neuron_amount;
		for (int j = 0; j < count; j++){
			for (int k = 0; k <= size; k++)
				tempresult += hide[i - 1].output[k] * hide[i-1].weight[k][j];
			hide[i].output[j] = f(tempresult);
		}
		hide[i].output[count] = hide[i].b;
	}

	count = output_count;
	tempresult = 0;
	size = hide[hide_layers_count].neuron_amount;
	for (int j = 0; j < count; j++){
		for (int k = 0; k <= size; k++)
			tempresult += hide[hide_layers_count].output[k] * hide[hide_layers_count].weight[k][j];
		output[j] = f(tempresult);
	}
}


void NeuralNet::initializeWeight(){
	srand(time(NULL));

	int count;
	int size;
	for (int i = 0; i < hide_layers_count; i++){
		count = hide[i].neuron_amount;
		size = hide[i + 1].neuron_amount;
		for (int j = 0; j <= count; j++){
			hide[i].weight.push_back(vector<double>(size));
			for (int k = 0; k < size; k++){
				hide[i].weight[j][k] = (rand() % RAND_MAX - RAND_MAX / 2) / double(RAND_MAX)/2.318;
			}	
		}
		//hide[i].b = 0;
		hide[i].b = (rand() % RAND_MAX - RAND_MAX / 2) / double(RAND_MAX);
	}
	count = hide[hide_layers_count].neuron_amount;
	size = output_count;
	for (int j = 0; j <= count; j++){
		hide[hide_layers_count].weight.push_back(vector<double>(size));
		for (int k = 0; k < size; k++)
			hide[hide_layers_count].weight[j][k] = (rand() % RAND_MAX - RAND_MAX / 2) / double(RAND_MAX)/2.318;
	}
	//hide[hide_layers_count].b = 0;
	hide[hide_layers_count].b = (rand() % RAND_MAX - RAND_MAX / 2) / double(RAND_MAX);



	for (int i = 0; i < hide_layers_count; i++){
		count = hide[i].neuron_amount;
		size = hide[i + 1].neuron_amount;
		for (int j = 0; j <= count; j++){
			hide[i].deltaweight.push_back(vector<double>(size,0));
		}
	}
	count = hide[hide_layers_count].neuron_amount;
	size = output_count;
	for (int j = 0; j <= count; j++)
		hide[hide_layers_count].deltaweight.push_back(vector<double>(size,0));


}

NeuralNet::NeuralNet(vector<unsigned int> &neuralnum, func fun, int inputcount, int hidelayer, int outputcount) :       //neuralnum包含输入层结构，隐藏层结构
                      input_count(inputcount), hide_layers_count(hidelayer), output_count(outputcount),f(fun){
	for (unsigned int i = 0; i < neuralnum.size(); i++){
		unsigned int size = neuralnum[i];
		hide.push_back(hidelayers(size));
	}
}

void NeuralNet::bp(int numbers){
	double lamda = 0.1;
	double alpha = 0.1;
	int times = MAXTIMES;
	vector<double> realoutput(output_count);
	vector<double> errornl(output_count);
	double deltaJ = 0.0;
	while (deltaJ<0.01||times){
		times--;
		for (int n = 0; n < numbers; n++){
			//前向传播
			forhead(realoutput, n);
			//计算反向残差
			for (int i = 0; i < output_count; i++){
				errornl[i] = (realoutput[i] - output[n][i])*deltasigmond(realoutput[i]);
				deltaJ += pow((realoutput[i] - output[n][i]), 2);
			}
			int size = hide[hide_layers_count].neuron_amount;
			int count = output_count;
			for (int j = 0; j <= size; j++){
				double tempresult = 0;
				for (int k = 0; k < count; k++){
					tempresult += errornl[k] * hide[hide_layers_count].weight[j][k];
				}
				hide[hide_layers_count].error[j] = deltasigmond(hide[hide_layers_count].output[j])*tempresult;
			}
			for (int i = hide_layers_count - 1; i > 0; i--){
				size = hide[i].neuron_amount;
				count = hide[i + 1].neuron_amount;
				for (int j = 0; j <= size; j++){
					double tempresult = 0;
					for (int k = 0; k < count; k++){
						tempresult += hide[i + 1].error[k] * hide[i].weight[j][k];
					}
					hide[i].error[j] = deltasigmond(hide[i].output[j])*tempresult;
				}
			}

			//累计计算梯度改变量
			for (int i = 0; i < hide_layers_count; i++){
				count = hide[i].neuron_amount;
				size = hide[i + 1].neuron_amount;
				for (int j = 0; j <= count; j++){
					for (int k = 0; k < size; k++){
						hide[i].deltaweight[j][k] += hide[i].output[j] * hide[i + 1].error[k];
					}
				}
				for (int k = 0; k < size; k++){
					hide[i].berror += hide[i + 1].error[k];
				}
			}
			count = hide[hide_layers_count].neuron_amount;
			size = output_count;
			for (int j = 0; j <= count; j++){
				for (int k = 0; k < size; k++)
					hide[hide_layers_count].deltaweight[j][k] += hide[hide_layers_count].output[j] * errornl[k];
				
			}
			for (int k = 0; k < size; k++){
				hide[hide_layers_count].berror += errornl[k];
			}
		}
		deltaJ /= (2*numbers);
		cout << deltaJ << endl;
		//double grad2 = J(lamda, alpha, numbers);
		//double grad1;
		//得出整体代价函数梯度改变量
		for (int i = 0; i < hide_layers_count; i++){
			int count = hide[i].neuron_amount;
			int size = hide[i + 1].neuron_amount;
			for (int j = 0; j <= count; j++){
				for (int k = 0; k < size; k++){
					hide[i].deltaweight[j][k] = hide[i].deltaweight[j][k] / numbers + lamda*hide[i].weight[j][k];
					hide[i].weight[j][k] -= alpha*hide[i].deltaweight[j][k];
					hide[i].deltaweight[j][k] = 0;
				}
			}
			for (int k = 0; k < size; k++){
				hide[i].berror /= numbers;
				hide[i].b -= alpha*hide[i].berror;
				hide[i].berror = 0;
			}
		}
		
		
	}
}

void NeuralNet::train(const vector<vector<double>> &input, const vector<vector<double>> &output){
	this->input = input;
	this->output = output;
	initializeWeight();
	int numbers = input.size();
	bp(numbers);
}

NeuralNet::~NeuralNet(){
}


double NeuralNet::J(double lamda, double alpha, int numbers){
	double Jwb1 = 0, Jwb2 = 0;
	double delta = 0.0001;
	double sumw = 0;
	hide[0].weight[0][0]+=delta;
	vector<double> realoutput(output_count);
	vector<double> errornl(output_count);
	for (int n = 0; n < numbers; n++){
		//前向传播
		forhead(realoutput, n);
		//计算反向残差
		for (int i = 0; i < output_count; i++){
			errornl[i] += pow((realoutput[i] - output[n][i]),2)/2;
		}
	}
	for (int i = 0; i < hide_layers_count; i++){
		int count = hide[i].neuron_amount;
		int size = hide[i + 1].neuron_amount;
		for (int j = 0; j <= count; j++){
			for (int k = 0; k < size; k++){
				sumw += pow(hide[i].weight[j][k],2);
			}
		}
	}
	Jwb1 = errornl[0] / numbers + lamda*sumw/ 2;
	hide[0].weight[0][0] -= 2*delta;
	sumw = 0;
	for (int i = 0; i < output_count; i++){
		errornl[i] = 0;
	}
	for (int n = 0; n < numbers; n++){
		//前向传播
		forhead(realoutput, n);
		//计算反向残差
		for (int i = 0; i < output_count; i++){
			errornl[i] += pow((realoutput[i] - output[n][i]), 2) / 2;
		}
	}
	for (int i = 0; i < hide_layers_count; i++){
		int count = hide[i].neuron_amount;
		int size = hide[i + 1].neuron_amount;
		for (int j = 0; j <= count; j++){
			for (int k = 0; k < size; k++){
				sumw += pow(hide[i].weight[j][k], 2);
			}
		}
	}
	Jwb2 = errornl[0] / numbers + lamda*sumw / 2;
	hide[0].weight[0][0] += delta;
	double grad=(Jwb1 - Jwb2) / delta / 2;
	return grad;
}


void NeuralNet::weightvisual(){
	
	double temp[64];
	for (int i = 0; i < 25; i++){
		FILE* file;
		std::stringstream ss;
		ss << "C:/Users/hermione/Documents/Visual Studio 2013/Projects/neuralnetwork/" << i << ".raw";
		string sname = ss.str();
		const char* name = sname.c_str();
		fopen_s(&file, name, "wb+");
		for (int j = 0; j < 64; j++){
			temp[j] = hide[1].weight[i][j];
		}
		fwrite(temp, sizeof(double), 8 * 8, file);
	}
}

void NeuralNet::test(const vector<vector<double>> &input, vector<vector<double>> &output,double max){
	this->input = input;
	this->output = output;
	int numbers = input.size();
	double deltaJ = 0.0;
	for (int i = 0; i < numbers; i++){
		forhead(output[i], i);
		double temp = 0;
		for (int j = 0; j < output_count; j++){
			temp = pow((output[i][j] - this->output[i][j]), 2);
			deltaJ += temp;
		}
		cout << "x1 = " << input[i][0] << "    x2 = " << input[i][1] << "   ";
		cout << "3x1+2x2 = " << 3 * input[i][0] + 2 * input[i][1] + 1 << "    ";
		cout << "y = " << this->output[i][0]*max <<"    残差 = "<<temp<< endl;
	}
	deltaJ /= (2 * numbers);
	cout <<"testdelta: "<< deltaJ << endl;
}
