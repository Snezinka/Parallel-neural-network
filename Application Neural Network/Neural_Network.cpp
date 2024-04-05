#include "Neural_Network.h"
#include <random>

// ��� �������� ����� ������������ ����� ��������������� �������������
double create_number(int n, int m)
{
	double low = -sqrt(6) / sqrt(n + m);
	double high = sqrt(6) / sqrt(n + m);
	uniform_real_distribution<> distr(low, high);
	random_device rd;
	mt19937 gen(rd());

	return distr(gen);
}

//������� ���������
double sigmoid(double z)
{
	return 1 / (1 + exp(-z));
}

//����������� ������� ���������
double derivative_sigmoid(double z)
{
	double delta = sigmoid(z) * (1 - sigmoid(z));

	return delta;
}
//��
void Neural_Network::SGD()
{
	cout << "SGD" << endl;
}


void Neural_Network::add_input_lauer(int number_neurons)
{
	/*
	* ��������� ������� ���� ��������� ����
	* ������� ���������:
	* int number_neurons - ���-�� �������� �� ������ ����
	* ��������:
	* neurons_value - ������ �������������� ���������� ���� �������� ������� ����
	*/
	// ��� ����� �� ��������, ��������� ����, ����� ����� ��� ������� 
	size_network[index_lauer] = number_neurons;

	Lauer lauer;
	lauer.neurons_value = new double[number_neurons];
	lauers[index_lauer] = lauer;

	if (index_lauer < L - 1)
	{
		index_lauer++;
	}

}

void Neural_Network::add_lauer(int number_neurons, string func)
{
	/*
	* ������� ���� ��������� ����
	* � �������� ������ ������������ struct lauer
	* ������� ���������: 
	* int number_neurons - ���-�� �������� �� ������ ����
	* func - ��� ������� ���������. ��������: sigmoid, ReLU
	* ��������:
	* neurons_value - ������ �������������� ���������� ���� �������� ������� ����
	* activations - ������ �� ���������� �������������� ����������
	* delta - ������ �� ���������� ���� ������� ������� �� �������� ������
	* biases - ������ ��������(����������� ���������� ���������� ��� ������ ������� compile())
	* weights - ������(���������) ����� (����������� ���������� ���������� ��� ������ ������� compile())
	* ���-�� ��������� ��� activations, delta, biases ����� ���-�� number_neurons ������� ���� 
	* ���-�� ����� ��� weights ����� ���-�� number_neurons �������� ����, � ���-�� �������� - number_neurons ����������� ����
	*/
	
	// ��� ����� �� ��������, ��������� ����, ����� ����� ��� ������� 
	size_network[index_lauer] = number_neurons;

	Lauer lauer;
	lauer.neurons_value = new double[number_neurons];
	lauer.activations = new double[number_neurons];
	lauer.delta = new double[number_neurons];
	lauer.biases = new double[number_neurons];
	lauer.weights = new double*[number_neurons];
	lauers[index_lauer] = lauer;

	if (index_lauer < L - 1)
	{
		index_lauer++;
	}

	if (func == "sigmoid")
	{
		lauer.activation_func = sigmoid;
		lauer.derivative_func = derivative_sigmoid;
	}
}

void Neural_Network::compile(string func)
{
	/*
	* ��� ������� ������ ��� ������������ ������� ������, � ����� ���������� ���� � �������� ����� ��������� ���� ���������� ���������
	* ������� ���������:
	* func - ��� ������������. ��������: SGD
	* 
	* void (A::*P)() = (A::&B);
	*/
	if (func == "SGD")
	{
		optimizer_network = "SGD";
		this->optimizer = &Neural_Network::SGD; //���������� ��������� ������
	}
	for (int i = 1; i < L - 1; i++)
	{
		for (int j = 0; j < size_network[i]; j++)
		{
			lauers[i].biases[j] = create_number(size_network[i - 1], size_network[i]);
			lauers[i].weights[j] = new double[size_network[i - 1]];
			for (int k = 0; k < size_network[i - 1]; k++)
			{
				lauers[i].weights[j][k] = create_number(size_network[i - 1], size_network[i]);
			}
		}
	}

}

void Neural_Network::summary()
{
	cout << "Neural Network" << endl;
	cout << "Number lauers: " << L << endl;
	cout << "Size network: ";
	for (int i = 0; i < L; i++)
	{
		cout << size_network[i] << " ";
	}
	cout << endl;
	cout << "Optimizer: ";
	(this->*optimizer)(); //����� ������ � ������� ���������
}


