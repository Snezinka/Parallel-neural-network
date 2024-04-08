#include "Neural_Network.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

Data* loading_data(string path, Neural_Network& network, double size_test = 0.2)
{
	cout << "Loading data..." << endl;
	int examples;
	int examples_train;
	int examples_test;
	Data* train_data;
	Data* test_data;

	ifstream fin;
	fin.open(path);
	fin >> examples;
	examples_train = examples * (1 - size_test);
	examples_test = examples - examples_train;
	cout << "Examples train: " << examples_train << "\t" << "examples test: " << examples_test << endl;
	train_data = new Data[examples_train];
	test_data = new Data[examples_test];

	int train = 0;
	int test = 0;

	for (int i = 0; i < examples_train; i++)
	{
		train_data[i].x = new double[network.size_network[0]];
		if (i < examples_test)
		{
			test_data[i].x = new double[network.size_network[0]];
		}
	}


	for (int i = 0; i < examples_train; i++)
	{

		fin >> train_data[i].y;
		for (int j = 0; j < network.size_network[0]; j++)
		{
			fin >> train_data[i].x[j];
		}
		train++;
	}

	for (int i = 0; i < examples_test; i++)
	{

		fin >> test_data[i].y;
		for (int j = 0; j < network.size_network[0]; j++)
		{
			fin >> test_data[i].x[j];
		}
		test++;
	}

	cout << "Data loaded." << endl;

	return train_data, test_data;
}


int main()
{
	
	string path = "Data_MNIST.txt";
	Data* train_data;
	Data* test_data;
	
	Neural_Network network(5);
	network.add_input_lauer(784);
	network.add_lauer(256, "sigmoid");
	network.add_lauer(128, "sigmoid");
	network.add_lauer(30, "sigmoid");
	network.add_lauer(10, "sigmoid");
	network.compile("SGD");
	network.summary();


	train_data, test_data = loading_data("Data_MNIST.txt", network);
}
