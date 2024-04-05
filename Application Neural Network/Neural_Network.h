#pragma once
#include <iostream>


using namespace std;

struct Data
{
	double* x;
	int y;
};

struct Lauer
{
	double* neurons_value;
	double* activations;
	double* delta;
	double* biases;
	double** weights;
	double(*activation_func)(double); //указатель на функцию активации
	double(*derivative_func)(double); //указатель на производную
	/*
	Lauer(int number_neurons)
	{
		neurons_value = new double[number_neurons];
		activations = new double[number_neurons];
		delta = new double[number_neurons];
		biases = new double[number_neurons];
		weights = new double[number_neurons];
	}*/
};


class Neural_Network
{
	int index_lauer = 0;
	void (Neural_Network::* optimizer)(); //указатель на метод класса
	string optimizer_network;
	Lauer* lauers;

public:
	int L;
	int* size_network;
	Neural_Network(int number_lauers)
	{
		L = number_lauers;
		size_network = new int[L];
		lauers = new Lauer[L];
	}
	void add_input_lauer(int number_neurons);
	void add_lauer(int number_neurons, string func);
	void compile(string func);
	void summary();
	void forwardfeed();
	void BackPropogation();
	void WeightsUpdater();
	void SGD();

	

};

