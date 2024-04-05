#include "Neural_Network.h"
#include <random>

// Для создания весов используется метод нормализованной инициализации
double create_number(int n, int m)
{
	double low = -sqrt(6) / sqrt(n + m);
	double high = sqrt(6) / sqrt(n + m);
	uniform_real_distribution<> distr(low, high);
	random_device rd;
	mt19937 gen(rd());

	return distr(gen);
}

//Функции активации
double sigmoid(double z)
{
	return 1 / (1 + exp(-z));
}

//Производные функций активаций
double derivative_sigmoid(double z)
{
	double delta = sigmoid(z) * (1 - sigmoid(z));

	return delta;
}
//хе
void Neural_Network::SGD()
{
	cout << "SGD" << endl;
}


void Neural_Network::add_input_lauer(int number_neurons)
{
	/*
	* Добавляет входной слой нейронной сети
	* Входные параметры:
	* int number_neurons - кол-во нейронов на данном слое
	* Атрибуты:
	* neurons_value - массив активационного потенциала всех нейронов данного слоя
	*/
	// Так можно не указывая, положение слоя, можно легко его создать 
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
	* Создает слой нейронной сети
	* В качестве основы используется struct lauer
	* Входные параметры: 
	* int number_neurons - кол-во нейронов на данном слое
	* func - тип функции активации. Варианты: sigmoid, ReLU
	* Атрибуты:
	* neurons_value - массив активационного потенциала всех нейронов данного слоя
	* activations - массив со значениями активационного потенциала
	* delta - массив со значениями меры влияния нейрона на величину ошибки
	* biases - массив смещений(заполняется случайными значениями при вызове функции compile())
	* weights - массив(двумерный) весов (заполняется случайными значениями при вызове функции compile())
	* Кол-во элементов для activations, delta, biases равно кол-во number_neurons данного слоя 
	* Кол-во строк для weights равно кол-во number_neurons текущего слоя, а кол-во столбцов - number_neurons предыдущего слоя
	*/
	
	// Так можно не указывая, положение слоя, можно легко его создать 
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
	* Эта функция задает тип оптимизатора функции потерь, а также заполначет веса и смещения слоев нейронной сети случайными значениям
	* Входные параметры:
	* func - тип оптимизатора. Варианты: SGD
	* 
	* void (A::*P)() = (A::&B);
	*/
	if (func == "SGD")
	{
		optimizer_network = "SGD";
		this->optimizer = &Neural_Network::SGD; //присвоение указателю метода
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
	(this->*optimizer)(); //вызов метода с помощью указателя
}


