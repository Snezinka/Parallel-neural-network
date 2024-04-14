#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

// Для создания весов используется метод нормированной инициализации
double create_number(int n, int m)
{
    double low = -sqrt(6) / sqrt(n + m);
    double high = sqrt(6) / sqrt(n + m);
    uniform_real_distribution<> distr(low, high);
    random_device rd;
    mt19937 gen(rd());

    return distr(gen);
}

double sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double derivative_sigmoid(double z)
{
    double delta = sigmoid(z) * (1 - sigmoid(z));

    return delta;
}

int main()
{

    cout << "Loading data\n" << endl;
    string path = "A:/University/Diploma/Pre-graduate practice/Data/lib_MNIST_edit.txt";
    ifstream fin;
    fin.open(path);
    int examples_data;

    fin >> examples_data;
    //examples_data -= 50048;
    cout << "Examples: " << examples_data << endl;
    double** x_data = new double* [examples_data];
    double* y_data = new double[examples_data];
    for (int i = 0; i < examples_data; ++i)
    {
        x_data[i] = new double[784];
    }

    for (int i = 0; i < examples_data; ++i)
    {
        fin >> y_data[i];
        for (int j = 0; j < 784; ++j)
        {
            fin >> x_data[i][j];
        }
    }
    fin.close();
    cout << "Data MNIST loaded \n";

    path = "A:/University/Diploma/Pre-graduate practice/Data/lib_10k.txt";
    int examples_test;
    fin.open(path);
    fin >> examples_test;
    // examples_test -= 9000;
    cout << "Examples: " << examples_test << endl;
    double** x_test = new double* [examples_test];
    double* y_test = new double[examples_test];
    for (int i = 0; i < examples_test; ++i)
    {
        x_test[i] = new double[784];
    }

    for (int i = 0; i < examples_test; ++i)
    {
        fin >> y_test[i];
        for (int j = 0; j < 784; ++j)
        {
            fin >> x_test[i][j];
        }
    }
    fin.close();


    cout << "Test MNIST loaded \n" << endl;
    cout << "Initialisation Network" << endl;
    //Инициализация нейросети
    const int L = 3;
    int size_network[L]{ 784, 256, 10 };
    cout << "Number lauers: " << L << endl;
    cout << "Size network: ";
    for (int i = 0; i < L; i++)
    {
        cout << size_network[i] << " ";
    }
    cout << endl;

    double** weights[L];
    double* biases[L];
    double* delta[L];
    for (int i = 0; i < L; i++)
    {
        weights[i] = new double* [size_network[i]];
        biases[i] = new double[size_network[i]];
        delta[i] = new double[size_network[i]];
    }
    for (int i = 1; i < L; i++)
    {
        for (int j = 0; j < size_network[i]; j++)
        {
            weights[i][j] = new double[size_network[i - 1]];
        }
    }

    for (int i = 1; i < L; i++)
    {
        for (int j = 0; j < size_network[i]; j++)
        {
            biases[i][j] = create_number(size_network[i - 1], size_network[i]);
            for (int k = 0; k < size_network[i - 1]; k++)
            {
                weights[i][j][k] = create_number(size_network[i - 1], size_network[i]);
            }
        }
    }

    double* neurons_value[L];
    double* activations[L];

    for (int i = 0; i < L; i++)
    {
        neurons_value[i] = new double[size_network[i]];
        activations[i] = new double[size_network[i]];
    }

    int size_mini_batch = 100;
    int count_batch = examples_data / size_mini_batch;
    int num_threads = 10;
    double epoch = 20;
    chrono::duration<double> time;
    double right_answers_data;
    double right_answers_test;

    omp_set_num_threads(num_threads);
    for (int p = 0; p < epoch; p++)
    {
        right_answers_data = 0;
        right_answers_test = 0;
        auto begin = chrono::steady_clock::now();

        for (int m = 0; m < count_batch; m++)
        {
            for (int v = size_mini_batch * m; v < size_mini_batch * (m + 1); v++)
            {
                //Добавление входных данных в  первый сллой сети
                for (int i = 0; i < size_network[0]; i++)
                {
                    activations[0][i] = x_data[v][i];
                    neurons_value[0][i] = x_data[v][i];
                }

                //Прямое распространение
                for (int i = 1; i < L; i++)
                {
                    for (int j = 0; j < size_network[i]; j++)
                    {
                        double z = 0;
                        for (int k = 0; k < size_network[i - 1]; k++)
                        {
                            z += weights[i][j][k] * neurons_value[i - 1][k];
                        }
                        activations[i][j] = z + biases[i][j];
                        neurons_value[i][j] = sigmoid(z + biases[i][j]);
                    }
                }

                //Посчет предсказания
                double max = neurons_value[L - 1][0];
                int predict = 0;
                for (int i = 1; i < size_network[L - 1]; i++)
                {
                    if (max < neurons_value[L - 1][i])
                    {
                        max = neurons_value[L - 1][i];
                        predict = i;
                    }

                }

                if (predict == y_data[v])
                {
                    right_answers_data++;
                }

                //Обрастное распространение ошибки
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < size_network[L - 1]; i++)
                {
                    if (i != y_data[v])
                    {
                        delta[L - 1][i] = neurons_value[L - 1][i] * derivative_sigmoid(activations[L - 1][i]);
                    }
                    else
                    {
                        delta[L - 1][i] = (neurons_value[L - 1][i] - y_data[v]) * derivative_sigmoid(activations[L - 1][i]);
                    }
                }

                for (int i = L - 2; i > 0; i--)
                {
#pragma omp for schedule(dynamic, 1)
                    for (int j = 0; j < size_network[i]; j++)
                    {
                        double sum = 0;
                        for (int k = 0; k < size_network[i + 1]; k++)
                        {
                            sum += delta[i + 1][k] * weights[i + 1][k][j];
                        }
                        delta[i][j] = derivative_sigmoid(activations[i][j]) * sum;
                    }
                }
            }
            //Изменение всех весов и смещений
            double learning_rate = 0.35 * exp(-p / epoch);
            //learning_rate = 0.35;
            for (int i = 1; i < L; i++)
            {
#pragma omp for schedule(dynamic, 1)
                for (int j = 0; j < size_network[i]; j++)
                {
                    for (int k = 0; k < size_network[i - 1]; k++)
                    {
                        weights[i][j][k] = weights[i][j][k] - learning_rate * delta[i][j] * neurons_value[i - 1][k]; // size_mini_batch;
                    }
                }
            }
#pragma omp for schedule(static, 1)
            for (int i = 1; i < L; i++)
            {
                for (int j = 0; j < size_network[i]; j++)
                {
                    biases[i][j] = biases[i][j] - learning_rate * delta[i][j]; // size_mini_batch;
                }
            }
        }
        //Оценка прогресса в обучении
        for (int v = 0; v < examples_test; v++)
        {
            //Добавление входных данных в  первый сллой сети
            for (int i = 0; i < size_network[0]; i++)
            {
                neurons_value[0][i] = x_test[v][i];
            }

            //Прямое распространение
            for (int i = 1; i < L; i++)
            {
                for (int j = 0; j < size_network[i]; j++)
                {
                    double z = 0;
                    for (int k = 0; k < size_network[i - 1]; k++)
                    {
                        z += weights[i][j][k] * neurons_value[i - 1][k];
                    }
                    activations[i][j] = z + biases[i][j];
                    neurons_value[i][j] = sigmoid(z + biases[i][j]);
                }
            }

            //Посчет предсказания
            double max = neurons_value[L - 1][0];
            int predict = 0;
            for (int i = 1; i < size_network[L - 1]; i++)
            {
                if (max < neurons_value[L - 1][i])
                {
                    max = neurons_value[L - 1][i];
                    predict = i;
                }

            }

            if (predict == y_test[v])
            {
                right_answers_test++;
            }
        }
        auto end = chrono::steady_clock::now();
        time = end - begin;

        cout << "Epoch: " << p + 1 << "/" << epoch << " Accuracy data: " << right_answers_data / examples_data * 100;
        cout << " Accuracy test: " << right_answers_test / examples_test * 100 << " Time: " << time.count() << endl;
    }


}
