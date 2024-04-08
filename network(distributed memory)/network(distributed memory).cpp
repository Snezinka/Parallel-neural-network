#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

struct Data
{
    double* x;
    int y;
};
/*
//Для выполнения обучения нейронной сети необходимо знать количество образцов и сами образцы
//использование дополнительной структуры позволит запихнуть все процесс сбора данных в функцию
struct Sample
{
    int examples;
    Data* sample;
};


void loading_data(string path, Sample* train_data, Sample* test_data, int count_x, double size_test = 0.2)
{

    int examples;
    cout << "Loading data..." << endl;
    ifstream fin;
    fin.open(path);
    fin >> examples;
    examples -= 60000;
    train_data->examples = (int)(examples * (1 - size_test));
    test_data->examples = examples - train_data->examples;
    cout << "Examples train: " << train_data->examples << "\t" << "examples test: " << test_data->examples << endl;

    train_data->sample = new Data[train_data->examples];
    test_data->sample = new Data[test_data->examples];

    for (int i = 0; i < train_data->examples; i++)
    {
        train_data->sample[i].x = new double[count_x];
        if (i < test_data->examples)
        {
            test_data->sample[i].x = new double[count_x];
        }
    }

    for (int i = 0; i < train_data->examples; i++)
    {
        fin >> train_data->sample[i].y;
        for (int j = 0; j < count_x; j++)
        {
            fin >> train_data->sample[i].x[j];
        }
    }

    for (int i = 0; i < test_data->examples; i++)
    {
        fin >> test_data->sample[i].y;
        for (int j = 0; j < count_x; j++)
        {
            fin >> test_data->sample[i].x[j];
        }
    }

    cout << "Data loaded." << endl;
    
    //return train_data, test_data;
}
*/

Data* loading_data(string path, int count_x, int& examples)
{
    cout << "Loading data..." << endl;

    Data* data;
    ifstream fin;
    fin.open(path);
    fin >> examples;
    examples -= 40000;
    data = new Data[examples];

    for (int i = 0; i < examples; i++)
    {
        data[i].x = new double[count_x];
    }

    for (int i = 0; i < examples; i++)
    {
        fin >> data[i].y;
        for (int j = 0; j < count_x; j++)
        {
            fin >> data[i].x[j];
        }
    }
    cout << "Data loaded." << endl;
    return data;
}


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

double sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double derivative_sigmoid(double z)
{
    double delta = sigmoid(z) * (1 - sigmoid(z));

    return delta;
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank_process, size_process, tag = 1;
    int rank_thread, size_threads = 10;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_process);
    MPI_Comm_size(MPI_COMM_WORLD, &size_process);

    omp_set_num_threads(size_threads);
    
    //Все основные вычисления будут производиться на нулевом процессе
    //Остальные процессы играют вспомогательную функцию
    if (rank_process == 0)
    {
        
        //Загрузка данных
        int count_x = 784;
        string path = "A:/University/Diploma/Pre-graduate practice/Data/lib_MNIST_edit.txt";
        Data* train_data;
        int examples_train;
        train_data = loading_data(path, count_x, examples_train);
        cout << train_data[0].y;
        /*
        Data* test_data;
        int examples_test;
        test_data = loading_data(path, count_x, examples_test);
        


        //Инициализация нейросети
        cout << "Initialisation Network" << endl;
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


        //Начало обучения
        int size_mini_batch = 100;
        int count_batch = examples_train / size_mini_batch;
        double epoch = 1;
        chrono::duration<double> time;
        
        for (int p = 0; p < epoch; p++)
        {
            double right_answers_data = 0;
            double right_answers_test = 0;
            auto begin = chrono::steady_clock::now();
            for (int m = 0; m < count_batch; m++)
            {
                for (int v = size_mini_batch * m; v < size_mini_batch * (m + 1); v++)
                {
                    //Добавление входных данных в  первый сллой сети
                    for (int i = 0; i < size_network[0]; i++)
                    {
                        activations[0][i] = train_data[v].x[i];
                        neurons_value[0][i] = train_data[v].x[i];
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
                }
            }
        }
*/
    }
    
    

    /*
    
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
    */
    MPI_Finalize();
}
