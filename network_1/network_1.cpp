#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
using namespace std;

struct Data
{
    int y;
    vector<double> x;
};

vector <Data> load_data(const char* path, int count_x)
{
    int examples;

    ifstream file;
    file.open(path);
    file >> examples;
    examples = (int)(examples * 0.1);
    vector <Data> data(examples);

    for (int i = 0; i < examples; ++i)
    {
        data[i].x.resize(count_x);
    }

    for (int i = 0; i < examples; ++i)
    {
        file >> data[i].y;
        for (int j = 0; j < count_x; ++j)
        {
            file >> data[i].x[j];
        }
    }

    return data;
}

// Для создания весов используется метод нормированной инициализации
double create_number(int n, int m)
{
    double low = -sqrt(6) / sqrt(n + m);
    double high = sqrt(6) / sqrt(n + m);
    double k = low + (double)(rand()) / RAND_MAX * (high - low);

    return k;
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
    int size_mini_batch = 100;
    int count_x = 784;
    const int L = 6;
    int count_batch;
    int num_parameters;
    double epoch = 5;
    double time_SGD = 0, error_time = 0;
    double right_answers_data;
    double** weights[L];
    double* biases[L];
    double* delta[L];
    double* neurons_value[L];
    double* activations[L];

    const char* path_train = "MNIST_train.txt";

    vector <Data> train_data;

    //cout << "Loading data..." << endl;
    train_data = load_data(path_train, count_x);
    //cout << "Data loaded." << endl;
    count_batch = train_data.size() / size_mini_batch;

    for (int a = 1; a < 12; a++)
    {
        num_parameters = 0;
        int size_network[L]{ 784, a * 40 * 2, a * 30 * 2, a * 20 * 2, a * 10 * 2,  10 };
        for (int i = 1; i < L; i++)
        {
            if (i == 1)
            {
                neurons_value[0] = new double[size_network[0]];
            }
            weights[i] = new double* [size_network[i]];
            biases[i] = new double[size_network[i]];
            delta[i] = new double[size_network[i]];
            neurons_value[i] = new double[size_network[i]];
            activations[i] = new double[size_network[i]];
        }

        for (int i = 1; i < L; i++)
        {
            srand(time(NULL));
            for (int j = 0; j < size_network[i]; j++)
            {
                weights[i][j] = new double[size_network[i - 1]];
                biases[i][j] = create_number(size_network[i - 1], size_network[i]);
                for (int k = 0; k < size_network[i - 1]; k++)
                {
                    weights[i][j][k] = create_number(size_network[i - 1], size_network[i]);
                }
            }
        }
        for (int i = 1; i < L; i++)
        {
            num_parameters += size_network[i - 1] * size_network[i] + size_network[i];
        }
        cout << "\nNumber of parameters: " << num_parameters << endl;

        time_SGD = 0;
        error_time = 0;

        for (int p = 1; p < epoch + 1; p++)
        {
            right_answers_data = 0;

            for (int m = 0; m < count_batch; m++)
            {
                for (int v = size_mini_batch * m; v < size_mini_batch * (m + 1); v++)
                {
                    //Добавление входных данных в  первый сллой сети
                    for (int i = 0; i < size_network[0]; i++)
                    {
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

                    if (predict == train_data[v].y)
                    {
                        right_answers_data++;
                    }

                    //Обрастное распространение ошибки
                    double error_t1 = omp_get_wtime();
                    for (int i = 0; i < size_network[L - 1]; i++)
                    {
                        if (i != train_data[v].y)
                        {
                            delta[L - 1][i] = neurons_value[L - 1][i] * derivative_sigmoid(activations[L - 1][i]);
                        }
                        else
                        {
                            delta[L - 1][i] = (neurons_value[L - 1][i] - train_data[v].y) * derivative_sigmoid(activations[L - 1][i]);
                        }
                    }

                    for (int i = L - 2; i > 0; i--)
                    {
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
                    double error_t2 = omp_get_wtime();
                    error_time += error_t2 - error_t1;
                }
                //Изменение всех весов и смещений
                double sgd_t1 = omp_get_wtime();
                double learning_rate = 0.35 * exp(-p / epoch);
                for (int i = 1; i < L; i++)
                {
                    for (int j = 0; j < size_network[i]; j++)
                    {
                        biases[i][j] = biases[i][j] - learning_rate * delta[i][j];
                        for (int k = 0; k < size_network[i - 1]; k++)
                        {
                            weights[i][j][k] = weights[i][j][k] - learning_rate * delta[i][j] * neurons_value[i - 1][k];
                        }
                    }
                }

                double sgd_t2 = omp_get_wtime();
                time_SGD += sgd_t2 - sgd_t1;
            }

            //cout << " Back propagation error: " << error_time / (size_mini_batch * count_batch * p) * 1000000 << " s * 10^-6. ";
            //cout << "SGD: " << time_SGD / (count_batch * p) * 1000000 << " s * 10^-6." << endl;

        }
        cout << "\nMean time:" << endl;
        cout <<"Back propagation error: " << error_time / (size_mini_batch * count_batch * epoch) * 1000000 << " s * 10^-6. ";
        cout << "SGD: " << time_SGD / (count_batch * epoch) * 1000000 << " s * 10^-6." << endl;

        for (int i = 1; i < L; i++)
        {
            if (i == 1)
            {
                delete[] neurons_value[0];
            }
            delete[] biases[i];
            delete[] delta[i];
            delete[] neurons_value[i];
            delete[] activations[i];
            for (int j = 0; j < size_network[i]; j++)
            {
                delete[] weights[i][j];
            }
            delete[] weights[i];
        }
    }
    
}
