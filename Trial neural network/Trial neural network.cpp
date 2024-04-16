#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
using namespace std;


struct Data
{
    int y;
    vector<double> x;
};


vector <Data> load_data(string path, int count_x)
{
    int examples;

    ifstream file;
    file.open(path);
    file >> examples;
    examples *= 0.1;
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

    int count_x = 784;
    vector <Data> train_data;
    vector <Data> test_data;
    string path_train = "A:/University/Diploma/Pre-graduate practice/Data/MNIST_train.txt";
    string path_test = "A:/University/Diploma/Pre-graduate practice/Data/MNIST_test.txt";


    cout << "Loading data..." << endl;
    train_data = load_data(path_train, count_x);
    test_data = load_data(path_test, count_x);
    cout << "Data loaded." << endl;

    //Инициализация нейросети
    const int L = 4;
    int size_network[L]{ 784, 256, 30, 10 };

    double* biases[L];
    double* delta[L];
    double* neurons_value[L];
    double* activations[L]; 
        
    cout << "Initialisation Network" << endl;
    cout << "Number lauers: " << L << endl;
    cout << "Size network: ";
    for (int i = 0; i < L; i++)
    {
        cout << size_network[i] << " ";
    }
    cout << endl;

    for (int i = 1; i < L; i++)
    {
        if (i == 1)
        {
            neurons_value[0] = new double[size_network[0]];
        }

        biases[i] = new double[size_network[i]];
        delta[i] = new double[size_network[i]];
        neurons_value[i] = new double[size_network[i]];
        activations[i] = new double[size_network[i]];
    }

    for (int i = 1; i < L; i++)
    {
        for (int j = 0; j < size_network[i]; j++)
        {
            biases[i][j] = create_number(size_network[i - 1], size_network[i]);
        }
    }
    bool t = false;
    double** weights3[L];
    double* weights2[L];
    if (t)
    {   
        
        for (int i = 1; i < L; i++)
        {
            weights3[i] = new double* [size_network[i]];
        }
        for (int i = 1; i < L; i++)
        {
            for (int j = 0; j < size_network[i]; j++)
            {
                weights3[i][j] = new double[size_network[i - 1]];
            }
        } 
        for (int i = 1; i < L; i++)
        {
            for (int j = 0; j < size_network[i]; j++)
            {
                for (int k = 0; k < size_network[i - 1]; k++)
                {
                    weights3[i][j][k] = create_number(size_network[i - 1], size_network[i]);
                }
            }
        }
    }
    else
    {
        
        for (int i = 1; i < L; i++)
        {
            weights2[i] = new double[size_network[i] * size_network[i - 1]];
        }
        for (int i = 1; i < L; i++)
        {
            for (int j = 0; j < size_network[i] * size_network[i - 1]; j++)
            {
                weights2[i][j] = create_number(size_network[i - 1], size_network[i]);
            }
        }    
    }



    int size_mini_batch = 10;
    int count_batch = train_data.size() / size_mini_batch;
    double epoch = 10;
    double right_answers_data, right_answers_test;
    double begin, end;
    chrono::duration<double> time;
    
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
                    //activations[0][i] = train_data[v].x[i];
                    neurons_value[0][i] = train_data[v].x[i];
                }

                //Прямое распространение
                for (int i = 1; i < L; i++)
                {
                    for (int j = 0; j < size_network[i]; j++)
                    {
                        double z = 0;
                        if (t)
                        {
                            for (int k = 0; k < size_network[i - 1]; k++)
                            {
                                 z += weights3[i][j][k] * neurons_value[i - 1][k];
                                
                            }
                        }
                        else 
                        {
                            int step = j * size_network[i - 1];
                            for (int k = step; k < step + size_network[i - 1]; k++)
                            {
                                z += weights2[i][k] * neurons_value[i - 1][k - step];
                            }

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
                        if (t)
                        {
                            for (int k = 0; k < size_network[i + 1]; k++)
                            {
                                sum += delta[i + 1][k] * weights3[i + 1][k][j];
                            }
                        }
                        else
                        {
                            int step = size_network[i];
                            for (int k = j; k < size_network[i] * size_network[i + 1] + 1; k += step)
                            {
                                int a = (k - j) / step;
                                sum += weights2[i + 1][k] * delta[i + 1][a];
                            }
                        }
                         
                        delta[i][j] = derivative_sigmoid(activations[i][j]) * sum;
                    }

                }
                

            }
            
            //Изменение всех весов и смещений
            double learning_rate = 0.35 * exp(-p / epoch);
            for (int i = 1; i < L; i++)
            {
                for (int j = 0; j < size_network[i]; j++)
                {
                    biases[i][j] = biases[i][j] - learning_rate * delta[i][j];
                    if (t)
                    {
                        for (int k = 0; k < size_network[i - 1]; k++)
                        {
                            weights3[i][j][k] = weights3[i][j][k] - learning_rate * delta[i][j] * neurons_value[i - 1][k]; // size_mini_batch;
                        }
                    }
                    else
                    {
                        int step = j * size_network[i - 1];
                        for (int k = step; k < step + size_network[i - 1]; k++)
                        {
                            weights2[i][k] = weights2[i][k] - learning_rate * delta[i][j] * neurons_value[i - 1][k - step];
                        }
                    }
                }
            }
            /*
            for (int i = 1; i < L; i++)
            {
                for (int j = 0; j < size_network[i]; j++)
                {
                    biases[i][j] = biases[i][j] - learning_rate * delta[i][j]; // size_mini_batch;
                }
            }*/
            
        }
        
        //Оценка прогресса в обучении
        for (int v = 0; v < test_data.size(); v++)
        {
            //Добавление входных данных в  первый сллой сети
            for (int i = 0; i < size_network[0]; i++)
            {
                neurons_value[0][i] = test_data[v].x[i];
            }

            //Прямое распространение
            for (int i = 1; i < L; i++)
            {
                for (int j = 0; j < size_network[i]; j++)
                {
                    double z = 0;

                    if (t)
                    {
                        for (int k = 0; k < size_network[i - 1]; k++)
                        {
                            z += weights3[i][j][k] * neurons_value[i - 1][k];
                        }
                    }
                    else
                    {
                        int step = j * size_network[i - 1];
                        for (int k = step; k < step + size_network[i - 1]; k++)
                        {
                            z += weights2[i][k] * neurons_value[i - 1][k - step];
                        }
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

            if (predict == test_data[v].y)
            {
                right_answers_test++;
            }
           
        }
        auto end = chrono::steady_clock::now();
        time = end - begin;

        cout << "Epoch: " << p + 1 << "/" << epoch << " Accuracy data: " << right_answers_data / train_data.size() * 100;
        cout << " Accuracy test: " << right_answers_test / test_data.size() * 100 << " Time: " << time.count() << endl;

    }   
}


