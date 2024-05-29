#include <mpi.h>
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
    examples =(int)(examples * 0.1);
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

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank_process, size_process;
    int size_threads = 12;
    int count_x = 784;
    int size_mini_batch = 100;
    int epoch = 5;
    int count_batch;
    int num_parameters;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_process);
    MPI_Comm_size(MPI_COMM_WORLD, &size_process);
    
    const int L = 6;
    

    vector <Data> train_data;

    const char* path_train = "MNIST_train.txt";
  
    if (rank_process == 0)
    {
        //cout << "Loading data..." << endl;
        train_data = load_data(path_train, count_x);
        //cout << "Data loaded." << endl;
    }
    if (rank_process == 0)
    {
        count_batch = train_data.size() / size_mini_batch;
        cout << "\nSize core: " << size_process * size_threads << endl;
    }
    MPI_Bcast(&count_batch, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int a = 1; a < 2; a++)
    {
        num_parameters = 0;
        int size_network[L]{ 784,  40 * a * 2,  30 * a * 2,  20 * a * 2,  10 * a * 2,  10 };

        if (rank_process == 0)
        {
            for (int i = 1; i < L; i++)
            {
                num_parameters += size_network[i - 1] * size_network[i] + size_network[i];
            }
            cout << "\nNumber of parameters: " << num_parameters << endl;
        }

        int network_process[L];
        for (int i = 0; i < L; i++)
        {
            network_process[i] = ceil((double)size_network[i] / size_process) * size_process;
        }

        double** process_weights = new double* [L];
        double** process_biases = new double* [L];
        double** delta = new double* [L];
        double** activations = new double* [L];
        double t1_SGD, t2_SGD, time_SGD = 0;
        double t1_error, t2_error, time_error = 0;
        double right_answers_data;

        activations[0] = new double[network_process[0]];
        for (int i = 1; i < L; i++)
        {
            process_weights[i] = new double[network_process[i - 1] * network_process[i] / size_process];
            process_biases[i] = new double[network_process[i] / size_process];
            activations[i] = new double[network_process[i]];
            delta[i] = new double[network_process[i]];
        }
        for (int j = 0; j < network_process[0]; j++)
        {
            activations[0][j] = 0;
        }

        for (int i = 1; i < L; i++)
        {
            srand(time(NULL));
            for (int j = 0; j < network_process[i - 1] * network_process[i] / size_process; j++)
            {
                process_weights[i][j] = create_number(network_process[i - 1], network_process[i]);
            }

            for (int j = 0; j < network_process[i] / size_process; j++)
            {
                process_biases[i][j] = create_number(network_process[i - 1], network_process[i]);
            }
        }


        //Начало обучения
        for (int p = 1; p < epoch + 1; p++)
        {
            if (rank_process == 0)
            {
                right_answers_data = 0;
            }

            for (int m = 0; m < count_batch; m++)
            {
                for (int v = size_mini_batch * m; v < size_mini_batch * (m + 1); v++)
                {
                    //Прямое распространение
                    if (rank_process == 0)
                    {
                        for (int j = 0; j < size_network[0]; j++)
                        {
                            activations[0][j] = train_data[v].x[j];
                        }
                    }
                    MPI_Bcast(activations[0], network_process[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

                    for (int i = 1; i < L; i++)
                    {
                        if (i == 1)
                        {
                            double* matrix_weights = new double[network_process[i - 1] * network_process[i]];
                            double* matrix_activations = new double[network_process[i] / size_process];
                            MPI_Gather(process_weights[i], network_process[i - 1] * network_process[i] / size_process, MPI_DOUBLE, matrix_weights, network_process[i - 1] * network_process[i] / size_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#pragma omp parallel num_threads(size_threads)
                            {
#pragma omp for schedule(static, 1)
                                for (int j = 0; j < network_process[i] / size_process; j++)
                                {
                                    double z = 0;
                                    for (int k = 0; k < network_process[i - 1]; k++)
                                    {
                                        z += activations[i - 1][k] * matrix_weights[j * network_process[i - 1] + k];
                                    }
                                    matrix_activations[j] = z + process_biases[i][j];
                                }
                            }
                            MPI_Gather(matrix_activations, network_process[i] / size_process, MPI_DOUBLE, activations[i], network_process[i] / size_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                            delete[] matrix_weights;
                            delete[] matrix_activations;
                        }
                        else
                        {
                            double* matrix_weights = new double[network_process[i - 1] * network_process[i]];
                            double* matrix_activations = new double[network_process[i] / size_process];
                            MPI_Gather(process_weights[i], network_process[i - 1] * network_process[i] / size_process, MPI_DOUBLE, matrix_weights, network_process[i - 1] * network_process[i] / size_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#pragma omp parallel num_threads(size_threads)
                            {
#pragma omp for schedule(static, 1)
                                for (int j = 0; j < network_process[i] / size_process; j++)
                                {
                                    double z = 0;
                                    for (int k = 0; k < network_process[i - 1]; k++)
                                    {
                                        z += sigmoid(activations[i - 1][k]) * matrix_weights[j * network_process[i - 1] + k];
                                    }
                                    matrix_activations[j] = z + process_biases[i][j];
                                }
                            }
                            MPI_Gather(matrix_activations, network_process[i] / size_process, MPI_DOUBLE, activations[i], network_process[i] / size_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                            delete[] matrix_weights;
                            delete[] matrix_activations;
                        }
                    }

                    if (rank_process == 0)
                    {
                        //Посчет предсказания
                        double max = sigmoid(activations[L - 1][0]);
                        int predict = 0;
                        for (int i = 1; i < network_process[L - 1]; i++)
                        {
                            if (max < sigmoid(activations[L - 1][i]))
                            {
                                max = sigmoid(activations[L - 1][i]);
                                predict = i;
                            }
                        }
                        if (predict == train_data[v].y)
                        {
                            right_answers_data++;
                        }
                    }
                    //Обратное распространение ошибки
                    if (rank_process == 0)
                    {
                        t1_error = MPI_Wtime();
                        for (int i = 0; i < network_process[L - 1]; i++)
                        {
                            if (i != train_data[v].y)
                            {

                                delta[L - 1][i] = sigmoid(activations[L - 1][i]) * derivative_sigmoid(activations[L - 1][i]);
                            }
                            else
                            {
                                delta[L - 1][i] = (sigmoid(activations[L - 1][i]) - train_data[v].y) * derivative_sigmoid(activations[L - 1][i]);
                            }
                        }
                    }
                    MPI_Bcast(delta[L - 1], network_process[L - 1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    for (int k = L - 2; k > 0; k--)
                    {
                        double* new_delta = new double[network_process[k] / size_process];
#pragma omp parallel num_threads(size_threads)
                        {
#pragma omp for schedule(static, 2)
                            for (int i = 0; i < network_process[k] / size_process; i++)
                            {
                                double sum = 0;
                                for (int j = 0; j < network_process[k + 1]; j++)
                                {
                                    sum += process_weights[k + 1][i * network_process[k + 1] + j] * delta[k + 1][j];
                                }
                                new_delta[i] = derivative_sigmoid(activations[k][i + rank_process * network_process[k] / size_process]) * sum;

                            }
                        }
                        MPI_Allgather(new_delta, network_process[k] / size_process, MPI_DOUBLE, delta[k], network_process[k] / size_process, MPI_DOUBLE, MPI_COMM_WORLD);
                        delete[] new_delta;
                    }
                    if (rank_process == 0)
                    {
                        t2_error = MPI_Wtime();
                        time_error += t2_error - t1_error;
                    }
                }
                //СГД
                if (rank_process == 0)
                {
                    t1_SGD = MPI_Wtime();
                }
                for (int k = 1; k < L; k++)
                {
                    double learning_rate = 0.35;
#pragma omp parallel num_threads(size_threads)
                    {
#pragma omp for schedule(static, 2)
                        for (int i = 0; i < network_process[k - 1] / size_process; i++)
                        {
                            for (int j = 0; j < network_process[k]; j++)
                            {
                                process_weights[k][i * network_process[k] + j] = process_weights[k][i * network_process[k] + j] - learning_rate * derivative_sigmoid(activations[k][i + rank_process * network_process[k - 1] / size_process]) * delta[k][j];
                            }
                        }
#pragma omp for schedule(static, 2)
                        for (int j = 0; j < network_process[k] / size_process; j++)
                        {
                            process_biases[k][j] = process_biases[k][j] - learning_rate * delta[k][j + rank_process * network_process[k] / size_process];
                        }
                    }
                }
                if (rank_process == 0)
                {
                    t2_SGD = MPI_Wtime();
                    time_SGD += t2_SGD - t1_SGD;
                }
            }
            if (rank_process == 0)
            {
                //cout << "Back propagation error: " << time_error / (size_mini_batch * count_batch * p) * 1000000 << " s * 10^-6. ";
                //cout << "SGD: " << time_SGD / (count_batch * p) * 1000000 << " s * 10^-6." << endl;
            }
        }

        delete[] activations[0];
        for (int i = 1; i < L; i++)
        {
            delete[] process_weights[i];
            delete[] process_biases[i];
            delete[] activations[i];
            delete[] delta[i];
        }

        delete[] process_weights;
        delete[] process_biases;
        delete[] delta;
        delete[] activations;


        if (rank_process == 0)
        {
            cout << "\nMean time:" << endl;
            cout << "Back propagation error: " << time_error / (size_mini_batch * count_batch * epoch) * 1000000 << " s * 10^-6. ";
            cout << "SGD: " << time_SGD / (count_batch * epoch) * 1000000 << " s * 10^-6." << endl << endl;
        }
    }
    

    MPI_Finalize();
}

