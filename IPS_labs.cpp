#include <chrono>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_max.h>
#include <cilk/reducer_min.h>
#include <cilk/reducer_opadd.h>
#include <cilk/reducer_vector.h>
using namespace std::chrono;

#define LAB1

#ifdef LAB1
/// Функция ReducerMinTest() определяет минимальный элемент массива,
/// переданного ей в качестве аргумента, и его позицию
/// mass_pointer - указатель исходный массив целых чисел
/// size - количество элементов в массиве
void ReducerMinTest(int* mass_pointer, const long size)
{
    cilk::reducer<cilk::op_min_index<long, int>> minimum;
    cilk_for(long i = 0; i < size; ++i)
    {
        minimum->calc_min(i, mass_pointer[i]);
    }
    printf("Minimal element = %d has index = %d\n\n",
        minimum->get_reference(), minimum->get_index_reference());
}

/// Функция ReducerMaxTest() определяет максимальный элемент массива,
/// переданного ей в качестве аргумента, и его позицию
/// mass_pointer - указатель исходный массив целых чисел
/// size - количество элементов в массиве
void ReducerMaxTest(int* mass_pointer, const long size)
{
    cilk::reducer<cilk::op_max_index<long, int>> maximum;
    cilk_for(long i = 0; i < size; ++i)
    {
        maximum->calc_max(i, mass_pointer[i]);
    }
    printf("Maximal element = %d has index = %d\n\n",
        maximum->get_reference(), maximum->get_index_reference());
}

/// Функция ParallelSort() сортирует массив в порядке возрастания
/// begin - указатель на первый элемент исходного массива
/// end - указатель на последний элемент исходного массива
void ParallelSort(int* begin, int* end)
{
    if (begin != end)
    {
        --end;
        int* middle = std::partition(begin, end, std::bind2nd(std::less<int>(), *end));
        std::swap(*end, *middle);
        cilk_spawn ParallelSort(begin, middle);
        ParallelSort(++middle, ++end);
        cilk_sync;
    }
}

void CompareForAndCilk_For(size_t sz)
{
    std::vector<int> vec;
    auto t0 = high_resolution_clock::now();
    for (size_t i = 0; i < sz; ++i)
        vec.push_back(rand() % 20000 + 1);

    auto t1 = high_resolution_clock::now();
    const duration<double> duration_vec = t1 - t0;

    cilk::reducer<cilk::op_vector<int>> red_vec;
    t0 = high_resolution_clock::now();
    cilk_for(size_t i = 0; i < sz; ++i)
        red_vec->push_back(rand() % 20000 + 1);

    t1 = high_resolution_clock::now();
    const duration<double> duration_red_vec = t1 - t0;

    printf("Size of array: %d\n", sz);
    printf("std::vector time: %f seconds\n", duration_vec.count());
    printf("cilk::reducer time: %f seconds\n", duration_red_vec.count());
}


void run_lab1()
{
    srand((unsigned)time(0));

    // устанавливаем количество работающих потоков = 4
    __cilkrts_set_param("nworkers", "4");

    long i;
    const long mass_size = 1000000;
    int* mass_begin, * mass_end;
    int* mass = new int[mass_size];

    for (i = 0; i < mass_size; ++i)
    {
        mass[i] = (rand() % 25000) + 1;
    }

    mass_begin = mass;
    mass_end = mass_begin + mass_size;

    printf("Unsorted:\n");
    ReducerMinTest(mass, mass_size);
    ReducerMaxTest(mass, mass_size);

    auto t0 = high_resolution_clock::now();
    ParallelSort(mass_begin, mass_end);
    auto t1 = high_resolution_clock::now();
    duration<double> duration = t1 - t0;
    printf("Size of array: %d\n", mass_size);
    printf("Duration is %f seconds\n", duration.count());

    printf("Sorted:\n");
    ReducerMinTest(mass, mass_size);
    ReducerMaxTest(mass, mass_size);

    delete[]mass;

    auto sizes = { 1000000, 100000, 10000, 1000, 500, 100, 50, 10 };
    for (auto sz : sizes)
    {
        CompareForAndCilk_For(sz);
        printf("\n");
    }
}

int main()
{
    run_lab1();
}
#endif // LAB1


#ifdef LAB2

// количество строк в исходной квадратной матрице
const int MATRIX_SIZE = 1500;

/// Функция InitMatrix() заполняет переданную в качестве 
/// параметра квадратную матрицу случайными значениями
/// matrix - исходная матрица СЛАУ
void InitMatrix(double** matrix)
{
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        matrix[i] = new double[MATRIX_SIZE + 1];
    }

    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        for (int j = 0; j <= MATRIX_SIZE; ++j)
        {
            matrix[i][j] = rand() % 2500 + 1;
        }
    }
}

/// Функция SerialGaussMethod() решает СЛАУ методом Гаусса 
/// matrix - исходная матрица коэффиициентов уравнений, входящих в СЛАУ,
/// последний столбей матрицы - значения правых частей уравнений
/// rows - количество строк в исходной матрице
/// result - массив ответов СЛАУ
void SerialGaussMethod(double** matrix, const int rows, double* result)
{
    int k;
    double koef;

    auto t0 = high_resolution_clock::now();
    // прямой ход метода Гаусса
    for (k = 0; k < rows; ++k)
    {
        //
        for (int i = k + 1; i < rows; ++i)
        {
            koef = -matrix[i][k] / matrix[k][k];

            for (int j = k; j <= rows; ++j)
            {
                matrix[i][j] += koef * matrix[k][j];
            }
        }
    }
    auto t1 = high_resolution_clock::now();

    // обратный ход метода Гаусса
    result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];

    for (k = rows - 2; k >= 0; --k)
    {
        result[k] = matrix[k][rows];

        for (int j = k + 1; j < rows; ++j)
        {
            result[k] -= matrix[k][j] * result[j];
        }

        result[k] /= matrix[k][k];
    }

    duration<double> dur = t1 - t0;
    printf("Serial version. Forward elimination time: %f\n", dur.count());
}

/// Функция ParallelSerialGaussMethod() решает СЛАУ методом Гаусса 
/// matrix - исходная матрица коэффиициентов уравнений, входящих в СЛАУ,
/// последний столбей матрицы - значения правых частей уравнений
/// rows - количество строк в исходной матрице
/// result - массив ответов СЛАУ
void ParallelSerialGaussMethod(double** matrix, const int rows, double* result)
{
    int k;

    auto t0 = high_resolution_clock::now();
    // прямой ход метода Гаусса
    for (k = 0; k < rows; ++k)
    {
        cilk_for(int i = k + 1; i < rows; ++i)
		{
			double koef = -matrix[i][k] / matrix[k][k];
			for (int j = k; j <= rows; ++j)
			{
				matrix[i][j] += koef * matrix[k][j];
			}
		}
	}
    auto t1 = high_resolution_clock::now();

    // обратный ход метода Гаусса
    result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];

    for (k = rows - 2; k >= 0; --k)
    {
        cilk::reducer_opadd<double> res_k(matrix[k][rows]);

        //result[k] = matrix[k][rows];

        cilk_for(int j = k + 1; j < rows; ++j)
        {
            res_k -= matrix[k][j] * result[j];
            //result[k] -= matrix[k][j] * result[j];
        }

        result[k] = res_k->get_value() / matrix[k][k];
        //result[k] /= matrix[k][k];
    }

    duration<double> dur = t1 - t0;
    printf("Parallel version. Forward elimination time: %f\n", dur.count());
}

void run_lab2()
{
    srand((unsigned)time(0));

    int i;

    const bool is_testing = true;
    const int test_matrix_lines = is_testing ? 4 : MATRIX_SIZE; // кол-во строк в матрице, приводимой в качестве примера

    double** test_matrix = new double* [test_matrix_lines];

    // цикл по строкам
    for (i = 0; i < test_matrix_lines; ++i)
    {
        // (test_matrix_lines + 1)- количество столбцов в тестовой матрице,
        // последний столбец матрицы отведен под правые части уравнений, входящих в СЛАУ
        test_matrix[i] = new double[test_matrix_lines + 1];
    }

    // массив решений СЛАУ
    double* result = new double[test_matrix_lines];

    if (is_testing)
    {
        // инициализация тестовой матрицы
        test_matrix[0][0] = 2; test_matrix[0][1] = 5;  test_matrix[0][2] = 4;  test_matrix[0][3] = 1;  test_matrix[0][4] = 20;
        test_matrix[1][0] = 1; test_matrix[1][1] = 3;  test_matrix[1][2] = 2;  test_matrix[1][3] = 1;  test_matrix[1][4] = 11;
        test_matrix[2][0] = 2; test_matrix[2][1] = 10; test_matrix[2][2] = 9;  test_matrix[2][3] = 7;  test_matrix[2][4] = 40;
        test_matrix[3][0] = 3; test_matrix[3][1] = 8;  test_matrix[3][2] = 9;  test_matrix[3][3] = 2;  test_matrix[3][4] = 37;
    }
    else
    {
        InitMatrix(test_matrix);
    }

    SerialGaussMethod(test_matrix, test_matrix_lines, result);
    ParallelSerialGaussMethod(test_matrix, test_matrix_lines, result);

    for (i = 0; i < test_matrix_lines; ++i)
    {
        delete[]test_matrix[i];
    }

    printf("Solution:\n");

    for (i = 0; i < test_matrix_lines; ++i)
    {
        printf("x(%d) = %lf\n", i, result[i]);
    }

    delete[] result;
}

int main()
{
    run_lab2();
}
#endif // LAB2


#ifdef LAB3

enum class eprocess_type
{
    by_rows = 0,
    by_cols
};

void InitMatrix(double** matrix, const size_t numb_rows, const size_t numb_cols)
{
    for (size_t i = 0; i < numb_rows; ++i)
    {
        for (size_t j = 0; j < numb_cols; ++j)
        {
            matrix[i][j] = rand() % 5 + 1;
        }
    }
}

void PrintMatrix(double** matrix, const size_t numb_rows, const size_t numb_cols)
{
    printf("Generated matrix:\n");
    for (size_t i = 0; i < numb_rows; ++i)
    {
        for (size_t j = 0; j < numb_cols; ++j)
        {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

void FindAverageValues(eprocess_type proc_type, double** matrix, const size_t numb_rows, const size_t numb_cols, double* average_vals)
{
    switch (proc_type)
    {
        case eprocess_type::by_rows:
        {
            cilk_for(size_t i = 0; i < numb_rows; ++i)
            {

                //double sum( 0.0 );
                cilk::reducer_opadd<double> sum(0.0);
                cilk_for(size_t j = 0; j < numb_cols; ++j)
                {
                    sum += matrix[i][j];
                }
                average_vals[i] = sum.get_value() / numb_cols;
            }
            break;
        }
        case eprocess_type::by_cols:
        {
            cilk_for(size_t j = 0; j < numb_cols; ++j)
            {
                //double sum( 0.0 );
                cilk::reducer_opadd<double> sum(0.0);
                cilk_for(size_t i = 0; i < numb_rows; ++i)
                {
                    sum += matrix[i][j];
                }
                average_vals[j] = sum.get_value() / numb_rows;
            }
            break;
        }
        default:
            return;
    }
}

void PrintAverageVals(eprocess_type proc_type, double* average_vals, const size_t dimension)
{
    switch (proc_type)
    {
    case eprocess_type::by_rows:
    {
        printf("\nAverage values in rows:\n");
        for (size_t i = 0; i < dimension; ++i)
        {
            printf("Row %u: %lf\n", i, average_vals[i]);
        }
        break;
    }
    case eprocess_type::by_cols:
    {
        printf("\nAverage values in columns:\n");
        for (size_t i = 0; i < dimension; ++i)
        {
            printf("Column %u: %lf\n", i, average_vals[i]);
        }
        break;
    }
    default:
    {
        throw("Incorrect value for parameter 'proc_type' in function PrintAverageVals() call!");
    }
    }
}

int run_lab3()
{
    __cilkrts_set_param("nworkers", "4");

    const unsigned ERROR_STATUS = -1;
    const unsigned OK_STATUS = 0;

    unsigned status = OK_STATUS;

    try
    {
        srand((unsigned)time(0));

        const size_t numb_rows = 2;
        const size_t numb_cols = 3;

        double** matrix = new double* [numb_rows];
        for (size_t i = 0; i < numb_rows; ++i)
        {
            matrix[i] = new double[numb_cols];
        }

        double* average_vals_in_rows = new double[numb_rows];
        double* average_vals_in_cols = new double[numb_cols];

        InitMatrix(matrix, numb_rows, numb_cols);

        PrintMatrix(matrix, numb_rows, numb_cols);

        std::thread first_thr(FindAverageValues, eprocess_type::by_rows, matrix, numb_rows, numb_cols, average_vals_in_rows);
        std::thread second_thr(FindAverageValues, eprocess_type::by_cols, matrix, numb_rows, numb_cols, average_vals_in_cols);

        first_thr.join();
        second_thr.join();

        PrintAverageVals(eprocess_type::by_rows, average_vals_in_rows, numb_rows);
        PrintAverageVals(eprocess_type::by_cols, average_vals_in_cols, numb_cols);

        delete[] average_vals_in_cols;
        delete[] average_vals_in_rows;

        for (size_t i = 0; i < numb_rows; ++i)
            delete[] matrix[i];
        delete[] matrix;
    }
    catch (std::exception & except)
    {
        //printf( "Error occured!\n" );
        except.what();
        status = ERROR_STATUS;
    }

    return status;
}


int main()
{
    return run_lab3();
}

#endif // LAB3


#ifdef INDIVIDUAL_TASK

double fun(double x)
{
    return 4.0 / sqrt(4.0 - x * x);
}

double integrate(double a, double b, size_t N)
{
    const double h = (b - a) / N;
    double sum = 0.0f;
    for (size_t i = 0; i < N; ++i)
        sum += fun(a + i * h);

    return sum * h;
}

double parallel_integrate(double a, double b, size_t n)
{
    const double h = (b - a) / n;
    cilk::reducer_opadd<double> sum(0.0);
    cilk_for(size_t i = 0; i < n; ++i)
        sum += fun(a + i * h);

    return sum->get_value() * h;
}

template<class Fun, class... Args>
auto measure_time(const std::string& name, Fun&& fun, Args&&... args)
{
    auto t0 = high_resolution_clock::now();
    const auto ans = fun(std::forward<Args>(args)...);
    auto t1 = high_resolution_clock::now();
    duration<double> est_time = t1 - t0;

    std::cout << name << " ans: " << ans << '\n' << "Spended time: " << est_time.count() << '\n';

    return est_time.count();
}

void run_task()
{
    const double a = 0.0;
    const double b = 1.0;
    const size_t N = 10000000;

    for (size_t i = 0; i < 5; ++i)
    {
        const size_t n = std::pow(10, 5 + i);
        std::cout << "n = " << n << '\n';
        const auto non_parallel_time = measure_time("Serial", integrate, a, b, n);
        const auto parallel_time = measure_time("Parallel", parallel_integrate, a, b, n);

        const double boost_time = non_parallel_time / parallel_time;
        std::cout << "Boost: " << boost_time << '\n' << "--------------------------------------\n";
    }
}

int main()
{
    run_task();
}

#endif // INDIVIDUAL_TASK
