#ifndef _LINEAR_ALGEBRA_
#define _LINEAR_ALGEBRA_

#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <iomanip> 
#include <iterator>
#include <thread>
#include <vector>
#include "function.h"
#include "solvers.h"

template <typename T>
class vec;

template <typename T>
class Matrix
{
public:
    Matrix(int row, int col);
    Matrix(Matrix<T> &other);
    Matrix& operator=(Matrix<T> &other);
    Matrix<T> operator+(const Matrix<T> &other) const;
    Matrix<T> operator-(const Matrix<T> &other)const;
    Matrix<T> operator*(const Matrix<T> &other)const;
    vec<T> operator*(const vec<T> &v)const;
    void input();
    T& operator()(int n, int m);
    void display();
    void display(std::string name);

    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    ~Matrix();

    //-------friend function ----------------------------
    template <typename U> // Changed from T to U
    friend Matrix<U> slice(Matrix<U> &M, int row_start, int row_end, int col_start, int col_end);
    
    template <typename U> // Changed from T to U
    friend Matrix<U> diagonalPreconditioner(Matrix<U> &M);
    
    template <typename U> // Changed from T to U
    friend Matrix<U> transpose(Matrix<U> &A);
    
    template <typename U> // Changed from T to U
    friend vec<U> cg(Matrix<U> &A, vec<U> &x, vec<U> &b, int max_iteration, double tolerance);
    
    template <typename U> // Changed from T to U
    friend vec<U> pcg(Matrix<U> &A, vec<U> &x, vec<U> &b, int max_iteration, double tolerance);

    template<typename U>
    friend vec<U> gausselimination(Matrix<U> &A,vec<U> &b);
    
    template <typename U> // Changed from T to U
    friend bool symmetrycheck(Matrix<U> &A);
    
    template <typename U> // Changed from T to U
    friend vec<U> flatten(Matrix<U> M);

private:
    int row;
    int col;
    T **arr2;
};

template <typename T>
class vec
{
public:
    vec(int size);
    vec();
    //vec() : arr1(nullptr), size(0) {}
    vec(vec<T> &other);
    vec& operator=(vec<T> &other);
    //Assignment operator to set all elements to a single value
    vec& operator=(const T& value); 
    vec<T> operator+(const vec<T> &other) const;
    vec<T> operator-(const vec<T> &other) const;
    T operator*(const vec<T> &other) const;
    vec<T> &operator/=(double scalar);
    vec<T> operator/(double scalar) const;
    vec<T> operator*(double scalar);
    void display();
    void display(std::string name);
    void set(double value);
    T& operator()(int n);
    double norm();
    int getSize();
    ~vec();

    vec(vec&& other) noexcept;
    vec& operator=(vec&& other) noexcept;

    //-------friend function ----------------------------
    template <typename U> // Changed from T to U
    friend vec<U> slice(vec<U> &v, int start, int end);
    
    template <typename U> // Changed from T to U
    friend class Matrix;
    
    template <typename U> // Changed from T to U
    friend vec<U> cg(Matrix<U> &A, vec<U> &x, vec<U> &b, int max_iteration, double tolerance);
    
    template <typename U> // Changed from T to U
    friend vec<U> pcg(Matrix<U> &A, vec<U> &x, vec<U> &b, int max_iteration, double tolerance);
    
    template <typename U> // Changed from T to U
    friend Matrix<U> unflatten(vec<U> v, int row, int col);
    
    template <typename U> // Changed from T to U
    friend Matrix<U> unflatten(vec<U> v);

    template<typename U>
    friend vec<U> gausselimination(Matrix<U> &A, vec<U> &b);

private:
    int size;
    T *arr1;
};


/*-----------------------------Declaration of Methods---------------------------------------*/
/*
the compiler needs to see template declaration and defination together, ifnot it throws linking error 
so it is best advised to use single header or implrmnation file declaration when using template.
will look into separte declartion in future, Now this library is header only
*/

template <typename T>
Matrix<T>::Matrix(int row, int col) : row(row), col(col)
{
    arr2 = new T*[row];
    for (int i = 0; i < row; ++i)
    {
        arr2[i] = new T[col];
        for (int j = 0; j < col; ++j)
        {
            arr2[i][j] = 0.0;
        }
    }
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &other) : row(other.row), col(other.col)
{
    arr2 = new T*[row];
    for (int i = 0; i < row; ++i)
    {
        arr2[i] = new T[col];
        std::copy(other.arr2[i], other.arr2[i] + col, arr2[i]);
    }
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &other)
{
    if (this == &other)
    {
        return *this;
    }

    if (row != other.row || col != other.col)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            arr2[i][j] = other.arr2[i][j];
        }
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other)const
{
    if (row != other.row || col != other.col)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for addition");
    }
    Matrix result(row, col);
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            result.arr2[i][j] = arr2[i][j] + other.arr2[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other)const
{
    if (row != other.row || col != other.col)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction");
    }
    Matrix result(row, col);
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            result.arr2[i][j] = arr2[i][j] - other.arr2[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other)const
{
    if (col != other.row)
    {
        throw std::invalid_argument("Matrix multiplication is not possible");
    }
    Matrix result(row, other.col);
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < other.col; ++j)
        {
            result.arr2[i][j] = 0;
            for (int k = 0; k < col; ++k)
            {
                result.arr2[i][j] += arr2[i][k] * other.arr2[k][j];
            }
        }
    }
    return result;
}

template <typename T>
vec<T> Matrix<T>::operator*(const vec<T> &v)const
{
    if (col != v.size)
    {
        throw std::invalid_argument("Matrix column count must match vector size for multiplication");
    }

    vec<T> result(row);

    auto thread_task = [&](int start, int end)
    {
        for (int i = start; i < end; ++i)
        {
            T sum = 0;
            for (int j = 0; j < col; ++j)
            {
                sum += arr2[i][j] * v.arr1[j];
            }
            result.arr1[i] = sum;
        }
    };

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 0)
    {
        num_threads = 1;
    }

    int rows_per_thread = row / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads - 1; ++i)
    {
        threads.emplace_back(thread_task, i * rows_per_thread, (i + 1) * rows_per_thread);
    }

    thread_task((num_threads - 1) * rows_per_thread, row);

    for (auto &thread : threads)
    {
        thread.join();
    }

    return result;
}

template <typename T>
void Matrix<T>::input()
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            std::cin >> arr2[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::display()
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "-------------Matrix display----------------" << std::endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            std::cout << arr2[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------------------------------" << std::endl;
}

template <typename T>
void Matrix<T>::display(std::string name)
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "displaying matrix :"<<name<< std::endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            std::cout << arr2[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------------------------------" << std::endl;
}

template <typename T>
Matrix<T>::~Matrix()
{
    for (int i = 0; i < row; ++i)
    {
        delete[] arr2[i];
    }
    delete[] arr2;
}

template <typename T>
T& Matrix<T>::operator()(int n, int m)
{
    if (n < 0 || n >= row || m < 0 || m >= col)
    {
        throw std::out_of_range("Matrix subscript out of bounds");
    }
    return arr2[n][m];
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept : row(other.row), col(other.col), arr2(other.arr2)
{
    other.arr2 = nullptr;
    other.row = 0;
    other.col = 0;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept
{
    if (this != &other)
    {
        for (int i = 0; i < row; ++i)
        {
            delete[] arr2[i];
        }
        delete[] arr2;

        row = other.row;
        col = other.col;
        arr2 = other.arr2;

        other.arr2 = nullptr;
        other.row = 0;
        other.col = 0;
    }
    return *this;
}

// ------------------------ Vector ------------------------
/*
Declaration of methods of vector class
*/
template <typename T>
vec<T>::vec(int size) : size(size)
{
    arr1 = new T[size];
    for( int i =0 ; i < size ; i++)
    {
        arr1[i] = 0;
    }
}


template <typename T>
vec<T>::vec() : arr1(nullptr), size(0) {}

template <typename T>
vec<T>::vec(vec<T> &other) : size(other.size)
{
    arr1 = new T[size];
    std::copy(other.arr1, other.arr1 + size, arr1);
}

template <typename T>
vec<T>& vec<T>::operator=(vec<T> &other)
{
    if (this == &other)
    {
        return *this;
    }

    if (size != other.size)
    {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    std::copy(other.arr1, other.arr1 + size, arr1);
    return *this;
}

//********
template <typename T>
vec<T>& vec<T>::operator=(const T &value)
{
    for (int i = 0; i < size; ++i)
    {
        arr1[i] = value;
    }
    return *this;
}

//*********

template <typename T>
vec<T> vec<T>::operator+( const vec<T> &other)const 
{
    if (size != other.size)
    {
        throw std::invalid_argument("Vectors must have the same dimensions for addition");
    }

    vec result(size);
    for (int i = 0; i < size; ++i)
    {
        result.arr1[i] = arr1[i] + other.arr1[i];
    }
    return result;
}

template <typename T>
vec<T> vec<T>::operator-(const  vec<T> &other)const 
{
    if (size != other.size)
    {
        throw std::invalid_argument("Vectors must have the same dimensions for subtraction");
    }

    vec result(size);
    for (int i = 0; i < size; ++i)
    {
        result.arr1[i] = arr1[i] - other.arr1[i];
    }
    return result;
}

template <typename T>
T vec<T>::operator*(const  vec<T> &other)const 
{
    if (size != other.size)
    {
        throw std::invalid_argument("Vectors must have the same dimensions for dot product");
    }

    T result = 0;
    for (int i = 0; i < size; ++i)
    {
        result += arr1[i] * other.arr1[i];
    }
    return result;
}

template <typename T>
vec<T> vec<T>::operator*(double scalar)
{
    vec result(size);
    for (int i = 0; i < size; ++i)
    {
        result.arr1[i] = arr1[i] * scalar;
    }
    return result;
}

template <typename T>
vec<T>& vec<T>::operator/=(double scalar)
{
    for (int i = 0; i < size; ++i)
    {
        arr1[i] /= scalar;
    }
    return *this; // Return a reference to the modified vector
}

template <typename T>
vec<T> vec<T>::operator/(double scalar) const
{
    vec<T> result(size);
    for (int i = 0; i < size; ++i)
    {
        result.arr1[i] = arr1[i] / scalar;
    }
    return result;
}


template <typename T>
void vec<T>::display()
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "-------------Vector display----------------" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << arr1[i] << "\t";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
}

template <typename T>
void vec<T>::display(std::string name)
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout<<"displaying vector :"<<name<<std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << arr1[i] << "\t";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

}

template <typename T>
void vec<T>::set(double value)
{
    std::fill(arr1, arr1 + size, value);
}

template <typename T>
T& vec<T>::operator()(int n)
{
    if (n < 0 || n >= size)
    {
        throw std::out_of_range("Vector subscript out of bounds");
    }
    return arr1[n];
}

template <typename T>
double vec<T>::norm()
{
    return std::sqrt((*this) * (*this));
}

template <typename T>
int vec<T>::getSize()
{
    return size;
}

template <typename T>
vec<T>::~vec()
{
    delete[] arr1;
}

template <typename T>
vec<T>::vec(vec&& other) noexcept : size(other.size), arr1(other.arr1)
{
    other.arr1 = nullptr;
    other.size = 0;
}

template <typename T>
vec<T>& vec<T>::operator=(vec&& other) noexcept
{
    if (this != &other)
    {
        delete[] arr1;

        size = other.size;
        arr1 = other.arr1;

        other.arr1 = nullptr;
        other.size = 0;
    }
    return *this;
}



#endif
