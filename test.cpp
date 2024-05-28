/*
Example code to solve some Linear system(symmetric)
*/
#include "include/slap.h"

#define n 5

int main()
{
    vec<double> b(n);
    vec<double> x(n);
    Matrix<double> A(n,n);

    //solving Ax = b when A is symmtric using cg,pcg and gaussian elimination

    // Define a 5x5 symmetric matrix
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 2; A(0,3) = 0; A(0,4) = 1;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 0; A(1,3) = 1; A(1,4) = 2;
    A(2,0) = 2; A(2,1) = 0; A(2,2) = 5; A(2,3) = 3; A(2,4) = 0;
    A(3,0) = 0; A(3,1) = 1; A(3,2) = 3; A(3,3) = 4; A(3,4) = 1;
    A(4,0) = 1; A(4,1) = 2; A(4,2) = 0; A(4,3) = 1; A(4,4) = 3;

    // Set the vector b with some values
    for(int i = 0; i < n; ++i)
    {
        b(i) = i + 1;
    }

    //initial solution vector
    x = 0;

    int max_iteration = 10000;
    double tol = 1e-4;
    vec<double> x_cg = cg(A,x,b,max_iteration,tol);
    vec<double> x_pcg = pcg(A,x,b,max_iteration,tol);
    vec<double> x_ge = gausselimination(A,b);

    x_cg.display("x_cg");
    x_cg.display("x_pcg");
    x_cg.display("x_ge");

}
