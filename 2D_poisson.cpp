#include <iostream>
#include <fstream>
#include <cmath>
#include "include/slap.h"

int main()
{
    int nx = 10;
    int ny = 10;
    int n = nx * ny;
    Matrix<double> A(n, n);
    Matrix<double> rhom(nx, ny);

    // Populate the matrix A
    for (int i = 0; i < n; ++i)
    {
        int row = i / nx;
        int col = i % nx;

        A(i, i) = -4; // Main diagonal

        if (col > 0) 
        {
            A(i, i - 1) = 1; // Left neighbor
        }

        if (col < nx - 1) 
        {
            A(i, i + 1) = 1; // Right neighbor
        }

        if (row > 0) 
        {
            A(i, i - nx) = 1; // Top neighbor
        }

        if (row < ny - 1)
        {
            A(i, i + nx) = 1; // Bottom neighbor
        }
    }

    // Populate the right-hand side vector with specific values
    for(int i = 0 ; i < nx; i++)
    {
        for(int j = 0 ; j < ny; j++)
        {
            rhom(i, j) = sin(i*j); // Initialize everywhere to zero
        }
    }
    
    // Solve the system using preconditioned conjugate gradient (PCG) method
    vec<double> rho= flatten(rhom);
    vec<double> x(n);

    x = cg(A, x, rho, 10000, 1e-4);
    vec<double> x_pcg = pcg(A, x, rho, 10000, 1e-4);
    vec<double> x_ge = gausselimination(A,rho);

    // Convert the solution vector back to a 2D array
    Matrix xarr = unflatten(x);
    Matrix gem = unflatten(x_ge);
    Matrix m_pcg = unflatten(x_pcg);

    xarr.display("cg_sol");
    m_pcg.display("pcg_sol");
    gem.display("gauss_elim_sol");


    return 0;
}



    

 