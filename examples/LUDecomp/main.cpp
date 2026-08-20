#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

#include "ad_core.hpp"

using namespace std;
using namespace std::chrono;


class LinearSolver {
public:
    template <typename T>
    static void luDecomposition(const Matrix<T>& A, Matrix<T>& L, Matrix<T>& U) {
        int n = A.getRows();
        for (int i = 0; i < n; i++) {
            for (int k = i; k < n; k++) {
                T sum = T(0.0);
                for (int j = 0; j < i; j++) sum += (L[i][j] * U[j][k]);
                U[i][k] = A[i][k] - sum;
            }
            for (int k = i; k < n; k++) {
                if (i == k) {
                    L[i][i] = T(1.0); 
                } else {
                    T sum = T(0.0);
                    for (int j = 0; j < i; j++) sum += (L[k][j] * U[j][i]);
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }
    }

    template <typename T>
    static Vector<T> solveLU(const Matrix<T>& L, const Matrix<T>& U, const Vector<T>& b) {
        int n = L.getRows();
        Vector<T> y(n), x(n);

        for (int i = 0; i < n; i++) {
            T sum = T(0.0);
            for (int j = 0; j < i; j++) sum += L[i][j] * y[j];
            y[i] = (b[i] - sum) / L[i][i];
        }
        for (int i = n - 1; i >= 0; i--) {
            T sum = T(0.0);
            for (int j = i + 1; j < n; j++) sum += U[i][j] * x[j];
            x[i] = (y[i] - sum) / U[i][i];
        }
        return x;
    }

    template <typename T>
    static Vector<T> solveGaussian(Matrix<T> & A, Vector<T> & b) {
        int n = A.getRows();
        for (int i = 0; i < n; i++) {
            for (int k = i + 1; k < n; k++) {
                T factor = A[k][i] / A[i][i];
                for (int j = i; j < n; j++) A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }
        Vector<T> x(n);
        for (int i = n - 1; i >= 0; i--) {
            T sum = 0.0;
            for (int j = i + 1; j < n; j++) sum += A[i][j] * x[j];
            x[i] = (b[i] - sum) / A[i][i];
        }
        return x;
    }
};


int main() {
    int N = 50;   
    int M = 200;  

    mt19937 gen(42);
    uniform_real_distribution<> dis(-10.0, 10.0);


    // Initialize strict diagonally dominant matrix using double
    Matrix<ADouble> A(N, N);
    for(int i = 0; i < N; i++) {
        double rowSum = 0;
        for(int j = 0; j < N; j++) {
            if(i != j) {
                A[i][j] = dis(gen);
                rowSum += std::abs(A[i][j].val);
            }
        }
        A[i][i] = rowSum + 1.0; 
    }

    std::vector<Vector<ADouble>> b_vectors;
    for(int k = 0; k < M; k++) {
        Vector<ADouble> b(N);
        for(int i = 0; i < N; i++) b[i] = dis(gen);
        b_vectors.push_back(b);
    }

    cout << "--- SeeD Performance Benchmark ---\n";
    cout << "Matrix Size: " << N << "x" << N << " | Iterations: " << M << "\n\n";

    // --- TEST 1: Gaussian Elimination ---
    auto start = high_resolution_clock::now();
    for(int k = 0; k < M; k++) LinearSolver::solveGaussian(A, b_vectors[k]);
    auto stop = high_resolution_clock::now();
    auto duration_gauss = duration_cast<milliseconds>(stop - start);
    cout << "Gaussian Time: " << duration_gauss.count() << " ms\n";

    // --- TEST 2: LU Decomposition ---
    start = high_resolution_clock::now();
    Matrix<ADouble> L(N, N), U(N, N);
    LinearSolver::luDecomposition(A, L, U);
    for(int k = 0; k < M; k++) LinearSolver::solveLU(L, U, b_vectors[k]);
    stop = high_resolution_clock::now();
    auto duration_lu = duration_cast<milliseconds>(stop - start);
    cout << "LU Time:       " << duration_lu.count() << " ms\n";
    cout << "> Speedup:     " << (double)duration_gauss.count() / std::max(1.0, (double)duration_lu.count()) << "x\n\n";
    
    return 0;
}
