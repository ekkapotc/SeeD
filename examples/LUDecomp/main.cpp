#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ad_core.hpp"

using namespace std;
using namespace std::chrono;

class LinearSolver {
public:
    template <typename T>
    static void luDecomposition(const Matrix<T>& A, Matrix<T>& L, Matrix<T>& U) {
        const std::size_t n = A.getRows();
        if (A.getCols() != n) throw std::invalid_argument("luDecomposition: A must be square");
        if (L.getRows() != n || L.getCols() != n || U.getRows() != n || U.getCols() != n) {
            throw std::invalid_argument("luDecomposition: L and U must match A's shape");
        }
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t k = i; k < n; k++) {
                T sum = T(0.0);
                for (std::size_t j = 0; j < i; j++) sum += (L[i][j] * U[j][k]);
                U[i][k] = A[i][k] - sum;
            }
            if (U[i][i] == 0.0) {
                throw std::domain_error("luDecomposition: zero pivot at row " + std::to_string(i) +
                                        " (matrix is singular or needs pivoting)");
            }
            for (std::size_t k = i; k < n; k++) {
                if (i == k) {
                    L[i][i] = T(1.0);
                } else {
                    T sum = T(0.0);
                    for (std::size_t j = 0; j < i; j++) sum += (L[k][j] * U[j][i]);
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }
    }

    template <typename T>
    static Vector<T> solveLU(const Matrix<T>& L, const Matrix<T>& U, const Vector<T>& b) {
        const std::size_t n = L.getRows();
        if (b.size() != n) throw std::invalid_argument("solveLU: b has the wrong length");
        Vector<T> y(n), x(n);

        for (std::size_t i = 0; i < n; i++) {
            T sum = T(0.0);
            for (std::size_t j = 0; j < i; j++) sum += L[i][j] * y[j];
            y[i] = (b[i] - sum) / L[i][i];
        }
        for (std::size_t i = n; i-- > 0;) {
            T sum = T(0.0);
            for (std::size_t j = i + 1; j < n; j++) sum += U[i][j] * x[j];
            x[i] = (y[i] - sum) / U[i][i];
        }
        return x;
    }

    // Takes A and b BY VALUE. Gaussian elimination is destructive; the previous
    // signature took non-const references, so calling it in a benchmark loop
    // fed every iteration after the first an already-triangularised matrix and
    // returned garbage.
    template <typename T>
    static Vector<T> solveGaussian(Matrix<T> A, Vector<T> b) {
        const std::size_t n = A.getRows();
        if (A.getCols() != n) throw std::invalid_argument("solveGaussian: A must be square");
        if (b.size() != n) throw std::invalid_argument("solveGaussian: b has the wrong length");

        for (std::size_t i = 0; i < n; i++) {
            if (A[i][i] == 0.0) {
                throw std::domain_error("solveGaussian: zero pivot at row " + std::to_string(i) +
                                        " (matrix is singular or needs pivoting)");
            }
            for (std::size_t k = i + 1; k < n; k++) {
                T factor = A[k][i] / A[i][i];
                for (std::size_t j = i; j < n; j++) A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }
        Vector<T> x(n);
        for (std::size_t i = n; i-- > 0;) {
            T sum = T(0.0);
            for (std::size_t j = i + 1; j < n; j++) sum += A[i][j] * x[j];
            x[i] = (b[i] - sum) / A[i][i];
        }
        return x;
    }
};

// Largest |A*x - b| over all rows, in plain double against the original system.
static double residual(const std::vector<std::vector<double>>& A, const Vector<ADouble>& x,
                       const std::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < A.size(); i++) {
        double s = 0.0;
        for (std::size_t j = 0; j < A[i].size(); j++) s += A[i][j] * x[j].val;
        worst = std::fmax(worst, std::fabs(s - b[i]));
    }
    return worst;
}

// Materialise the AD copies of the problem. These must be built *after* the
// tape reset that precedes the solve: an ADouble created in an earlier tape
// generation no longer has a valid id.
static Matrix<ADouble> make_A(const std::vector<std::vector<double>>& src) {
    Matrix<ADouble> A(src.size(), src.size());
    for (std::size_t i = 0; i < src.size(); i++)
        for (std::size_t j = 0; j < src.size(); j++) A[i][j] = src[i][j];
    return A;
}

static Vector<ADouble> make_b(const std::vector<double>& src) {
    Vector<ADouble> b(src.size());
    for (std::size_t i = 0; i < src.size(); i++) b[i] = src[i];
    return b;
}

int main() {
    const std::size_t N = 100;
    const std::size_t M = 50;

    mt19937 gen(42);
    uniform_real_distribution<> dis(-10.0, 10.0);

    // Problem data is kept in plain doubles so it survives tape resets.
    // Strictly diagonally dominant, so no pivoting is needed.
    std::vector<std::vector<double>> A_data(N, std::vector<double>(N, 0.0));
    for (std::size_t i = 0; i < N; i++) {
        double rowSum = 0;
        for (std::size_t j = 0; j < N; j++) {
            if (i != j) {
                A_data[i][j] = dis(gen);
                rowSum += std::abs(A_data[i][j]);
            }
        }
        A_data[i][i] = rowSum + 1.0;
    }

    std::vector<std::vector<double>> b_data(M, std::vector<double>(N, 0.0));
    for (std::size_t k = 0; k < M; k++)
        for (std::size_t i = 0; i < N; i++) b_data[k][i] = dis(gen);

    cout << "--- SeeD Performance Benchmark ---\n";
    cout << "Matrix Size: " << N << "x" << N << " | Right-hand sides: " << M << "\n\n";

    // --- TEST 1: Gaussian elimination (full O(n^3) factor + solve per RHS) ---
    // The tape is reset between right-hand sides and the AD inputs are rebuilt
    // outside the timed region. Without the resets this loop accumulated ~6.7e7
    // tape nodes and peaked above 2 GB of RSS, so the timing measured the
    // allocator rather than the solver.
    double gauss_worst = 0.0;
    nanoseconds gauss_time{0};
    for (std::size_t k = 0; k < M; k++) {
        tape().reset();
        Matrix<ADouble> Ak = make_A(A_data);
        Vector<ADouble> bk = make_b(b_data[k]);

        auto t0 = high_resolution_clock::now();
        Vector<ADouble> x = LinearSolver::solveGaussian(Ak, bk);
        gauss_time += high_resolution_clock::now() - t0;

        gauss_worst = std::fmax(gauss_worst, residual(A_data, x, b_data[k]));
    }
    auto duration_gauss = duration_cast<milliseconds>(gauss_time);
    cout << "Gaussian  (factor+solve x" << M << "):      " << setw(6) << duration_gauss.count()
         << " ms   max |Ax-b| = " << scientific << setprecision(3) << gauss_worst << defaultfloat
         << "\n";

    // --- TEST 2: LU decomposition (one O(n^3) factorisation + M O(n^2) solves) ---
    double lu_worst = 0.0;
    tape().reset();
    Matrix<ADouble> A = make_A(A_data);
    Matrix<ADouble> L(N, N), U(N, N);

    auto t0 = high_resolution_clock::now();
    LinearSolver::luDecomposition(A, L, U);
    for (std::size_t k = 0; k < M; k++) {
        Vector<ADouble> bk = make_b(b_data[k]);
        Vector<ADouble> x = LinearSolver::solveLU(L, U, bk);
        lu_worst = std::fmax(lu_worst, residual(A_data, x, b_data[k]));
    }
    auto duration_lu = duration_cast<milliseconds>(high_resolution_clock::now() - t0);
    cout << "LU        (factor once + " << M << " solves): " << setw(6) << duration_lu.count()
         << " ms   max |Ax-b| = " << scientific << setprecision(3) << lu_worst << defaultfloat
         << "\n";

    cout << "> Speedup: "
         << (double)duration_gauss.count() / std::max(1.0, (double)duration_lu.count())
         << "x  (re-factorising per RHS vs. reusing one factorisation)\n";
    cout << "> Peak tape: " << tape().nodes.capacity() << " nodes, "
         << (tape().nodes.capacity() * sizeof(TapeNode) +
             tape().adjoints.capacity() * sizeof(double)) /
                (1024 * 1024)
         << " MB\n\n";

    tape().release();
    return 0;
}
