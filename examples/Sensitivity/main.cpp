#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ad_core.hpp"

using ad_type = ADouble;

int main() {
    const std::size_t n = 3;

    // 1. Matrix A (constant standard doubles)
    std::vector<std::vector<double>> A = {{2.0, -1.0, -2.0}, {-4.0, 6.0, 3.0}, {-4.0, -2.0, 8.0}};

    // 2. Vector b as independent AD variables
    std::vector<ad_type> b;
    b.reserve(n);
    b.push_back(ad_type(1.0));
    b.push_back(ad_type(2.0));
    b.push_back(ad_type(3.0));

    // 3. L and U in plain doubles (A is constant, so it needs no tape nodes)
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));

    // 4. LU decomposition of A
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t k = i; k < n; k++) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; j++) sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }
        if (U[i][i] == 0.0) {
            std::cerr << "Zero pivot at row " << i << "; A needs pivoting.\n";
            return 1;
        }
        for (std::size_t k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1.0;
            } else {
                double sum = 0.0;
                for (std::size_t j = 0; j < i; j++) sum += (L[k][j] * U[j][i]);
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }

    // 5. Forward substitution (L * y = b). The tape starts tracking here.
    std::vector<ad_type> y(n);
    for (std::size_t i = 0; i < n; i++) {
        ad_type sum = ad_type(0.0);
        for (std::size_t j = 0; j < i; j++) sum = sum + (y[j] * L[i][j]);
        y[i] = b[i] - sum;  // L[i][i] == 1
    }

    // 6. Backward substitution (U * x = y)
    std::vector<ad_type> x(n);
    for (std::size_t i = n; i-- > 0;) {
        ad_type sum = ad_type(0.0);
        for (std::size_t j = i + 1; j < n; j++) sum = sum + (x[j] * U[i][j]);
        x[i] = (y[i] - sum) / U[i][i];
    }

    // 7. Solution vector
    std::cout << "--- Solution Vector (x) ---\n";
    for (std::size_t i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << x[i] << "\n";
    }
    std::cout << "\n";

    // 8. Jacobian dx/db. Since Ax = b, it should equal A inverse.
    std::vector<std::vector<double>> jacobian(n, std::vector<double>(n, 0.0));

    std::cout << "--- Jacobian Matrix (dx/db) ---\n";
    for (std::size_t i = 0; i < n; ++i) {
        // One reverse sweep per output component.
        tape().compute_adjoints(x[i].id);

        std::cout << "[";
        for (std::size_t j = 0; j < n; ++j) {
            jacobian[i][j] = b[j].grad();  // bounds-checked adjoint lookup

            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << jacobian[i][j];
            if (j < n - 1) std::cout << ", ";
        }
        std::cout << " ]\n";
    }

    tape().release();
    return 0;
}
