#include <iostream>
#include <vector>
#include <iomanip>
#include "../../inc/ad_core.hpp" 

using ad_type = ADouble; 

int main() {
    int n = 3;
    
    // 1. Initialize the Matrix A (Constant standard doubles)
    std::vector<std::vector<double>> A = {
        { 2.0, -1.0, -2.0},
        {-4.0,  6.0,  3.0},
        {-4.0, -2.0,  8.0}
    };

    // 2. Initialize the Vector b as Independent AD Variables
    std::vector<ad_type> b(n);
    b[0] = ad_type(1.0);
    b[1] = ad_type(2.0);
    b[2] = ad_type(3.0);

    // 3. Allocate L and U matrices
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));

    // 4. Perform LU Decomposition on Matrix A
    for (int i = 0; i < n; i++) {
        // Upper Triangular
        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += (L[i][j] * U[j][k]);
            }
            U[i][k] = A[i][k] - sum;
        }
        // Lower Triangular
        for (int k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1.0; 
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += (L[k][j] * U[j][i]);
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }

    // 5. Forward Substitution (L * y = b)
    // The AD tape begins tracking operations here
    std::vector<ad_type> y(n);
    for (int i = 0; i < n; i++) {
        ad_type sum = ad_type(0.0);
        for (int j = 0; j < i; j++) {
            sum = sum + (y[j] * L[i][j]);
        }
        y[i] = b[i] - sum;
    }

    // 6. Backward Substitution (U * x = y)
    std::vector<ad_type> x(n);
    for (int i = n - 1; i >= 0; i--) {
        ad_type sum = ad_type(0.0);
        for (int j = i + 1; j < n; j++) {
            sum = sum + (x[j] * U[i][j]);
        }
        x[i] = (y[i] - sum) / U[i][i];
    }

    // 7. Output the computed vector x
    std::cout << "--- Solution Vector (x) ---\n";
    for (int i = 0; i < n; i++) {
        // Replace .value() with your API's method to get the raw float/double
        std::cout << "x[" << i << "] = " << x[i].val << "\n";
    }
    std::cout << "\n";

    // 8. Compute and Output the Jacobian (dx/db)
    // Since Ax = b, the Jacobian dx/db should exactly equal A_inverse
    
    std::vector<std::vector<double>> jacobian(n, std::vector<double>(n, 0.0));
    
    std::cout << "--- Jacobian Matrix (dx/db) ---\n";
    for (int i = 0; i < n; ++i) {
        
        // A. Trigger reverse-mode AD pass for the i-th element.[cite: 3]
        global_tape.compute_adjoints(x[i].id); 

        std::cout << "[";
        // B. Extract the sensitivities (gradients) with respect to each input element of vector b
        for (int j = 0; j < n; ++j) {
            jacobian[i][j] = global_tape.adjoints[b[j].id]; //[cite: 3]
            
            // Print the extracted gradient element for the Jacobian row
            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << jacobian[i][j];
            if (j < n - 1) std::cout << ", ";
        }
        std::cout << " ]\n";
    }    
    
    return 0;
}
