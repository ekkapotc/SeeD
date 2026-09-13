// SeeD regression suite.
//
//  * finite-difference gradient checks for every operator and function
//  * regression tests for each fixed bug, named by its review number
//
// Build and run with `make test`, or under sanitizers with `make asan`.
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_core.hpp"
#include "neural_net.hpp"

static int g_failures = 0;
static int g_checks = 0;

static void report(bool ok, const std::string& name, const std::string& detail = "") {
    ++g_checks;
    if (ok) {
        std::printf("  ok   %s\n", name.c_str());
    } else {
        ++g_failures;
        std::printf("  FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : " -- ",
                    detail.c_str());
    }
}

static void check_close(double got, double want, double tol, const std::string& name) {
    const bool ok = std::fabs(got - want) <= tol * std::fmax(1.0, std::fabs(want));
    report(ok, name, ok ? "" : "got " + std::to_string(got) + ", want " + std::to_string(want));
}

static void check_throws(const std::function<void()>& fn, const std::string& name) {
    try {
        fn();
    } catch (const std::exception&) {
        report(true, name);
        return;
    }
    report(false, name, "expected an exception, none thrown");
}

// ---------------------------------------------------------------------------
// Finite-difference gradient check for a unary AD function
// ---------------------------------------------------------------------------
static void grad_check(const std::string& name, const std::function<ADouble(const ADouble&)>& f_ad,
                       const std::function<double(double)>& f_d, double x0) {
    tape().reset();
    ADouble x(x0);
    ADouble y = f_ad(x);
    tape().compute_adjoints(y.id);
    const double ad = x.grad();

    const double h = 1e-6;
    const double fd = (f_d(x0 + h) - f_d(x0 - h)) / (2 * h);
    check_close(ad, fd, 1e-5, "d/dx " + name + " @ " + std::to_string(x0));
}

// ---------------------------------------------------------------------------
static void test_elementary() {
    std::puts("[elementary functions vs central differences]");
    grad_check("sin", [](const ADouble& v) { return sin(v); }, [](double v) { return std::sin(v); }, 0.7);
    grad_check("cos", [](const ADouble& v) { return cos(v); }, [](double v) { return std::cos(v); }, 0.7);
    grad_check("tan", [](const ADouble& v) { return tan(v); }, [](double v) { return std::tan(v); }, 0.7);
    grad_check("exp", [](const ADouble& v) { return exp(v); }, [](double v) { return std::exp(v); }, 0.7);
    grad_check("log", [](const ADouble& v) { return log(v); }, [](double v) { return std::log(v); }, 2.3);
    grad_check("sqrt", [](const ADouble& v) { return sqrt(v); }, [](double v) { return std::sqrt(v); }, 2.3);
    grad_check("tanh", [](const ADouble& v) { return tanh(v); }, [](double v) { return std::tanh(v); }, 0.7);
    grad_check("sigmoid", [](const ADouble& v) { return sigmoid(v); },
               [](double v) { return 1.0 / (1.0 + std::exp(-v)); }, 0.7);
    grad_check("relu(+)", [](const ADouble& v) { return relu(v); },
               [](double v) { return v > 0 ? v : 0.0; }, 0.7);
    grad_check("relu(-)", [](const ADouble& v) { return relu(v); },
               [](double v) { return v > 0 ? v : 0.0; }, -0.7);
    grad_check("abs(+)", [](const ADouble& v) { return abs(v); },
               [](double v) { return std::fabs(v); }, 0.7);
    grad_check("abs(-)", [](const ADouble& v) { return abs(v); },
               [](double v) { return std::fabs(v); }, -0.7);
    grad_check("neg", [](const ADouble& v) { return -v; }, [](double v) { return -v; }, 0.7);
    grad_check("pow(x,3)", [](const ADouble& v) { return pow(v, 3.0); },
               [](double v) { return std::pow(v, 3.0); }, 1.4);
    grad_check("x*x", [](const ADouble& v) { return v * v; }, [](double v) { return v * v; }, 3.0);
    grad_check("1/x", [](const ADouble& v) { return 1.0 / v; }, [](double v) { return 1.0 / v; }, 2.0);
    grad_check("x/(x+1)", [](const ADouble& v) { return v / (v + 1.0); },
               [](double v) { return v / (v + 1.0); }, 2.0);
    grad_check("composite", [](const ADouble& v) { return exp(sin(v * 2.0)) / (v + 3.0); },
               [](double v) { return std::exp(std::sin(v * 2.0)) / (v + 3.0); }, 0.6);
}

static void test_compound_assign() {
    std::puts("[compound assignment]");
    tape().reset();
    ADouble x(3.0);
    ADouble y = x;
    y *= x;  // y = x*x
    tape().compute_adjoints(y.id);
    check_close(y.val, 9.0, 1e-12, "x *= x value");
    check_close(x.grad(), 6.0, 1e-12, "x *= x gradient");

    tape().reset();
    ADouble a(2.0), b(5.0);
    ADouble s(0.0);
    s += a;
    s += b;
    s -= a;
    tape().compute_adjoints(s.id);
    check_close(s.val, 5.0, 1e-12, "+=/-= value");
    check_close(a.grad(), 0.0, 1e-12, "+=/-= gradient wrt a");
    check_close(b.grad(), 1.0, 1e-12, "+=/-= gradient wrt b");
}

// --- Bug 1: reset() must invalidate Parameter bindings -----------------------
static void test_bug1_reset_invalidates_parameters() {
    std::puts("[bug 1] reset() invalidates Parameter tape bindings");

    // y = p*p at p=3 -> dy/dp = 6, in two successive tape generations.
    Parameter p(3.0);
    for (int gen = 0; gen < 3; ++gen) {
        tape().reset();
        ADouble y = p.var() * p.var();
        tape().compute_adjoints(y.id);
        check_close(y.val, 9.0, 1e-12, "generation " + std::to_string(gen) + " value");
        check_close(p.grad(), 6.0, 1e-12, "generation " + std::to_string(gen) + " gradient");
        // deliberately NO update() between generations
    }

    // Full model: an inference pass between epochs must not perturb gradients.
    std::vector<double> X, Y;
    for (double xv = 0.0; xv <= 1.0; xv += 0.25) {
        X.push_back(xv);
        Y.push_back(2.5 * std::sin(1.2 * xv + 0.8));
    }
    auto loss_at = [&](double a, double bb, double c) {
        double s = 0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            const double d = a * std::sin(bb * X[i] + c) - Y[i];
            s += d * d;
        }
        return s;
    };
    auto ad_grads = [&](Parameter& A, Parameter& B, Parameter& C, double g[3]) {
        ADouble total(0.0);
        for (std::size_t i = 0; i < X.size(); ++i) {
            total = total + pow(A.var() * sin(B.var() * X[i] + C.var()) - Y[i], 2.0);
        }
        tape().compute_adjoints(total.id);
        g[0] = A.grad();
        g[1] = B.grad();
        g[2] = C.grad();
    };

    const double h = 1e-6;
    const double fd[3] = {(loss_at(2.0 + h, 1.0, 1.0) - loss_at(2.0 - h, 1.0, 1.0)) / (2 * h),
                          (loss_at(2.0, 1.0 + h, 1.0) - loss_at(2.0, 1.0 - h, 1.0)) / (2 * h),
                          (loss_at(2.0, 1.0, 1.0 + h) - loss_at(2.0, 1.0, 1.0 - h)) / (2 * h)};

    Parameter A(2.0), B(1.0), C(1.0);
    double g[3];
    tape().reset();
    ad_grads(A, B, C, g);          // epoch 1, result discarded
    tape().reset();                // "validation" pass with no update() in between
    for (std::size_t i = 0; i < X.size(); ++i) (void)(A.var() * sin(B.var() * X[i] + C.var()));
    tape().reset();                // epoch 2
    ad_grads(A, B, C, g);
    check_close(g[0], fd[0], 1e-5, "dLoss/dA after an inference pass");
    check_close(g[1], fd[1], 1e-5, "dLoss/dB after an inference pass");
    check_close(g[2], fd[2], 1e-5, "dLoss/dC after an inference pass");
}

// --- Bug 2: value semantics; no id=-1 sentinel ------------------------------
static void test_bug2_move_semantics() {
    std::puts("[bug 2] moved-from ADouble/Parameter stay usable");
    tape().reset();
    ADouble a(2.0), b(3.0);
    ADouble c = std::move(a);
    report(a.id == c.id, "moved-from ADouble keeps its tape id");
    ADouble d = a + b;  // would have recorded parent_id = -1 before the fix
    tape().compute_adjoints(d.id);
    check_close(d.val, 5.0, 1e-12, "arithmetic on a moved-from ADouble");
    check_close(b.grad(), 1.0, 1e-12, "gradient through a moved-from ADouble");

    // A Parameter surviving a vector reallocation (which moves it) keeps working.
    tape().reset();
    std::vector<Parameter> ps;
    for (int i = 0; i < 64; ++i) ps.push_back(Parameter{static_cast<double>(i + 1)});
    ADouble sum(0.0);
    for (auto& pp : ps) sum = sum + pp.var() * 2.0;
    tape().compute_adjoints(sum.id);
    check_close(ps[0].grad(), 2.0, 1e-12, "gradient after vector reallocation moves Parameters");
    check_close(ps[63].grad(), 2.0, 1e-12, "gradient of the last reallocated Parameter");

    // An invalid id must raise rather than corrupt the heap.
    check_throws([] { tape().compute_adjoints(-1); }, "compute_adjoints(-1) throws");
    check_throws([] { tape().gradient(1 << 30); }, "gradient(out of range) throws");
}

// --- Bug 3: destructive solvers must not alias their inputs -----------------
template <typename T>
static Vector<T> solve_gaussian(Matrix<T> A, Vector<T> b) {  // by value, as in the example
    const std::size_t n = A.getRows();
    for (std::size_t i = 0; i < n; i++) {
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

static void test_bug3_solver_does_not_destroy_input() {
    std::puts("[bug 3] repeated solves on the same matrix stay correct");
    const std::size_t n = 4;
    Matrix<double> A(n, n);
    const double raw[4][4] = {{10, 1, 2, 1}, {2, 12, 1, 3}, {1, 2, 9, 1}, {3, 1, 2, 11}};
    for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) A[i][j] = raw[i][j];

    for (int k = 0; k < 5; ++k) {
        Vector<double> b(n);
        for (std::size_t i = 0; i < n; i++) b[i] = 1.0 + i + k;
        Vector<double> x = solve_gaussian(A, b);
        double worst = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            double s = 0;
            for (std::size_t j = 0; j < n; j++) s += raw[i][j] * x[j];
            worst = std::fmax(worst, std::fabs(s - b[i]));
        }
        check_close(worst, 0.0, 1e-9, "solve #" + std::to_string(k) + " residual");
    }
    report(A[1][0] == 2.0, "input matrix is unmodified after 5 solves");
}

// --- Bug 5 / 6 / 12: input validation ---------------------------------------
static void test_input_validation() {
    std::puts("[bugs 5, 6, 12] input validation");
    tape().reset();
    check_throws(
        [] {
            std::vector<ADouble> preds{ADouble(1.0), ADouble(2.0)};
            std::vector<double> targets{0.0};
            (void)mse_loss(preds, targets);
        },
        "mse_loss with mismatched sizes throws");

    check_throws(
        [] {
            Matrix<double> M(2, 3);
            Vector<double> v(2);
            (void)(M * v);
        },
        "Matrix*Vector with mismatched sizes throws");

    check_throws(
        [] {
            Matrix<double> A(2, 3), B(2, 2);
            (void)(A * B);
        },
        "Matrix*Matrix with mismatched inner dimension throws");

    check_throws([] { NeuralNetwork nn({4}); }, "NeuralNetwork with a 1-entry topology throws");
    check_throws([] { NeuralNetwork nn({2, 0, 1}); }, "NeuralNetwork with a zero layer throws");
    check_throws(
        [] {
            NeuralNetwork nn({2, 3, 1}, 1u);
            std::vector<double> x{1.0, 2.0, 3.0};
            (void)nn.forward(x);
        },
        "forward() with the wrong input width throws");
#if !defined(NDEBUG)
    check_throws(
        [] {
            Vector<double> v(3);
            (void)v[5];
        },
        "Vector index out of range throws");
#endif
}

// --- Bug 8: default-constructed containers must not alias tape nodes --------
static void test_bug8_no_shared_tape_nodes() {
    std::puts("[bug 8] Matrix/Vector cells get independent tape nodes");
    tape().reset();
    Matrix<ADouble> M(2, 2);
    Vector<ADouble> v(3);
    const bool matrix_distinct = M[0][0].id != M[0][1].id && M[0][1].id != M[1][0].id &&
                                 M[1][0].id != M[1][1].id;
    const bool vector_distinct = v[0].id != v[1].id && v[1].id != v[2].id;
    report(matrix_distinct, "Matrix cells have distinct tape ids");
    report(vector_distinct, "Vector cells have distinct tape ids");

    // Accumulating into default-constructed cells gives the right gradients.
    tape().reset();
    Matrix<ADouble> A(2, 2), B(2, 2);
    const double av[2][2] = {{1, 2}, {3, 4}};
    const double bv[2][2] = {{5, 6}, {7, 8}};
    for (std::size_t i = 0; i < 2; i++)
        for (std::size_t j = 0; j < 2; j++) {
            A[i][j] = av[i][j];
            B[i][j] = bv[i][j];
        }
    Matrix<ADouble> C = A * B;
    tape().compute_adjoints(C[0][0].id);
    check_close(C[0][0].val, 19.0, 1e-12, "C00 value");
    check_close(B[0][0].grad(), 1.0, 1e-12, "dC00/dB00");
    check_close(B[1][0].grad(), 2.0, 1e-12, "dC00/dB10");
    check_close(B[0][1].grad(), 0.0, 1e-12, "dC00/dB01");
    check_close(A[1][1].grad(), 0.0, 1e-12, "dC00/dA11");
}

// --- Bug 9 / 10: domain handling --------------------------------------------
static void test_domain_handling() {
    std::puts("[bugs 9, 10] domain handling");
    tape().reset();
    ADouble base(-2.0), e(3.0);
    ADouble y = pow(base, e);
    tape().compute_adjoints(y.id);
    check_close(y.val, -8.0, 1e-12, "pow(-2,3) value");
    check_close(base.grad(), 12.0, 1e-12, "d pow(-2,3)/d base");
    report(!std::isnan(e.grad()), "d pow(-2,3)/d exponent is not NaN");

    check_throws(
        [] {
            ADouble a(1.0), b(0.0);
            (void)(a / b);
        },
        "division by zero throws");
    check_throws([] { (void)log(ADouble(-1.0)); }, "log of a negative value throws");
    check_throws([] { (void)log(ADouble(0.0)); }, "log(0) throws");
    check_throws([] { (void)sqrt(ADouble(-1.0)); }, "sqrt of a negative value throws");
}

// --- Bug 13: mse_loss is a mean ---------------------------------------------
static void test_bug13_mse_is_mean() {
    std::puts("[bug 13] mse_loss returns a mean, sse_loss a sum");
    tape().reset();
    std::vector<ADouble> preds{ADouble(1.0), ADouble(3.0)};
    std::vector<double> targets{0.0, 0.0};
    ADouble mse = mse_loss(preds, targets);
    check_close(mse.val, 5.0, 1e-12, "mse_loss value (1+9)/2");
    tape().reset();
    std::vector<ADouble> preds2{ADouble(1.0), ADouble(3.0)};
    ADouble sse = sse_loss(preds2, targets);
    check_close(sse.val, 10.0, 1e-12, "sse_loss value 1+9");
}

// --- Bug 17: seeded networks are reproducible -------------------------------
static void test_bug17_reproducible() {
    std::puts("[bug 17] seeded NeuralNetwork is reproducible");
    tape().reset();
    NeuralNetwork a({2, 3, 1}, 12345u);
    NeuralNetwork b({2, 3, 1}, 12345u);
    NeuralNetwork c({2, 3, 1}, 54321u);
    const double wa = a.layers[0].neurons[0].weights[0].val;
    const double wb = b.layers[0].neurons[0].weights[0].val;
    const double wc = c.layers[0].neurons[0].weights[0].val;
    report(wa == wb, "same seed gives identical weights");
    report(wa != wc, "different seeds give different weights");
}

// --- End-to-end: the network actually learns --------------------------------
static void test_training_converges() {
    std::puts("[end to end] AND network converges");
    NeuralNetwork nn({2, 5, 5, 1}, 20240513u);
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> Y = {{0}, {0}, {0}, {1}};
    double final_loss = 0.0;
    for (int epoch = 0; epoch < 3000; ++epoch) {
        tape().reset();
        ADouble total(0.0);
        for (std::size_t i = 0; i < X.size(); ++i) {
            total = total + mse_loss(nn.forward(X[i]), Y[i]);
        }
        total = total / static_cast<double>(X.size());
        tape().compute_adjoints(total.id);
        nn.update_adam(0.01);
        final_loss = total.val;

        // An inference pass every epoch: this is the pattern that used to
        // silently corrupt the gradients (bug 1).
        tape().reset();
        (void)nn.forward(X[0]);
    }
    report(final_loss < 1e-4, "AND loss below 1e-4 after 3000 epochs",
           "final loss " + std::to_string(final_loss));
}

int main() {
    test_elementary();
    test_compound_assign();
    test_bug1_reset_invalidates_parameters();
    test_bug2_move_semantics();
    test_bug3_solver_does_not_destroy_input();
    test_input_validation();
    test_bug8_no_shared_tape_nodes();
    test_domain_handling();
    test_bug13_mse_is_mean();
    test_bug17_reproducible();
    test_training_converges();

    tape().release();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
