#pragma once
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// 0. Internal checking helpers
// ============================================================================
// Bounds/domain checks are ON by default. Define NDEBUG (see `make release`)
// to compile out the container index checks on hot paths. Tape-integrity and
// domain checks are always on: they guard against heap corruption and silent
// NaN propagation, and cost a single predictable branch next to a push_back.
namespace seed_detail {
[[noreturn]] void throw_out_of_range(const char* what, long long index, long long size);
[[noreturn]] void throw_domain(const char* what);
}  // namespace seed_detail

#if defined(NDEBUG)
#define SEED_CHECK_INDEX(i, n, what) ((void)0)
#else
#define SEED_CHECK_INDEX(i, n, what)                                              \
    do {                                                                          \
        if (static_cast<std::size_t>(i) >= static_cast<std::size_t>(n))           \
            ::seed_detail::throw_out_of_range((what), static_cast<long long>(i),  \
                                              static_cast<long long>(n));         \
    } while (0)
#endif

// ============================================================================
// 1. Adjoint tape
// ============================================================================
struct TapeNode {
    int child_id;
    int parent_id;
    double partial;
};

struct AdjointTape {
    std::vector<TapeNode> nodes;
    std::vector<double> adjoints;
    int var_count = 0;

    // Bumped by every reset(). A Parameter records the generation it bound to,
    // so a tape id from a previous generation can never be mistaken for a live
    // one. Starts at 1 so that 0 can mean "never bound".
    unsigned long long generation = 1;

    AdjointTape() = default;

    // Optional pre-allocation. Nothing is reserved by default -- an AD program
    // that never runs a big tape should not pay for a 24 MB reservation.
    void reserve(std::size_t node_capacity, std::size_t var_capacity);

    int register_variable();
    void record(int child_id, int parent_id, double partial);
    void compute_adjoints(int output_id);

    // Bounds-checked adjoint lookup. Prefer this over indexing `adjoints`.
    double gradient(int id) const;

    // Invalidates every tape id handed out so far and bumps `generation`.
    void reset();
    // reset() plus returning the buffers to the allocator.
    void release();
};

// The one tape every ADouble records onto.
//
// This is a function, not a global object, on purpose: a namespace-scope
// AdjointTape would have an unspecified construction order relative to any
// namespace-scope ADouble in another translation unit, which produced tape ids
// pointing at slots that did not exist (a link-order-dependent bug). A
// function-local static is constructed on first use, so the ordering is always
// correct.
AdjointTape& tape();

// ============================================================================
// 2. ADouble
// ============================================================================
class ADouble {
public:
    double val;
    int id;

    ADouble(double v = 0.0);  // NOLINT(google-explicit-constructor) -- by design

    // ADouble is a non-owning value type: it is a (value, tape id) pair and owns
    // no resources. All five special members are therefore the compiler's.
    // (An earlier hand-written move set `id = -1` on the source; because nothing
    // validated that sentinel, using a moved-from ADouble indexed adjoints[-1].)
    ADouble(const ADouble&) = default;
    ADouble(ADouble&&) noexcept = default;
    ADouble& operator=(const ADouble&) = default;
    ADouble& operator=(ADouble&&) noexcept = default;
    ~ADouble() = default;

    // Adjoint of this node after the most recent compute_adjoints().
    double grad() const;

    ADouble& operator+=(const ADouble& other);
    ADouble& operator-=(const ADouble& other);
    ADouble& operator*=(const ADouble& other);
    ADouble& operator/=(const ADouble& other);

private:
    // Attach to an existing tape node instead of registering a new one.
    // Only Parameter may do this; an arbitrary caller forging a tape id is how
    // the tape invariant gets broken.
    ADouble(double v, int existing_id);
    friend struct Parameter;
};

// ============================================================================
// 3. Parameter (trainable leaf + optimiser state)
// ============================================================================
struct Parameter {
    double val;
    int current_tape_id;
    unsigned long long tape_generation;

    // Adam optimiser state
    double m;
    double v;
    int t;

    explicit Parameter(double v = 0.0);

    // Plain value type -- see the note on ADouble's special members.
    Parameter(const Parameter&) = default;
    Parameter& operator=(const Parameter&) = default;
    Parameter(Parameter&&) noexcept = default;
    Parameter& operator=(Parameter&&) noexcept = default;
    ~Parameter() = default;

    // Returns this parameter as a tape node, registering one on first use in
    // the current tape generation and re-using it for the rest of the pass so
    // that batch gradients accumulate onto a single leaf.
    ADouble var();

    // True when this parameter holds a tape id belonging to the *current* tape
    // generation, i.e. when grad()/update() have something meaningful to read.
    bool is_bound() const;

    // Adjoint of this parameter after the most recent compute_adjoints().
    // Returns 0.0 when the parameter did not take part in the current tape.
    double grad() const;

    // Solvers. Both are no-ops when the parameter is not bound to the current
    // tape generation (i.e. it did not participate in the forward pass).
    void update(double lr);
    void update_adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    // Clears the Adam moments and step counter.
    void zero_state();
};

// ============================================================================
// 4. Math operators
// ============================================================================
ADouble operator+(const ADouble& lhs, const ADouble& rhs);
ADouble operator+(const ADouble& lhs, double rhs);
ADouble operator+(double lhs, const ADouble& rhs);

ADouble operator-(const ADouble& lhs, const ADouble& rhs);
ADouble operator-(const ADouble& lhs, double rhs);
ADouble operator-(double lhs, const ADouble& rhs);
ADouble operator-(const ADouble& x);  // unary negation

ADouble operator*(const ADouble& lhs, const ADouble& rhs);
ADouble operator*(const ADouble& lhs, double rhs);
ADouble operator*(double lhs, const ADouble& rhs);

ADouble operator/(const ADouble& lhs, const ADouble& rhs);
ADouble operator/(const ADouble& lhs, double rhs);
ADouble operator/(double lhs, const ADouble& rhs);

// --- Comparisons (compare values; a tape node has no ordering of its own) ---
bool operator==(const ADouble& a, const ADouble& b);
bool operator!=(const ADouble& a, const ADouble& b);
bool operator<(const ADouble& a, const ADouble& b);
bool operator<=(const ADouble& a, const ADouble& b);
bool operator>(const ADouble& a, const ADouble& b);
bool operator>=(const ADouble& a, const ADouble& b);
bool operator==(const ADouble& a, double b);
bool operator!=(const ADouble& a, double b);
bool operator<(const ADouble& a, double b);
bool operator<=(const ADouble& a, double b);
bool operator>(const ADouble& a, double b);
bool operator>=(const ADouble& a, double b);
bool operator==(double a, const ADouble& b);
bool operator!=(double a, const ADouble& b);
bool operator<(double a, const ADouble& b);
bool operator<=(double a, const ADouble& b);
bool operator>(double a, const ADouble& b);
bool operator>=(double a, const ADouble& b);

std::ostream& operator<<(std::ostream& os, const ADouble& x);

// --- Transcendental & math functions ---
ADouble sin(const ADouble& x);
ADouble cos(const ADouble& x);
ADouble tan(const ADouble& x);
ADouble exp(const ADouble& x);
ADouble log(const ADouble& x);  // natural logarithm
ADouble sqrt(const ADouble& x);
ADouble abs(const ADouble& x);
ADouble fabs(const ADouble& x);

// --- Power functions ---
ADouble pow(const ADouble& base, const ADouble& exponent);
ADouble pow(const ADouble& base, double exponent);

// --- Modern ML activations ---
ADouble tanh(const ADouble& x);
ADouble relu(const ADouble& x);
ADouble sigmoid(const ADouble& x);

// --- Selection ---
ADouble fmax(const ADouble& a, const ADouble& b);
ADouble fmin(const ADouble& a, const ADouble& b);

// ============================================================================
// 5. Templated Matrix & Vector classes
// ============================================================================
template <typename T>
class Vector {
private:
    std::vector<T> data;

public:
    Vector() = default;

    // Value-initialises each element *independently*. This matters for AD
    // element types: copy-filling from one prototype would give every element
    // the same tape id.
    explicit Vector(std::size_t n) : data(n) {}

    // Explicit fill. For AD element types every element aliases `val`'s tape
    // node -- that is what "fill with this variable" means; use Vector(n) when
    // you want independent elements.
    Vector(std::size_t n, const T& val) : data(n, val) {}

    std::size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }

    T& operator[](std::size_t i) {
        SEED_CHECK_INDEX(i, data.size(), "Vector index");
        return data[i];
    }
    const T& operator[](std::size_t i) const {
        SEED_CHECK_INDEX(i, data.size(), "Vector index");
        return data[i];
    }
    T& at(std::size_t i) { return data.at(i); }
    const T& at(std::size_t i) const { return data.at(i); }
};

template <typename T>
class Matrix {
private:
    // Declared in initialisation order: rows/cols before data.
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::vector<T>> data;

public:
    Matrix() = default;

    // Value-initialises every cell independently (see Vector above).
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r) {
        for (auto& row : data) row.resize(c);
    }

    // Explicit fill; every cell is a copy of `val` (aliases its tape node).
    Matrix(std::size_t r, std::size_t c, const T& val) : rows(r), cols(c), data(r) {
        for (auto& row : data) row.assign(c, val);
    }

    std::size_t getRows() const { return rows; }
    std::size_t getCols() const { return cols; }

    std::vector<T>& operator[](std::size_t i) {
        SEED_CHECK_INDEX(i, rows, "Matrix row index");
        return data[i];
    }
    const std::vector<T>& operator[](std::size_t i) const {
        SEED_CHECK_INDEX(i, rows, "Matrix row index");
        return data[i];
    }

    Vector<T> operator*(const Vector<T>& vec) const {
        if (vec.size() != cols) {
            throw std::invalid_argument(
                "Matrix*Vector: matrix has " + std::to_string(cols) +
                " columns but vector has " + std::to_string(vec.size()) + " elements");
        }
        Vector<T> result(rows);
        for (std::size_t i = 0; i < rows; i++) {
            T sum = T(0.0);
            for (std::size_t j = 0; j < cols; j++) sum += data[i][j] * vec[j];
            result[i] = sum;
        }
        return result;
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (other.getRows() != cols) {
            throw std::invalid_argument(
                "Matrix*Matrix: left operand has " + std::to_string(cols) +
                " columns but right operand has " + std::to_string(other.getRows()) + " rows");
        }
        Matrix<T> result(rows, other.getCols());
        for (std::size_t i = 0; i < rows; i++) {
            for (std::size_t j = 0; j < other.getCols(); j++) {
                for (std::size_t k = 0; k < cols; k++) result[i][j] += data[i][k] * other[k][j];
            }
        }
        return result;
    }
};
