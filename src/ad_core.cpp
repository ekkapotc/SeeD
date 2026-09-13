#include "ad_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

// ============================================================================
// Error helpers
// ============================================================================
namespace seed_detail {

void throw_out_of_range(const char* what, long long index, long long size) {
    throw std::out_of_range(std::string(what) + " " + std::to_string(index) +
                            " is out of range [0, " + std::to_string(size) + ")");
}

void throw_domain(const char* what) { throw std::domain_error(std::string("SeeD: ") + what); }

}  // namespace seed_detail

// ============================================================================
// AdjointTape
// ============================================================================
AdjointTape& tape() {
    static AdjointTape instance;
    return instance;
}

void AdjointTape::reserve(std::size_t node_capacity, std::size_t var_capacity) {
    nodes.reserve(node_capacity);
    adjoints.reserve(var_capacity);
}

int AdjointTape::register_variable() {
    if (var_count == std::numeric_limits<int>::max()) {
        throw std::length_error(
            "SeeD: adjoint tape exhausted (2^31-1 variables). Call tape().reset() "
            "between independent forward passes.");
    }
    int id = var_count++;
    adjoints.push_back(0.0);
    return id;
}

void AdjointTape::record(int child_id, int parent_id, double partial) {
    // Always on: an invalid id here becomes an out-of-bounds write during the
    // reverse sweep, which is silent heap corruption rather than a clean error.
    if (child_id < 0 || child_id >= var_count) {
        seed_detail::throw_out_of_range("SeeD: tape child id", child_id, var_count);
    }
    if (parent_id < 0 || parent_id >= var_count) {
        seed_detail::throw_out_of_range("SeeD: tape parent id", parent_id, var_count);
    }
    nodes.push_back({child_id, parent_id, partial});
}

void AdjointTape::compute_adjoints(int output_id) {
    if (output_id < 0 || static_cast<std::size_t>(output_id) >= adjoints.size()) {
        seed_detail::throw_out_of_range("SeeD: output id", output_id,
                                        static_cast<long long>(adjoints.size()));
    }
    std::fill(adjoints.begin(), adjoints.end(), 0.0);
    adjoints[static_cast<std::size_t>(output_id)] = 1.0;
    // Reverse sweep over std::size_t so the loop is correct past INT_MAX nodes.
    for (std::size_t i = nodes.size(); i-- > 0;) {
        const TapeNode& node = nodes[i];
        adjoints[static_cast<std::size_t>(node.parent_id)] +=
            adjoints[static_cast<std::size_t>(node.child_id)] * node.partial;
    }
}

double AdjointTape::gradient(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= adjoints.size()) {
        seed_detail::throw_out_of_range("SeeD: adjoint id", id,
                                        static_cast<long long>(adjoints.size()));
    }
    return adjoints[static_cast<std::size_t>(id)];
}

void AdjointTape::reset() {
    nodes.clear();
    adjoints.clear();
    var_count = 0;
    ++generation;  // every id handed out before now is stale from here on
}

void AdjointTape::release() {
    reset();
    nodes.shrink_to_fit();
    adjoints.shrink_to_fit();
}

// ============================================================================
// ADouble
// ============================================================================
ADouble::ADouble(double v) : val(v), id(tape().register_variable()) {}

ADouble::ADouble(double v, int existing_id) : val(v), id(existing_id) {}

double ADouble::grad() const { return tape().gradient(id); }

ADouble& ADouble::operator+=(const ADouble& other) {
    *this = *this + other;
    return *this;
}

ADouble& ADouble::operator-=(const ADouble& other) {
    *this = *this - other;
    return *this;
}

ADouble& ADouble::operator*=(const ADouble& other) {
    *this = *this * other;
    return *this;
}

ADouble& ADouble::operator/=(const ADouble& other) {
    *this = *this / other;
    return *this;
}

// ============================================================================
// Parameter
// ============================================================================
Parameter::Parameter(double initial_value)
    : val(initial_value), current_tape_id(-1), tape_generation(0), m(0.0), v(0.0), t(0) {}

ADouble Parameter::var() {
    AdjointTape& tp = tape();
    if (current_tape_id == -1 || tape_generation != tp.generation) {
        // First use in this tape generation: register a fresh leaf.
        ADouble node(val);
        current_tape_id = node.id;
        tape_generation = tp.generation;
        return node;
    }
    // Later uses in the same generation attach to the same leaf so that batch
    // gradients accumulate onto one node.
    return ADouble(val, current_tape_id);
}

bool Parameter::is_bound() const {
    return current_tape_id != -1 && tape_generation == tape().generation;
}

double Parameter::grad() const { return is_bound() ? tape().gradient(current_tape_id) : 0.0; }

void Parameter::update(double lr) {
    if (!is_bound()) {
        current_tape_id = -1;  // drop any id left over from a previous generation
        return;
    }
    val -= lr * tape().gradient(current_tape_id);
    // `val` has changed, so the tape node still holds the old value: unbind.
    current_tape_id = -1;
}

void Parameter::update_adam(double lr, double beta1, double beta2, double epsilon) {
    if (!is_bound()) {
        current_tape_id = -1;
        return;
    }
    t += 1;
    const double grad_val = tape().gradient(current_tape_id);

    m = beta1 * m + (1.0 - beta1) * grad_val;
    v = beta2 * v + (1.0 - beta2) * (grad_val * grad_val);

    const double m_hat = m / (1.0 - std::pow(beta1, t));
    const double v_hat = v / (1.0 - std::pow(beta2, t));

    val -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

    current_tape_id = -1;
}

void Parameter::zero_state() {
    m = 0.0;
    v = 0.0;
    t = 0;
}

// ============================================================================
// Arithmetic
// ============================================================================
ADouble operator+(const ADouble& lhs, const ADouble& rhs) {
    AdjointTape& tp = tape();
    ADouble res(lhs.val + rhs.val);
    tp.record(res.id, lhs.id, 1.0);
    tp.record(res.id, rhs.id, 1.0);
    return res;
}

ADouble operator+(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val + rhs);
    tape().record(res.id, lhs.id, 1.0);
    return res;
}

ADouble operator+(double lhs, const ADouble& rhs) {
    ADouble res(lhs + rhs.val);
    tape().record(res.id, rhs.id, 1.0);
    return res;
}

ADouble operator-(const ADouble& lhs, const ADouble& rhs) {
    AdjointTape& tp = tape();
    ADouble res(lhs.val - rhs.val);
    tp.record(res.id, lhs.id, 1.0);
    tp.record(res.id, rhs.id, -1.0);
    return res;
}

ADouble operator-(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val - rhs);
    tape().record(res.id, lhs.id, 1.0);
    return res;
}

ADouble operator-(double lhs, const ADouble& rhs) {
    ADouble res(lhs - rhs.val);
    tape().record(res.id, rhs.id, -1.0);
    return res;
}

ADouble operator-(const ADouble& x) {
    ADouble res(-x.val);
    tape().record(res.id, x.id, -1.0);
    return res;
}

ADouble operator*(const ADouble& lhs, const ADouble& rhs) {
    AdjointTape& tp = tape();
    ADouble res(lhs.val * rhs.val);
    tp.record(res.id, lhs.id, rhs.val);
    tp.record(res.id, rhs.id, lhs.val);
    return res;
}

ADouble operator*(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val * rhs);
    tape().record(res.id, lhs.id, rhs);
    return res;
}

ADouble operator*(double lhs, const ADouble& rhs) {
    ADouble res(lhs * rhs.val);
    tape().record(res.id, rhs.id, lhs);
    return res;
}

ADouble operator/(const ADouble& lhs, const ADouble& rhs) {
    AdjointTape& tp = tape();
    if (rhs.val == 0.0) seed_detail::throw_domain("division by zero");
    ADouble res(lhs.val / rhs.val);
    tp.record(res.id, lhs.id, 1.0 / rhs.val);
    tp.record(res.id, rhs.id, -lhs.val / (rhs.val * rhs.val));
    return res;
}

ADouble operator/(const ADouble& lhs, double rhs) {
    if (rhs == 0.0) seed_detail::throw_domain("division by zero");
    ADouble res(lhs.val / rhs);
    tape().record(res.id, lhs.id, 1.0 / rhs);
    return res;
}

ADouble operator/(double lhs, const ADouble& rhs) {
    if (rhs.val == 0.0) seed_detail::throw_domain("division by zero");
    ADouble res(lhs / rhs.val);
    tape().record(res.id, rhs.id, -lhs / (rhs.val * rhs.val));
    return res;
}

// ============================================================================
// Comparisons & streaming
// ============================================================================
bool operator==(const ADouble& a, const ADouble& b) { return a.val == b.val; }
bool operator!=(const ADouble& a, const ADouble& b) { return a.val != b.val; }
bool operator<(const ADouble& a, const ADouble& b) { return a.val < b.val; }
bool operator<=(const ADouble& a, const ADouble& b) { return a.val <= b.val; }
bool operator>(const ADouble& a, const ADouble& b) { return a.val > b.val; }
bool operator>=(const ADouble& a, const ADouble& b) { return a.val >= b.val; }
bool operator==(const ADouble& a, double b) { return a.val == b; }
bool operator!=(const ADouble& a, double b) { return a.val != b; }
bool operator<(const ADouble& a, double b) { return a.val < b; }
bool operator<=(const ADouble& a, double b) { return a.val <= b; }
bool operator>(const ADouble& a, double b) { return a.val > b; }
bool operator>=(const ADouble& a, double b) { return a.val >= b; }
bool operator==(double a, const ADouble& b) { return a == b.val; }
bool operator!=(double a, const ADouble& b) { return a != b.val; }
bool operator<(double a, const ADouble& b) { return a < b.val; }
bool operator<=(double a, const ADouble& b) { return a <= b.val; }
bool operator>(double a, const ADouble& b) { return a > b.val; }
bool operator>=(double a, const ADouble& b) { return a >= b.val; }

std::ostream& operator<<(std::ostream& os, const ADouble& x) { return os << x.val; }

// ============================================================================
// Transcendental functions
// ============================================================================
ADouble sin(const ADouble& x) {
    ADouble res(std::sin(x.val));
    tape().record(res.id, x.id, std::cos(x.val));
    return res;
}

ADouble cos(const ADouble& x) {
    ADouble res(std::cos(x.val));
    tape().record(res.id, x.id, -std::sin(x.val));
    return res;
}

ADouble tan(const ADouble& x) {
    const double c = std::cos(x.val);
    if (c == 0.0) seed_detail::throw_domain("tan() at an odd multiple of pi/2");
    ADouble res(std::tan(x.val));
    tape().record(res.id, x.id, 1.0 / (c * c));
    return res;
}

ADouble exp(const ADouble& x) {
    ADouble res(std::exp(x.val));
    tape().record(res.id, x.id, res.val);
    return res;
}

ADouble log(const ADouble& x) {
    if (x.val <= 0.0) seed_detail::throw_domain("log() of a non-positive value");
    ADouble res(std::log(x.val));
    tape().record(res.id, x.id, 1.0 / x.val);
    return res;
}

ADouble sqrt(const ADouble& x) {
    if (x.val < 0.0) seed_detail::throw_domain("sqrt() of a negative value");
    if (x.val == 0.0) seed_detail::throw_domain("sqrt() is not differentiable at 0");
    ADouble res(std::sqrt(x.val));
    tape().record(res.id, x.id, 0.5 / res.val);
    return res;
}

ADouble abs(const ADouble& x) {
    ADouble res(std::fabs(x.val));
    // Sub-gradient 0 at the kink, matching the relu() convention.
    tape().record(res.id, x.id, (x.val > 0.0) ? 1.0 : (x.val < 0.0 ? -1.0 : 0.0));
    return res;
}

ADouble fabs(const ADouble& x) { return abs(x); }

// ============================================================================
// Powers
// ============================================================================
ADouble pow(const ADouble& base, const ADouble& exponent) {
    if (base.val == 0.0 && exponent.val < 1.0) {
        seed_detail::throw_domain("pow() derivative is unbounded at base 0 for exponent < 1");
    }
    AdjointTape& tp = tape();
    ADouble res(std::pow(base.val, exponent.val));

    const double d_base = exponent.val * std::pow(base.val, exponent.val - 1.0);
    tp.record(res.id, base.id, d_base);

    // d/de of b^e is b^e * ln(b), which only exists for b > 0. For b <= 0 the
    // expression is only defined for integral exponents, where it is constant
    // in e; recording 0 keeps the rest of the graph clean instead of flooding
    // every ancestor of the exponent with NaN.
    const double d_exponent = (base.val > 0.0) ? res.val * std::log(base.val) : 0.0;
    tp.record(res.id, exponent.id, d_exponent);
    return res;
}

ADouble pow(const ADouble& base, double exponent) {
    if (base.val == 0.0 && exponent < 1.0) {
        seed_detail::throw_domain("pow() derivative is unbounded at base 0 for exponent < 1");
    }
    ADouble res(std::pow(base.val, exponent));
    tape().record(res.id, base.id, exponent * std::pow(base.val, exponent - 1.0));
    return res;
}

// ============================================================================
// Activations
// ============================================================================
ADouble tanh(const ADouble& x) {
    ADouble res(std::tanh(x.val));
    tape().record(res.id, x.id, 1.0 - (res.val * res.val));
    return res;
}

ADouble relu(const ADouble& x) {
    ADouble res(std::max(0.0, x.val));
    tape().record(res.id, x.id, (x.val > 0.0) ? 1.0 : 0.0);
    return res;
}

ADouble sigmoid(const ADouble& x) {
    const double s = 1.0 / (1.0 + std::exp(-x.val));
    ADouble res(s);
    tape().record(res.id, x.id, s * (1.0 - s));
    return res;
}

// ============================================================================
// Selection
// ============================================================================
ADouble fmax(const ADouble& a, const ADouble& b) {
    AdjointTape& tp = tape();
    const bool take_a = a.val >= b.val;
    ADouble res(take_a ? a.val : b.val);
    tp.record(res.id, a.id, take_a ? 1.0 : 0.0);
    tp.record(res.id, b.id, take_a ? 0.0 : 1.0);
    return res;
}

ADouble fmin(const ADouble& a, const ADouble& b) {
    AdjointTape& tp = tape();
    const bool take_a = a.val <= b.val;
    ADouble res(take_a ? a.val : b.val);
    tp.record(res.id, a.id, take_a ? 1.0 : 0.0);
    tp.record(res.id, b.id, take_a ? 0.0 : 1.0);
    return res;
}
