#include <cmath>
#include <algorithm>
#include "ad_core.hpp"

// Actual instantiation of the global tape
AdjointTape global_tape;

AdjointTape::AdjointTape(size_t node_capacity , size_t var_capacity){
    nodes.reserve(node_capacity);
    adjoints.reserve(var_capacity);
}

int AdjointTape::register_variable() {
    int id = var_count++;
    adjoints.push_back(0.0);
    return id;
}

void AdjointTape::record(int child_id, int parent_id, double partial) {
    nodes.push_back({child_id, parent_id, partial});
}

void AdjointTape::compute_adjoints(int output_id) {
    std::fill(adjoints.begin(), adjoints.end(), 0.0);
    adjoints[output_id] = 1.0;
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
        const auto& node = nodes[i];
        adjoints[node.parent_id] += adjoints[node.child_id] * node.partial;
    }
}

void AdjointTape::reset() {
    nodes.clear();
    adjoints.clear();
    var_count = 0;
}

// Registers a brand-new node on the tape
ADouble::ADouble(double v) : val(v), id(global_tape.register_variable()) {}

// Reuses an existing tape ID (for batch gradient accumulation)
ADouble::ADouble(double v, int existing_id) : val(v), id(existing_id) {}

// Copy Constructor (Aliasing)
ADouble::ADouble(const ADouble& other) {
    val = other.val;
    id = other.id;
}

// Assignment Operator (Aliasing)
ADouble& ADouble::operator=(const ADouble& other) {
    if (this != &other) { // Protect against self-assignment
        val = other.val;
        id = other.id;
    }
    return *this;
}

// Move Constructor
ADouble::ADouble(ADouble&& other) noexcept {
    val = other.val;
    id = other.id;
    other.id = -1; // Invalidate source to prevent accidental use
}

// Move Assignment Operator
ADouble& ADouble::operator=(ADouble&& other) noexcept {
    if (this != &other) {
        val = other.val;
        id = other.id;
        other.id = -1; // Invalidate source
    }
    return *this;
}

Parameter::Parameter(double v) : val(v), current_tape_id(-1), m(0.0), v(0.0), t(0) {}

// Copy Constructor
Parameter::Parameter(const Parameter& other) {
    val = other.val;
    current_tape_id = other.current_tape_id;
    m = other.m;
    v = other.v;
    t = other.t;
}
    
// Assignment Operator
Parameter& Parameter::operator=(const Parameter& other) {
    if (this != &other) {
        val = other.val;
        current_tape_id = other.current_tape_id;
        m = other.m;
        v = other.v;
        t = other.t;
    }
    return *this;
}

// Move Constructor
Parameter::Parameter(Parameter&& other) noexcept {
    val = other.val;
    current_tape_id = other.current_tape_id;
    m = other.m;
    v = other.v;
    t = other.t;
    
    other.current_tape_id = -1; // Invalidate source
    other.m = 0.0;
    other.v = 0.0;
    other.t = 0;
}

// Explicit Move Assignment
Parameter& Parameter::operator=(Parameter&& other) noexcept {
    if (this != &other) {
        val = other.val;
        current_tape_id = other.current_tape_id;
        m = other.m;
        v = other.v;
        t = other.t;
        
        other.current_tape_id = -1; // Invalidate source
        other.m = 0.0;
        other.v = 0.0;
        other.t = 0;
    }
    return *this;
}

ADouble Parameter::var() {
    if (current_tape_id == -1) {
        // First forward pass of the epoch: register a new node on the tape
        ADouble node(val);
        current_tape_id = node.id;
        return node;
    } else {
        // Subsequent passes in the same epoch: attach to the existing node ID
        return ADouble(val, current_tape_id);
    }
}

void Parameter::update(double lr) {
    if (current_tape_id != -1) {
        val -= lr * global_tape.adjoints[current_tape_id];
        // Reset tape ID tracker for the next epoch
        current_tape_id = -1; 
    }
}

void Parameter::update_adam(double lr, double beta1, double beta2, double epsilon) {
    if (current_tape_id != -1) {
        t += 1; // Increment time step
        double grad = global_tape.adjoints[current_tape_id];
        
        // Update biased first moment estimate
        m = beta1 * m + (1.0 - beta1) * grad;
        // Update biased second raw moment estimate
        v = beta2 * v + (1.0 - beta2) * (grad * grad);
        
        // Compute bias-corrected first moment estimate
        double m_hat = m / (1.0 - std::pow(beta1, t));
        // Compute bias-corrected second raw moment estimate
        double v_hat = v / (1.0 - std::pow(beta2, t));
        
        // Apply update
        val -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        
        // Reset tape ID tracker for the next epoch
        current_tape_id = -1; 
    }
}

ADouble operator+(const ADouble& lhs, const ADouble& rhs) {
    ADouble res(lhs.val + rhs.val);
    global_tape.record(res.id, lhs.id, 1.0);
    global_tape.record(res.id, rhs.id, 1.0);
    return res;
}

ADouble operator+(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val + rhs);
    global_tape.record(res.id, lhs.id, 1.0);
    return res;
}

ADouble operator+(double lhs, const ADouble& rhs) {
    ADouble res(lhs + rhs.val);
    global_tape.record(res.id, rhs.id, 1.0);
    return res;
}

ADouble operator-(const ADouble& lhs, const ADouble& rhs) {
    ADouble res(lhs.val - rhs.val);
    global_tape.record(res.id, lhs.id, 1.0);
    global_tape.record(res.id, rhs.id, -1.0);
    return res;
}

ADouble operator-(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val - rhs);
    global_tape.record(res.id, lhs.id, 1.0);
    return res;
}

ADouble operator-(double lhs, const ADouble& rhs) {
    ADouble res(lhs - rhs.val);
    global_tape.record(res.id, rhs.id, -1.0);
    return res;
}

ADouble operator*(const ADouble& lhs, const ADouble& rhs) {
    ADouble res(lhs.val * rhs.val);
    global_tape.record(res.id, lhs.id, rhs.val);
    global_tape.record(res.id, rhs.id, lhs.val);
    return res;
}

ADouble operator*(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val * rhs);
    global_tape.record(res.id, lhs.id, rhs);
    return res;
}

ADouble operator*(double lhs, const ADouble& rhs) {
    ADouble res(lhs * rhs.val);
    global_tape.record(res.id, rhs.id, lhs);
    return res;
}

ADouble operator/(const ADouble& lhs, const ADouble& rhs) {
    ADouble res(lhs.val / rhs.val);
    global_tape.record(res.id, lhs.id, 1.0 / rhs.val);
    global_tape.record(res.id, rhs.id, -lhs.val / (rhs.val * rhs.val));
    return res;
}

ADouble operator/(const ADouble& lhs, double rhs) {
    ADouble res(lhs.val / rhs);
    global_tape.record(res.id, lhs.id, 1.0 / rhs);
    return res;
}

ADouble operator/(double lhs, const ADouble& rhs) {
    ADouble res(lhs / rhs.val);
    global_tape.record(res.id, rhs.id, -lhs / (rhs.val * rhs.val));
    return res;
}

ADouble sin(const ADouble& x) {
    ADouble res(std::sin(x.val));
    global_tape.record(res.id, x.id, std::cos(x.val));
    return res;
}

ADouble cos(const ADouble& x) {
    ADouble res(std::cos(x.val));
    global_tape.record(res.id, x.id, -std::sin(x.val));
    return res;
}

ADouble tan(const ADouble& x) {
    ADouble res(std::tan(x.val));
    double c = std::cos(x.val);
    global_tape.record(res.id, x.id, 1.0 / (c * c));
    return res;
}

ADouble exp(const ADouble& x) {
    ADouble res(std::exp(x.val));
    global_tape.record(res.id, x.id, res.val); 
    return res;
}

ADouble log(const ADouble& x) {
    ADouble res(std::log(x.val));
    global_tape.record(res.id, x.id, 1.0 / x.val);
    return res;
}

ADouble pow(const ADouble& base, const ADouble& exponent) {
    ADouble res(std::pow(base.val, exponent.val));
    double d_base = exponent.val * std::pow(base.val, exponent.val - 1.0);
    global_tape.record(res.id, base.id, d_base);
    double d_exponent = res.val * std::log(base.val);
    global_tape.record(res.id, exponent.id, d_exponent);
    return res;
}

ADouble pow(const ADouble& base, double exponent) {
    ADouble res(std::pow(base.val, exponent));
    global_tape.record(res.id, base.id, exponent * std::pow(base.val, exponent - 1.0));
    return res;
}

ADouble tanh(const ADouble& x) {
    ADouble res(std::tanh(x.val));
    global_tape.record(res.id, x.id, 1.0 - (res.val * res.val));
    return res;
}

ADouble relu(const ADouble& x) {
    ADouble res(std::max(0.0, x.val));
    global_tape.record(res.id, x.id, (x.val > 0.0) ? 1.0 : 0.0);
    return res;
}

ADouble sigmoid(const ADouble& x) {
    double s = 1.0 / (1.0 + std::exp(-x.val));
    ADouble res(s);
    global_tape.record(res.id, x.id, s * (1.0 - s));
    return res;
}
