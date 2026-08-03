#pragma once
#include <vector>

// Forward declarations and core structs
struct TapeNode {
    int child_id;
    int parent_id;
    double partial;
};

struct AdjointTape {
    std::vector<TapeNode> nodes;
    std::vector<double> adjoints;
    int var_count = 0;

    AdjointTape(size_t node_capacity=1000000, size_t var_capacity=1000000);

    int register_variable();
    void record(int child_id, int parent_id, double partial);
    void compute_adjoints(int output_id);
    void reset();
};

// Declare the global tape (defined in the .cpp file)
extern AdjointTape global_tape;

class ADouble {
public:
    double val;
    int id;
    ADouble(double v = 0.0);
    ADouble(double v, int existing_id); // Construct ADouble with existing Tape ID
    
    // Rule of Five Memory Management
    ADouble(const ADouble& other);
    ADouble(ADouble&& other) noexcept;
    ADouble& operator=(const ADouble& other);
    ADouble& operator=(ADouble&& other) noexcept;
    ~ADouble() = default;
};

struct Parameter {
    double val;
    int current_tape_id;

    // Adam Optimizer State
    double m;
    double v;
    int t;

    Parameter(double v = 0.0);
    
    // Rule of Five Memory Management
    Parameter(const Parameter& other);
    Parameter& operator=(const Parameter& other);
    Parameter(Parameter&& other) noexcept;
    Parameter& operator=(Parameter&& other) noexcept;
    
    ADouble var();
    
    // Solvers
    void update(double lr);
    void update_adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    
    ~Parameter() = default;
};

// --- Math Operator Declarations ---
ADouble operator+(const ADouble& lhs, const ADouble& rhs);
ADouble operator+(const ADouble& lhs, double rhs);
ADouble operator+(double lhs, const ADouble& rhs);

ADouble operator-(const ADouble& lhs, const ADouble& rhs);
ADouble operator-(const ADouble& lhs, double rhs);
ADouble operator-(double lhs, const ADouble& rhs);

ADouble operator*(const ADouble& lhs, const ADouble& rhs);
ADouble operator*(const ADouble& lhs, double rhs);
ADouble operator*(double lhs, const ADouble& rhs);

ADouble operator/(const ADouble& lhs, const ADouble& rhs);
ADouble operator/(const ADouble& lhs, double rhs);
ADouble operator/(double lhs, const ADouble& rhs);

// --- Transcendental & Math Functions ---
ADouble sin(const ADouble& x);
ADouble cos(const ADouble& x);
ADouble tan(const ADouble& x);
ADouble exp(const ADouble& x);
ADouble log(const ADouble& x); // Natural logarithm (ln)

// --- Power Functions ---
ADouble pow(const ADouble& base, const ADouble& exponent);
ADouble pow(const ADouble& base, double exponent);

// --- Modern ML Activations ---
ADouble tanh(const ADouble& x);
ADouble relu(const ADouble& x);
ADouble sigmoid(const ADouble& x);
