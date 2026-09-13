# SeeD — Bug Review

Reviewed: `inc/ad_core.hpp`, `inc/neural_net.hpp`, `src/ad_core.cpp`, `src/neural_net.cpp`,
`Makefile`, and all six `examples/*/main.cpp`. Every finding below was reproduced by
building the code (g++ 13.3, `-std=c++17`) and running it under AddressSanitizer +
UndefinedBehaviorSanitizer, with derivatives cross-checked against central finite differences.

The good news first: the elementary derivative rules are all correct. `sin`, `cos`, `tan`,
`exp`, `log`, `tanh`, `sigmoid`, `relu`, the four arithmetic operators, `pow(x, double)`,
and the `Matrix`/`Vector` products all match finite differences to 1e-8. The reverse sweep
ordering in `compute_adjoints` is sound. The `Sensitivity` example reproduces `A⁻¹` exactly.
The bugs are in tape lifetime management, memory safety, and one example.

---

## CRITICAL

### 1. `AdjointTape::reset()` silently corrupts every `Parameter` — wrong gradients, no crash

`src/ad_core.cpp:32-36`, `src/ad_core.cpp:153-163`, `src/ad_core.cpp:165-171`

`reset()` sets `var_count = 0` and clears `adjoints`, invalidating every tape id in
existence. But `Parameter::current_tape_id` is only cleared inside `update()` /
`update_adam()`. Any `reset()` that is **not** followed by an update leaves every parameter
pointing at an id from a dead tape generation.

On the next forward pass `var()` takes the `else` branch, hands back
`ADouble(val, <stale id>)`, and — critically — does **not** call `register_variable()`.
So the tape is now short by one slot per parameter, every subsequent id is shifted, and
parameter ids collide with unrelated temporaries.

Reproduction (same model as the CurveFitting examples):

```
clean run            : dA=-3.580052    dB=0.859784     dC=0.333684
after a val pass     : dA=-1.949149    dB=1.431277     dC=3.434940
finite difference    : dA=-3.580052    dB=0.859784     dC=0.333684
```

No crash, no warning — just wrong gradients. A minimal case is even more striking: for
`y = p*p` at `p = 3`, the second generation reports `dy/dp = 16` instead of `6`, because
the result node is handed the same id `0` as `p`, producing a self-referential tape edge.

This is not hypothetical: **the shipped `AND_SGD` and `AND_ADAM` examples already do it.**
Their evaluation loops (`examples/AND_SGD/main.cpp:47`, `examples/AND_ADAM/main.cpp:44`)
call `global_tape.reset()` then `nn.forward(...)` with no update in between. They get away
with it only because no backward pass follows. Add a validation-loss print inside the
training loop — the single most natural thing a user will do — and training silently
diverges. Worse, if the number of tape slots lost pushes a stale id past
`adjoints.size()`, `compute_adjoints` writes out of bounds.

**Fix:** version-stamp the tape. Add `int generation = 0;` to `AdjointTape`, increment it in
`reset()`, store a matching `tape_generation` in `Parameter`, and have `var()` re-register
whenever the generation differs:

```cpp
ADouble Parameter::var() {
    if (current_tape_id == -1 || tape_generation != global_tape.generation) {
        ADouble node(val);
        current_tape_id  = node.id;
        tape_generation  = global_tape.generation;
        return node;
    }
    return ADouble(val, current_tape_id);
}
```

This also makes `update()` safe to call (or skip) in any order.

### 2. Moved-from `ADouble` carries `id = -1`; using it writes `adjoints[-1]`

`src/ad_core.cpp:60-74`, `src/ad_core.cpp:19-30`

The move constructor and move assignment set `other.id = -1` "to prevent accidental use."
`ADouble` owns nothing, so there is nothing to invalidate — and the poisoned value is never
checked. `record()` stores `-1` without validation and `compute_adjoints` then indexes with
it. Both confirmed under ASan:

```
ADouble a(2.0), b(3.0);
ADouble c = std::move(a);
ADouble d = a + b;                  // records parent_id = -1
global_tape.compute_adjoints(d.id); // heap-buffer-overflow READ, ad_core.cpp:28
```

```
ADouble y = std::move(x);
global_tape.compute_adjoints(x.id); // heap-buffer-overflow WRITE, ad_core.cpp:25
```

`Parameter`'s move operations have the same problem (`current_tape_id = -1`), which means
moving a `Parameter` — e.g. any `std::vector<Neuron>` reallocation during
`Neuron::weights.push_back` — is only safe by accident.

**Fix:** make all five special members `= default` on both types (they are pure value types;
the compiler-generated versions are correct), and add defensive bounds checks in `record()`
and `compute_adjoints()` so an invalid id asserts instead of corrupting the heap.

### 3. `examples/LUDecomp` — the Gaussian benchmark destroys its own input

`examples/LUDecomp/main.cpp:56`, `:109`, `:117`

`solveGaussian(Matrix<T>& A, Vector<T>& b)` takes both by non-const reference and does the
elimination **in place**. The benchmark then calls it 50 times on the same `A`:

```cpp
for(int k = 0; k < M; k++) LinearSolver::solveGaussian(A, b_vectors[k]);
```

After the first call `A` is upper triangular, so iterations 2–50 solve a different system
and return garbage. Measured residuals `max |A₀x − b|` on a 4×4 case:

```
solve #0 : 1.78e-15    <- correct
solve #1 : 4.7671      <- garbage
solve #2 : 11.53       <- garbage
```

It compounds: `luDecomposition(A, L, U)` at line 117 then factorizes the **destroyed** `A`,
so TEST 2 is not solving the same system as TEST 1. The reported
`> Speedup: 193.51x` is meaningless — it compares 50 full O(n³) eliminations against one
O(n³) factorization plus 50 O(n²) back-substitutions, on different matrices.

**Fix:** take a copy inside `solveGaussian` (`Matrix<T> A = A_in;`) or pass by value, and
restore `b_vectors` between the two tests.

---

## HIGH

### 4. Unbounded tape growth — 2.2 GB for a 100×100 benchmark

`examples/LUDecomp/main.cpp` (no `reset()` anywhere), `src/ad_core.cpp:32-36`

Every `ADouble` temporary registers a permanent tape slot. The Gaussian benchmark performs
~1.7e7 multiply/subtract pairs, producing ~3.3e7 tape variables and ~6.7e7 nodes.

Measured peak RSS: **2246 MB**, Gaussian wall time 9.5 s. On a machine with less headroom
this is an OOM, not a benchmark.

Two contributing issues: the example never calls `global_tape.reset()`, and `reset()` itself
only calls `clear()`, so capacity is never returned to the OS. Add a `shrink_to_fit()` path
(or a `release()` method) for callers who are done with a large tape.

### 5. `mse_loss` indexes `targets` by `preds.size()` — out-of-bounds read

`src/neural_net.cpp:67-75`

```cpp
for (std::size_t i = 0; i < preds.size(); ++i) {
    ADouble diff = preds[i] - targets[i];   // no check that targets.size() >= preds.size()
```

Confirmed heap-buffer-overflow under ASan with 2 predictions and 1 target. A mismatched
network topology and label vector is an ordinary user error and should be an exception or
assertion, not silent memory corruption.

### 6. `Matrix` products perform no dimension checking

`inc/ad_core.hpp:134-142` and `:144-152`

`Matrix::operator*(const Vector<T>&)` iterates `j < cols` and reads `vec[j]` without
checking `vec.size() == cols`. `Matrix::operator*(const Matrix<T>&)` iterates `k < cols` and
reads `other[k][j]` without checking `other.getRows() == cols`. Both confirmed as
heap-buffer-overflow reads under ASan. `Vector::operator[]` and `Matrix::operator[]` are
likewise unchecked.

### 7. Static initialization order fiasco around `global_tape`

`src/ad_core.cpp:6`, `inc/ad_core.hpp:25`

`global_tape` is a namespace-scope object in one TU; `ADouble`'s constructor calls into it.
Any namespace-scope `ADouble` in another TU has unspecified construction order relative to
it. Confirmed — same source, two link orders:

```
g++ sio.cpp src/ad_core.cpp   ->  g_weight.id=0  var_count=0  adjoints.size=0   <-- dangling id
g++ src/ad_core.cpp sio.cpp   ->  g_weight.id=0  var_count=1  adjoints.size=1
```

In the first case the global's id points at a slot that does not exist; the first backward
pass reads out of bounds. ASan also reports an 8-byte leak from the registration that
`global_tape`'s constructor subsequently discarded.

**Fix:** Meyers singleton — replace the global object with
`AdjointTape& tape() { static AdjointTape t; return t; }`.

---

## MEDIUM

### 8. `Matrix`/`Vector` default fill shares ONE tape node across every cell

`inc/ad_core.hpp:116`, `:128`

The default argument `T val = T(0.0)` constructs a **single** `ADouble` — registering one
tape node — and then copy-fills every element with it. Since `ADouble`'s copy constructor
aliases the id, all cells share one node:

```
Matrix<ADouble> M(2,2);  Vector<ADouble> v(3);
M ids: 0 0 0 0    v ids: 1 1 1
tape var_count after constructing 7 cells = 2
```

Any cell that is never explicitly assigned (the lower triangle of `U` and upper triangle of
`L` in `luDecomposition`, for instance) remains aliased to every other untouched cell, and
`+=` into such a cell records an edge onto the shared node. It happens to be harmless in the
current examples because the shared node is a leaf, but it is a live trap for anyone who
accumulates into a default-constructed matrix.

**Fix:** value-initialize each element independently rather than copy-filling from one
prototype, or require an explicit fill value for AD types.

### 9. `pow(ADouble, ADouble)` produces NaN adjoints for non-positive bases

`src/ad_core.cpp:303-310`

`double d_exponent = res.val * std::log(base.val);` is evaluated unconditionally.
For `base ≤ 0` this is NaN/−inf even when the derivative w.r.t. the exponent is never used.
Confirmed: `pow(-2.0, 3.0)` gives the correct `dy/dbase = 12` but `dy/dexponent = -nan`,
and that NaN then propagates to every ancestor of the exponent node on the reverse sweep,
poisoning an otherwise healthy graph.

**Fix:** record the exponent edge only when `base.val > 0`, and record `0.0` (or raise)
otherwise.

### 10. No domain guards on division, `log`, or `tan`

`src/ad_core.cpp:253-270`, `:297-301`, `:284-289`

`operator/` computes `1.0 / rhs.val` and `-lhs.val / (rhs.val * rhs.val)` with no zero check;
`log` computes `1.0 / x.val`; `tan` computes `1.0 / (c*c)` at `c = cos(x) = 0`. Each writes
`inf` or `NaN` onto the tape, which silently contaminates the entire backward pass. A
debug-mode assertion or a documented policy would save a lot of debugging time.

### 11. Makefile has no header dependencies — stale objects link against a new header

`Makefile:25-26`

```make
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
```

No dependency on `inc/*.hpp`. Verified: after `touch inc/ad_core.hpp`, `make` recompiled all
six examples against the new header but left `src/ad_core.o` and `src/neural_net.o`
untouched. Since `ADouble`, `Parameter` and `AdjointTape` are all defined in that header,
any layout change produces a silent ODR/ABI mismatch at link time — exactly the kind of bug
that presents as inexplicable corruption. Stale `src/ad_core.o` and `src/neural_net.o` are
sitting in the working tree right now.

**Fix:** add `-MMD -MP` to `CXXFLAGS` and `-include $(OBJS:.o=.d)`.

### 12. Degenerate network topology dereferences an empty vector

`inc/neural_net.hpp:52`, `src/neural_net.cpp:48-53`

`NeuralNetwork({4})` (or `{}`) builds zero layers because the constructor loops from `i = 1`.
`forward()` then evaluates `layers[0]` unconditionally → null dereference, confirmed as a
SEGV under ASan with a UBSan trail of null-reference bindings. Validate
`topology.size() >= 2` in the constructor.

### 13. `mse_loss` returns a sum, not a mean

`src/neural_net.cpp:67-75`

The function accumulates squared errors and never divides by `preds.size()`. The name
implies otherwise, and the consequence is that the effective learning rate scales with batch
size. The AND examples divide by `X.size()` for the printed "Mean Loss" only — the gradient
that actually drives `nn.update(0.01)` is 4× the mean. Either divide inside the function or
rename it `sse_loss`.

---

## LOW / HYGIENE

14. **`Matrix` member order** (`inc/ad_core.hpp:125-128`): `data` is declared before
    `rows, cols` but the mem-initializer list is written `rows(r), cols(c), data(...)`. This
    is the only warning the build emits (`-Wreorder`, from `examples/LUDecomp/main.cpp:85`).
    Harmless today because the initializer uses the constructor *parameters*, but reorder the
    declarations so the code means what it appears to mean.

15. **`int` used for tape indices**: `Vector::size()` narrows `size_t` to `int`
    (`inc/ad_core.hpp:117`), and `compute_adjoints` casts `nodes.size()` to `int`
    (`src/ad_core.cpp:26`). The LU benchmark already reaches 6.7e7 nodes; a 10× larger
    problem is within a factor of ~30 of the `INT_MAX` cliff, past which the reverse loop
    silently does nothing.

16. **Unconditional 24 MB reservation** (`src/ad_core.cpp:8-11`): the default
    `AdjointTape` constructor reserves 1e6 nodes (16 MB) plus 1e6 adjoints (8 MB) at program
    start, whether or not the program uses AD.

17. **Thread safety and reproducibility**: `global_tape` is unsynchronized mutable global
    state — no two threads can tape concurrently. `Neuron::Neuron` (`src/neural_net.cpp:6-7`)
    uses a function-local `static` RNG seeded from `std::random_device`, so it is both
    non-thread-safe and impossible to seed for reproducible training runs. Expose a seed
    parameter.

18. **API gaps on `ADouble`**: no comparison operators, no `operator<<`, no `sqrt`, `abs`,
    `min`, `max`. `Matrix` has no default constructor. `ADouble(double, int)` is a public
    constructor that lets any caller forge a tape id — it should be private with `Parameter`
    as a friend.

19. **Include style** (`examples/Sensitivity/main.cpp:4`): uses
    `"../../inc/ad_core.hpp"` while every other example relies on the `-I./inc` path.
    Makes the file non-relocatable for no benefit.

---

## Suggested order of work

1. Bug 1 (tape generation stamping) — everything else is secondary to gradients being right.
2. Bug 2 (`= default` the special members; bounds-check `record`/`compute_adjoints`).
3. Bug 3 + 4 (fix the LU example; it is the project's showcase benchmark and is currently
   both wrong and a 2 GB memory hog).
4. Bugs 5, 6, 12 (input validation — cheap, and they turn memory corruption into errors).
5. Bug 11 (Makefile dependencies) before anyone else clones the repo.
6. The rest as cleanup.

A `tests/` target running finite-difference gradient checks over every operator, plus a
sanitizer build (`make asan`), would have caught 1, 2, 5, 6 and 12 automatically.
