# Changelog

## Bug-fix pass — 2026-09-13

Fixes for the 19 issues in the code review. `make test` runs the new regression
suite (71 checks); `make asan` runs it under AddressSanitizer + UBSan.

### Breaking API changes

| Before | After | Why |
| --- | --- | --- |
| `global_tape` | `tape()` | A namespace-scope tape had unspecified construction order relative to globals in other TUs. |
| `mse_loss` returned a sum | returns the mean | The name said mean; gradients scaled with output count. `sse_loss` is the sum. |
| `Neuron(int)`, `Layer(int,int,bool)` | take a `std::mt19937&` | A function-local static RNG is neither thread-safe nor seedable. |
| `NeuralNetwork(topology)` | also `NeuralNetwork(topology, seed)` | Reproducible training runs. |
| `Vector::size()`, `Matrix::getRows/getCols` return `int` | return `std::size_t` | Consistent indexing. |
| `Matrix(r, c, T val = T(0.0))` | `Matrix(r, c)` and `Matrix(r, c, val)` | The default argument shared one tape node across every cell. |
| `ADouble(double, int)` public | private, `friend struct Parameter` | Forging tape ids broke the tape invariant. |
| `Parameter(double)` implicit | `explicit` | — |

### Critical

1. **`reset()` silently corrupted every `Parameter`.** `AdjointTape` now carries a
   `generation` counter bumped by `reset()`; `Parameter` records the generation it
   bound to and `var()` re-registers whenever they differ. A `reset()` with no
   intervening `update()` — an inference or validation pass — is now safe.
2. **Moved-from `ADouble`/`Parameter` carried `id = -1`.** Both are non-owning value
   types, so all five special members are `= default`. `record()` and
   `compute_adjoints()` validate ids and throw instead of corrupting the heap.
3. **`solveGaussian` destroyed its input.** It now takes `Matrix<T>` and `Vector<T>`
   by value. The benchmark verifies `max |Ax-b|` for both solvers.

### High

4. **Unbounded tape growth.** The LU benchmark resets the tape per right-hand side
   and rebuilds its AD inputs outside the timed region; added `AdjointTape::release()`
   to return capacity. Peak RSS 2246 MB → 47 MB.
5. **`mse_loss` read past `targets`.** Size and emptiness are validated.
6. **`Matrix` products did not check dimensions.** Both overloads throw
   `std::invalid_argument`; `operator[]` is bounds-checked unless `NDEBUG`.
7. **Static initialization order.** `global_tape` replaced by the `tape()` Meyers
   singleton.

### Medium

8. **Shared tape nodes in `Matrix`/`Vector`.** The value-initialising constructors
   build each element independently; the fill constructor is now explicit.
9. **`pow(ADouble, ADouble)` emitted NaN adjoints** for non-positive bases. The
   exponent partial is recorded as `0` there (the exponent derivative does not
   exist); the base partial is unchanged.
10. **No domain guards.** Division by zero, `log(x <= 0)`, `sqrt(x <= 0)`, `tan` at a
    pole and `pow` with an unbounded derivative throw `std::domain_error`.
11. **Makefile had no header dependencies.** `-MMD -MP` plus `-include $(DEPS)`.
12. **Degenerate topologies segfaulted.** `NeuralNetwork` validates `topology`, and
    `forward()` validates input width.
13. **`mse_loss` was a sum.** Now the mean; `sse_loss` added for the sum.

### Low

14. `Matrix` members reordered to match the initialiser list (`-Wreorder` gone).
15. The reverse sweep iterates over `std::size_t`; `register_variable()` throws
    before `var_count` overflows.
16. No 24 MB reservation at startup; use `tape().reserve(...)` to pre-allocate.
17. RNG is injected and seedable; no shared static.
18. Added comparisons, `operator<<`, `sqrt`, `abs`/`fabs`, unary `-`, `fmax`/`fmin`,
    `ADouble::grad()`, `Parameter::grad()`/`is_bound()`/`zero_state()`,
    `Matrix` default constructor, `at()` accessors.
19. `examples/Sensitivity` uses the `-I./inc` include path like every other example.
