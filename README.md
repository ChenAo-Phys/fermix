# fermix

Fast batched `slogdet` / `slogpf` / `det` / `pf` for fp32 matrices on NVIDIA GPUs, written in JAX with Pallas
(Triton) kernels, with gradients that stay finite for singular matrices.

Designed for variational Monte Carlo with fermionic neural quantum states: thousands of small matrices
(n ~ 64–256, batch ~ 4096), forward and backward per optimisation step.

## Install

```bash
pip install -e .          # needs a CUDA build of jax >= 0.11
pytest                    # a GPU is required for the tests
```

## Usage

```python
import jax, jax.numpy as jnp
from fermix import slogdet, slogpf, det, pf

A = jax.random.normal(jax.random.PRNGKey(0), (4096, 128, 128), jnp.float32)
sign, logabs = slogdet(A)                    # like jnp.linalg.slogdet, ~4x faster at this size
S = A - jnp.swapaxes(A, -1, -2)
sign, logabs = slogpf(S)                     # Pfaffian of the skew-symmetric batch (input is skew-symmetrised)
d = det(A[:, :40, :40]); p = pf(S[:, :40, :40])

g = jax.grad(lambda a: slogdet(a)[1].sum())(A)    # A^-T, finite even for exactly singular members
```

All functions accept any leading batch dimensions, any n (padded internally to a multiple of 32 without extra flops),
and are `jit`/`vmap`/`grad`/`jvp` compatible. Options: `prec="tf32x3"` (default, tensor cores at fp32 accuracy) or
`"ieee"` (exact fp32 FMA); `slogpf`/`pf` take `skew_symmetrize` (default True; the input is replaced by `(a - a^T)/2`).

## What is inside

| function | forward | backward |
|---|---|---|
| `slogdet` | blocked LU with partial pivoting (register-resident 8-column panels, tensor-core block updates); explicit polynomial for n ≤ 4 | `d log|det| = tr(A⁻¹ dA)`, `d sign = 0`; `A⁻ᵀ` is rebuilt from the forward's own LU factors (no second factorisation); exactly singular matrices get a **zero** gradient, never NaN |
| `slogpf` | blocked Parlett–Reid tridiagonalisation of the skew-symmetric matrix, rank-2 updates as one GEMM over the lower-triangle tiles; polynomial for n ≤ 6 | `d log|pf| = ½ tr(S⁻¹ dS)` with the same LU-based inverse of S; zero gradient for exactly singular S |
| `det` | `sign · exp(logabs)` | `d det = tr(adj(A) dA)`: the adjugate is formed from the LU (finite for singular A: rank-1 for rank n−1, zero for rank ≤ n−2) |
| `pf` | `sign · exp(logabs)` | `½ pf S⁻ᵀ` continued to singular S: for rank n−2 the finite rank-2 matrix from the null space and one minor Pfaffian, zero for rank ≤ n−4 |

Neither `jnp.linalg.slogdet` nor lrux's `slogpf` is singular-safe in the backward (they return inf/NaN for a
rank-deficient input); `jnp.linalg.det` is, but only when the zero pivot is the last one. Here the det/pf gradients
are correct for any position of up to two exact zero pivots, and near-singular matrices get the exact large gradient.

## Performance (A100-80GB, B = 4096, fp32)

| n | `slogdet` | `jnp.linalg.slogdet` | `slogpf` | `lrux.slogpf` | `grad(slogdet)` | `jax.grad(jnp.linalg.slogdet)` | `grad(slogpf)` |
|---|---|---|---|---|---|---|---|
| 128 | 2.7 ms | 11.8 ms | 2.6 ms | 90 ms | 7.8 ms | 27 ms | 8.5 ms |
| 256 | 14 ms | 61 ms | 13.6 ms | 670 ms | 30 ms | 106 ms | 38 ms |

Forward numbers on an idle GPU, gradient numbers on a shared one (compare within a column block). Time per GiB of
input stays flat (13–30 ms) from n = 64 to 1024. First call compiles ~15 s at n = 256 (kernels are specialised per
n), ~65 s at n = 1024.

Accuracy: signs exact; log|det| rms error 1e-4 – 1e-3 at n = 128–1024, the same fp32 class as cuSOLVER.

## Notes and limits

- The kernels run for float32 on CUDA GPUs. Any other dtype (float64) or device (CPU, an array committed to a
  CPU device, a non-CUDA default backend) emits a `FermixFallbackWarning` and uses a generic jax.numpy implementation
  (LU for det, masked batched Parlett–Reid for pf) with the same outputs and the same singular-safe derivative rules,
  but far slower. Other dtypes raise `TypeError`. Tested on A100 with jax 0.11.1; Hopper tile configs untested.
- Memory: the forward needs two (B, n, n) work buffers; the gradient about five more.
- `det`/`pf` overflow fp32 for large, badly scaled matrices — use `slogdet`/`slogpf` then.
- Structured singularities such as a zero row/column give exact zero pivots (and the exact zero-gradient / adjugate
  behaviour above); duplicate rows give roundoff-level pivots and correspondingly large but finite gradients.
- `fermix._inverse.GRAD_INVERSE = "cusolver"` switches the log-derivative inverse to a cuSOLVER reference path.

## Development

`tests/` holds a compact pytest suite (forward vs float64 NumPy, tiny-n polynomials, gradients vs analytic
references, singular inputs). `benchmarks/bench.py` reproduces the table above.
