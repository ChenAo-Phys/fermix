# fermix

Fast batched `slogdet` / `slogpf` / `det` / `pf` for fp32 matrices on NVIDIA GPUs, written in JAX with Pallas (Triton) kernels.

Designed for quantum Monte Carlo in fermionic systems: a large batch of moderate-size matrices (n ~ 64-1024), forward and backward.

## Install

```bash
pip install -e .          # needs a CUDA build of jax >= 0.11
pytest                    # on a CUDA GPU this tests the kernels, elsewhere the fallback
```

## Usage

```python
import jax, jax.numpy as jnp
from fermix import slogdet, slogpf, det, pf

A = jax.random.normal(jax.random.key(0), (4096, 128, 128), jnp.float32)
sign, logabs = slogdet(A) # like jnp.linalg.slogdet, ~4x faster at this size
S = A - jnp.swapaxes(A, -1, -2)
sign, logabs = slogpf(S) # Pfaffian of the skew-symmetric batch
```

All functions accept any leading batch dimensions, any n, and are `jit`/`vmap`/`grad`/`jvp` compatible.

Neither `slogdet` nor `slogpf` is singular-safe in the backward (they return inf/NaN for a
rank-deficient input), while `det` and `pf` have singular-safe backward.

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

Internal developer notes live in `AGENTS.md`. `tests/` holds a compact pytest suite (forward vs float64 NumPy, tiny-n polynomials, gradients vs analytic
references, singular inputs). `benchmarks/bench.py` reproduces the table above.

`pip install -e .[dev]` adds the tooling used by CI (`.github/workflows/`): `black --check .`,
`pyright`, and `pytest` — the last one on CPU, which exercises the generic fallback path rather than the kernels.
