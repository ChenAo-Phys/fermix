# fermix

Fast batched `slogdet` / `slogpf` / `det` / `pf` for fp32 matrices on NVIDIA GPUs, written in JAX with Pallas (Triton) kernels.

Designed for quantum Monte Carlo in fermionic systems: a large batch of moderate-size matrices (n ~ 64-1024), forward and backward.

## Install

```bash
pip install -e .          # needs a CUDA build of jax >= 0.7.1
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

`slogdet`/`slogpf` guard exactly singular inputs with a **zero** gradient rather than inf/NaN, but
d log|det| is genuinely infinite there, so that value is a guard and not a derivative. `det` and `pf`
are truly singular-safe: their adjugate / null-space gradients are the correct finite derivatives for
up to two exact zero pivots, in any position.

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
references, singular inputs). `benchmarks/bench.py` times the forward
and the gradient against `jnp.linalg.slogdet` and, for the Pfaffian, against the generic XLA path.

`pip install -e .[dev]` adds the tooling used by CI (`.github/workflows/`): `black --check .`,
`pyright`, and `pytest` — the last one on CPU, which exercises the generic fallback path rather than the kernels.
