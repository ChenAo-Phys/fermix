"""Forward and gradient timings of fermix vs jnp.linalg.slogdet / lrux.slogpf. Usage: python bench.py [cuda_id]"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"
import jax, jax.numpy as jnp
from fermix import slogdet, slogpf


def bench(f, *args, n_rep=5, **kw):
    out = f(*args, **kw); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        out = f(*args, **kw)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_rep * 1e3


if __name__ == "__main__":
    try:
        import lrux
    except ImportError:
        lrux = None
    B = 4096
    g_det = jax.jit(jax.grad(lambda a: slogdet(a)[1].sum()))
    g_jnp = jax.jit(jax.grad(lambda a: jnp.linalg.slogdet(a)[1].sum()))
    g_pf = jax.jit(jax.grad(lambda a: slogpf(a, skew_symmetrize=False)[1].sum()))
    print(f"device: {jax.devices()[0].device_kind}, B={B}  (ms)")
    print(f"{'n':>4} | {'slogdet':>8} {'jnp':>8} | {'slogpf':>8} {'lrux':>8} | {'grad det':>9} {'jax grad':>9} | {'grad pf':>8}")
    for n in (64, 128, 192, 256):
        A = jax.random.normal(jax.random.PRNGKey(0), (B, n, n), jnp.float32)
        S = A - jnp.swapaxes(A, -1, -2)
        t_lrux = bench(jax.jit(lrux.slogpf), S, n_rep=2) if lrux is not None else float("nan")
        print(f"{n:>4} | {bench(slogdet, A):>8.2f} {bench(jax.jit(jnp.linalg.slogdet), A, n_rep=3):>8.2f} | "
              f"{bench(slogpf, S, skew_symmetrize=False):>8.2f} {t_lrux:>8.1f} | {bench(g_det, A):>9.2f} {bench(g_jnp, A, n_rep=3):>9.2f} | "
              f"{bench(g_pf, S):>8.2f}", flush=True)
