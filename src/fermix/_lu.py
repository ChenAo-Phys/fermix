"""Blocked LU with partial pivoting (Pallas Triton kernels) for slogdet: 1-warp register-resident inner panels,
4-warp tensor-core inter-panel update / U-row / trailing GEMM kernels, ping-pong buffers, virtual pivots.
With factors=True, _lu_core also returns what _inverse needs to rebuild the packed LU."""
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._common import (f32, INNER, LU_CHUNK_COST, _next_pow2, _layout, _argmax_chunks, _dot, _unit_lower_inv,
                      _full, _vec, _embed, _panel_warps, _lu_block)


# ----------------------------------------------------------------------------- inner panel (1 warp)
def _lu_inner_kernel(p_ref, q_ref, pos_ref, sl_ref, piv_ref, po_ref, qo_ref, poso_ref, slo_ref, pivo_ref, src_ref,
                     *, n, r0, last, i, b, bi, layout, unroll_steps=False):
    """Inner panel i (columns [c0, c0+8)) of the block starting at r0; the m = n - r0 active rows are held as a list of
    power-of-2 register chunks (exact cover, no padding). Pivoting is virtual (pos), rows are never moved."""
    del po_ref, qo_ref, poso_ref, slo_ref, pivo_ref  # aliased
    chunks = list(layout)               # (offset rel. to r0, height); negative offsets = dead rows above r0
    nin = b // bi
    c0 = r0 + i * bi
    ci = lax.broadcasted_iota(jnp.int32, (bi,), 0)
    ci2 = ci[None, :]
    ib = lax.broadcasted_iota(jnp.int32, (bi, bi), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bi, bi), 1)
    offs = [off for off, h in chunks]
    trs = [lax.broadcasted_iota(jnp.int32, (h,), 0) + off for off, h in chunks]      # active-local row index
    rows = lambda off, h: pl.ds(r0 + off, h)

    if i == 0:
        poss = list(trs)
    else:
        poss = [pos_ref[rows(off, h)] for off, h in chunks]
    unas = [pos >= i * bi for pos in poss]
    sign = sl_ref[0]
    logabs = sl_ref[1]
    Ws = [p_ref[rows(off, h), pl.ds(c0, bi)] for off, h in chunks]
    if i % 2 == 1:
        # prologue: W -= L_{i-1} @ (Linv_{i-1} @ A'[piv_{i-1}, cols_i])
        cp = c0 - bi
        piv_prev = piv_ref[pl.ds((i - 1) * bi, bi)]
        Linv_prev = q_ref[pl.ds(cp, bi), pl.ds(cp, bi)]
        Ap = p_ref[piv_prev, pl.ds(c0, bi)]
        U = jnp.sum(Linv_prev[:, :, None] * Ap[None, :, :], axis=1)             # (bi, bi)
        for c, (off, h) in enumerate(chunks):
            Lp = p_ref[rows(off, h), pl.ds(cp, bi)]
            for t in range(bi):
                lt = jnp.sum(jnp.where(ci2 == t, Lp, 0.0), axis=1)
                ut = jnp.sum(jnp.where(ib == t, U, 0.0), axis=0)
                Ws[c] = Ws[c] - lt[:, None] * ut[None, :]

    def step(j, carry):
        Ws, Ut, piv_i, poss, unas, sign, logabs = carry
        jg = i * bi + j
        u = jnp.sum(jnp.where((jb == j) & (ib < j), Ut, 0.0), axis=1)
        v = jnp.where(ci == j, 1.0, jnp.where(ci < j, -u, 0.0))
        cols = [jnp.sum(W * v[None, :], axis=1) for W in Ws]
        cands = [jnp.where(un, jnp.abs(col), -1.0) for un, col in zip(unas, cols)]
        p = _argmax_chunks(cands, offs)
        rowp = sum(jnp.sum(jnp.where(tr[:, None] == p, W, 0.0), axis=0) for tr, W in zip(trs, Ws))
        urow = rowp - jnp.sum(Ut * rowp[:, None], axis=0)
        pivot = jnp.sum(jnp.where(ci == j, urow, 0.0))
        Ut = jnp.where(ib == j, jnp.where(jb >= j, urow[None, :], 0.0), Ut)
        piv_i = jnp.where(ci == j, p, piv_i)
        q = sum(jnp.sum(jnp.where(tr == p, pos, 0)) for tr, pos in zip(trs, poss))
        poss = [jnp.where(tr == p, jg, jnp.where(pos == jg, q, pos)) for tr, pos in zip(trs, poss)]
        unas = [un & (tr != p) for tr, un in zip(trs, unas)]
        sign = sign * lax.select(q != jg, f32(-1.0), f32(1.0)) * jnp.sign(pivot)
        logabs = logabs + jnp.log(jnp.abs(pivot))
        inv = lax.select(pivot == 0.0, f32(0.0), 1.0 / pivot)      # zero pivot: L column 0, log -> -inf, sign 0
        Ws = [jnp.where((ci2 == j) & un[:, None], (col * inv)[:, None], W) for W, col, un in zip(Ws, cols, unas)]
        return Ws, Ut, piv_i, poss, unas, sign, logabs

    carry = (Ws, jnp.zeros((bi, bi), f32), jnp.zeros((bi,), jnp.int32), poss, unas, sign, logabs)
    if unroll_steps:
        for j in range(bi):
            carry = step(j, carry)
    else:
        carry = lax.fori_loop(0, bi, step, carry)
    Ws, Ut, piv_i, poss, unas, sign, logabs = carry
    for c, (off, h) in enumerate(chunks):
        if off < 0:     # padded tile: keep the dead rows above r0 intact (they hold the previous block's U rows)
            plgpu.store(p_ref.at[rows(off, h), pl.ds(c0, bi)], Ws[c], mask=(trs[c] >= 0)[:, None])
        else:
            p_ref[rows(off, h), pl.ds(c0, bi)] = Ws[c]
        pos_ref[rows(off, h)] = poss[c]
    plgpu.debug_barrier()
    sl_ref[0] = sign
    sl_ref[1] = logabs
    piv_ref[pl.ds(i * bi, bi)] = r0 + piv_i                                     # global pivot rows
    if i % 2 == 0:
        # 8x8 unit-lower inverse of L_ii (pivot rows) -> Q diag block (used by the next inner panel's prologue)
        Lii = p_ref[r0 + piv_i, pl.ds(c0, bi)]
        q_ref[pl.ds(c0, bi), pl.ds(c0, bi)] = _unit_lower_inv(Lii, bi)
    else:
        # 16x16 unit-lower inverse of the L block of panels (i-1, i) -> Q diag (update kernel + urow kernel)
        plgpu.debug_barrier()
        cp = c0 - bi
        piv16 = piv_ref[pl.ds((i - 1) * bi, 2 * bi)]
        L16 = p_ref[piv16, pl.ds(cp, 2 * bi)]
        q_ref[pl.ds(cp, 2 * bi), pl.ds(cp, 2 * bi)] = _unit_lower_inv(L16, 2 * bi)
    if i == nin - 1:
        for tr, pos, un in zip(trs, poss, unas):
            plgpu.store(src_ref.at[r0 + pos], r0 + tr, mask=un)


# ----------------------------------------------------------------------------- inter-panel update (4 warps)
def _lu_update_kernel(p_ref, q_ref, piv_ref, po_ref, *, n, r0, b, bi, ch, prec, lo, mid, hi):
    """Right-looking update of inner panels [mid, hi) by the finished panels [lo, mid) (units of bi=8 columns):
    P[rows, target] -= L[rows, source] @ (L_ss^{-1} @ P[piv_source, target]).  K = 8(mid-lo) in {16, 32}."""
    del po_ref
    rc = r0 + pl.program_id(1) * ch
    rows = rc + lax.broadcasted_iota(jnp.int32, (ch,), 0)
    rvalid = rows < n
    K = (mid - lo) * bi
    Wd = (hi - mid) * bi
    c_lo = r0 + lo * bi
    c_mid = r0 + mid * bi
    piv_src = piv_ref[pl.ds(lo * bi, K)]
    if K == 16:
        Linv = q_ref[pl.ds(c_lo, 16), pl.ds(c_lo, 16)]
        A = p_ref[piv_src, pl.ds(c_mid, Wd)]
        U = _dot(Linv, A, prec)                                           # (16, Wd)
        Wl = plgpu.load(p_ref.at[pl.ds(rc, ch), pl.ds(c_lo, 16)], mask=rvalid[:, None], other=0.0)
        upd = _dot(Wl, U, prec)
    else:   # K == 32: block forward substitution with the two 16x16 inverse blocks
        pivA = piv_ref[pl.ds(lo * bi, 16)]
        pivB = piv_ref[pl.ds(lo * bi + 16, 16)]
        Linv_a = q_ref[pl.ds(c_lo, 16), pl.ds(c_lo, 16)]
        Linv_b = q_ref[pl.ds(c_lo + 16, 16), pl.ds(c_lo + 16, 16)]
        L_ba = p_ref[pivB, pl.ds(c_lo, 16)]
        Ua = _dot(Linv_a, p_ref[pivA, pl.ds(c_mid, Wd)], prec)
        Ub = _dot(Linv_b, p_ref[pivB, pl.ds(c_mid, Wd)] - _dot(L_ba, Ua, prec), prec)
        Wa = plgpu.load(p_ref.at[pl.ds(rc, ch), pl.ds(c_lo, 16)], mask=rvalid[:, None], other=0.0)
        Wb = plgpu.load(p_ref.at[pl.ds(rc, ch), pl.ds(c_lo + 16, 16)], mask=rvalid[:, None], other=0.0)
        upd = _dot(Wa, Ua, prec) + _dot(Wb, Ub, prec)
    T = plgpu.load(p_ref.at[pl.ds(rc, ch), pl.ds(c_mid, Wd)], mask=rvalid[:, None], other=0.0)
    # never write the pivot rows: other programs gather A from them (cross-program RAW hazard otherwise)
    keep = (jnp.sum((rows[:, None] == piv_src[None, :]).astype(jnp.int32), axis=1) == 0) & rvalid
    plgpu.store(p_ref.at[pl.ds(rc, ch), pl.ds(c_mid, Wd)], T - upd, mask=keep[:, None])


# ----------------------------------------------------------------------------- urow (4 warps)
def _lu_urow_kernel(p_ref, piv_ref, q_ref, out_ref, *, n, r0, b, tn, prec):
    del out_ref
    bu = 16
    nblk = b // bu
    c0 = r0 + b + pl.program_id(1) * tn
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    cvalid = (c0 + cj) < n
    U_list = []
    for i in range(nblk):
        piv_i = piv_ref[pl.ds(i * bu, bu)]
        R = plgpu.load(p_ref.at[piv_i, pl.ds(c0, tn)], mask=cvalid[None, :], other=0.0)
        for s_ in range(i):
            Lis = p_ref[piv_i, pl.ds(r0 + s_ * bu, bu)]
            R = R - _dot(Lis, U_list[s_], prec)
        Linv = q_ref[pl.ds(r0 + i * bu, bu), pl.ds(r0 + i * bu, bu)]
        Ui = _dot(Linv, R, prec)
        U_list.append(Ui)
        plgpu.store(q_ref.at[pl.ds(r0 + i * bu, bu), pl.ds(c0, tn)], Ui, mask=cvalid[None, :])


# ----------------------------------------------------------------------------- gemm (4 warps)
def _lu_gemm_kernel(p_ref, q_ref, src_ref, out_ref, *, n, r0, b, tm, tn, prec):
    del out_ref
    i0 = pl.program_id(1) * tm
    c0 = pl.program_id(2) * tn
    s0 = r0 + b
    mrem = n - s0
    ri = lax.broadcasted_iota(jnp.int32, (tm,), 0)
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    rvalid = (i0 + ri) < mrem
    cvalid = (c0 + cj) < mrem
    src = plgpu.load(src_ref.at[pl.ds(s0 + i0, tm)], mask=rvalid, other=n - 1)
    Lt = p_ref[src, pl.ds(r0, b)]
    cols = pl.ds(s0 + c0, tn)
    U = plgpu.load(q_ref.at[pl.ds(r0, b), cols], mask=cvalid[None, :], other=0.0)
    C = plgpu.load(p_ref.at[src, cols], mask=rvalid[:, None] & cvalid[None, :], other=0.0)
    C = C - _dot(Lt, U, prec)
    plgpu.store(q_ref.at[pl.ds(s0 + i0, tm), cols], C, mask=rvalid[:, None] & cvalid[None, :])


# ----------------------------------------------------------------------------- calls


def _inner_call(P, Q, pos, sl, piv, *, r0, last, i, b, bi, layout, num_warps, unroll_steps, inner_kw=(), fresh=False):
    """fresh=True: Q, pos, sl, piv, src are allocated as uninitialized outputs (first call only)."""
    B, n, _ = P.shape
    kern = functools.partial(_lu_inner_kernel, n=n, r0=r0, last=last, i=i, b=b, bi=bi, layout=layout, unroll_steps=unroll_steps)
    out_specs = [_full(n), _full(n), _vec(n), _vec(2), _vec(b), _vec(n)]
    out_shape = [jax.ShapeDtypeStruct((B, n, n), f32), jax.ShapeDtypeStruct((B, n, n), f32),
                 jax.ShapeDtypeStruct((B, n), jnp.int32), jax.ShapeDtypeStruct((B, 2), f32),
                 jax.ShapeDtypeStruct((B, b), jnp.int32), jax.ShapeDtypeStruct((B, n), jnp.int32)]
    if fresh:
        return pl.pallas_call(
            functools.partial(_fresh_wrapper, kern, B=B),
            grid=(B,), in_specs=[_full(n)], out_specs=out_specs, out_shape=out_shape,
            input_output_aliases={0: 0},
            compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
        )(P)
    return pl.pallas_call(
        kern, grid=(B,),
        in_specs=[_full(n), _full(n), _vec(n), _vec(2), _vec(b)],
        out_specs=out_specs, out_shape=out_shape,
        input_output_aliases={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
    )(P, Q, pos, sl, piv)


def _fresh_wrapper(kern, p_ref, po_ref, qo_ref, poso_ref, slo_ref, pivo_ref, src_ref, *, B):
    """First inner call: initialise sign/log and call the kernel with the outputs acting as the (uninitialised) state."""
    slo_ref[0] = f32(1.0)
    slo_ref[1] = f32(0.0)
    kern(p_ref, qo_ref, poso_ref, slo_ref, pivo_ref, po_ref, qo_ref, poso_ref, slo_ref, pivo_ref, src_ref)

def _update_call(P, Q, piv, *, r0, b, bi, ch, prec, num_warps, lo, mid, hi):
    B, n, _ = P.shape
    m = n - r0
    ch = min(ch, _next_pow2(m))
    kern = functools.partial(_lu_update_kernel, n=n, r0=r0, b=b, bi=bi, ch=ch, prec=prec, lo=lo, mid=mid, hi=hi)
    return pl.pallas_call(
        kern, grid=(B, -(-m // ch)),
        in_specs=[_full(n), _full(n), _vec(b)],
        out_specs=_full(n),
        out_shape=jax.ShapeDtypeStruct((B, n, n), f32),
        input_output_aliases={0: 0},
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
    )(P, Q, piv)

def _urow_call(P, piv, Q, *, r0, b, tn, prec, num_warps):
    B, n, _ = P.shape
    nct = -(-(n - r0 - b) // tn)
    kern = functools.partial(_lu_urow_kernel, n=n, r0=r0, b=b, tn=tn, prec=prec)
    return pl.pallas_call(
        kern, grid=(B, nct),
        in_specs=[_full(n), _vec(b), _full(n)],
        out_specs=_full(n),
        out_shape=jax.ShapeDtypeStruct((B, n, n), f32),
        input_output_aliases={2: 0},
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
    )(P, piv, Q)

def _gemm_call(P, Q, src, *, r0, b, tm, tn, prec, num_warps):
    B, n, _ = P.shape
    mrem = n - r0 - b
    kern = functools.partial(_lu_gemm_kernel, n=n, r0=r0, b=b, tm=tm, tn=tn, prec=prec)
    return pl.pallas_call(
        kern, grid=(B, -(-mrem // tm), -(-mrem // tn)),
        in_specs=[_full(n), _full(n), _vec(n)],
        out_specs=_full(n),
        out_shape=jax.ShapeDtypeStruct((B, n, n), f32),
        input_output_aliases={1: 0},
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
    )(P, Q, src)


def _lu_core(A, n, prec, unroll_steps, block=None, factors=False):
    """Blocked LU of the (B, n, n) batch; returns (sign, logabs) and, with factors=True, also the raw material for
    the packed LU (see _packed_lu): the two ping-pong buffers, per-block pivot rows / compaction maps and a snapshot of
    each block's panel columns taken before its factorisation (the pivot rows' U part is recomputed from it)."""
    b = _lu_block(n) if block is None else block
    bi = INNER
    nin = b // bi
    assert nin in (4, 8)
    P, nb = _embed(A, n, lambda m: jnp.eye(m, dtype=f32), b)
    N = P.shape[1]
    Q = pos = sl = piv = None
    pivs, srcs, snaps = [], [], []
    for k in range(nb):
        r0 = k * b
        last = k == nb - 1
        m = N - r0
        layout = tuple(_layout(m, r0, LU_CHUNK_COST))
        pw = _panel_warps(sum(h for _, h in layout))
        if factors:
            # the barrier forces the slice to be taken before the in-place panel kernels (otherwise XLA fuses it
            # into its downstream gather and has to copy the whole buffer once per block)
            snap, P = lax.optimization_barrier((P[:, r0:, r0:r0 + b], P))
            snaps.append(snap)
        for i in range(nin):
            P, Q, pos, sl, piv, src = _inner_call(P, Q, pos, sl, piv, r0=r0, last=last, i=i, b=b, bi=bi, layout=layout, num_warps=pw,
                                                  unroll_steps=unroll_steps, fresh=(k == 0 and i == 0))
            if i == 1:
                P = _update_call(P, Q, piv, r0=r0, b=b, bi=bi, ch=64, prec=prec, num_warps=4, lo=0, mid=2, hi=4)
            if nin == 8 and i == 3:
                P = _update_call(P, Q, piv, r0=r0, b=b, bi=bi, ch=64, prec=prec, num_warps=4, lo=0, mid=4, hi=8)
            if nin == 8 and i == 5:
                P = _update_call(P, Q, piv, r0=r0, b=b, bi=bi, ch=64, prec=prec, num_warps=4, lo=4, mid=6, hi=8)
        if factors:
            pivs.append(piv)
            srcs.append(src)
        if not last:
            Q = _urow_call(P, piv, Q, r0=r0, b=b, tn=64, prec=prec, num_warps=4)
            Q = _gemm_call(P, Q, src, r0=r0, b=b, tm=64, tn=64, prec=prec, num_warps=4)
            P, Q = Q, P
    if factors:
        bufs = (P, Q) if (nb - 1) % 2 == 0 else (Q, P)        # bufs[k % 2] holds the panel of block k
        return sl[:, 0], sl[:, 1], (bufs, pivs, srcs, snaps, b, N)
    return sl[:, 0], sl[:, 1]
