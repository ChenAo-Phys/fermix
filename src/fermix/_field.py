"""The scalar field of a kernel: float32 / float64 values are plain arrays, complex64 /
complex128 values are ``CVal`` pairs (re, im) of real arrays, so one kernel body serves
the four dtypes (Triton has no complex type). The module also holds the ref access
helpers that read / write a value through its k component refs and ``_pcall``, the
pallas_call wrapper that flattens the components of every logical argument.

Conventions: a *value* is an array or a CVal; *parts* are the tuple of its real
components (length 1 or 2); a matrix buffer of the field is always handled as parts
(``tuple[Array, ...]``) and enters a kernel as a tuple of refs."""

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

PREC = {
    "ieee": lax.Precision.HIGHEST,
    "tf32x3": lax.DotAlgorithmPreset.TF32_TF32_F32_X3,
}
KINDS = {
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.complex64): "c64",
    jnp.dtype(jnp.complex128): "c128",
}


@tree_util.register_pytree_node_class
class CVal:
    """A complex value as two real arrays; supports the arithmetic the kernels use
    (+, -, unary -, * by a CVal / real array / scalar, indexing, .T, conj)."""

    __slots__ = ("re", "im")

    def __init__(self, re, im):
        self.re = re
        self.im = im

    def tree_flatten(self):
        return (self.re, self.im), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    @property
    def shape(self):
        return jnp.shape(self.re)

    @property
    def ndim(self):
        return jnp.ndim(self.re)

    @property
    def dtype(self):
        return jnp.result_type(self.re)

    def __getitem__(self, idx):
        return CVal(self.re[idx], self.im[idx])

    @property
    def T(self):
        return CVal(self.re.T, self.im.T)

    def conj(self):
        return CVal(self.re, -self.im)

    def __neg__(self):
        return CVal(-self.re, -self.im)

    def __add__(self, o):
        if isinstance(o, CVal):
            return CVal(self.re + o.re, self.im + o.im)
        return CVal(self.re + o, self.im)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, CVal):
            return CVal(self.re - o.re, self.im - o.im)
        return CVal(self.re - o, self.im)

    def __rsub__(self, o):
        return CVal(o - self.re, -self.im)

    def __mul__(self, o):
        if isinstance(o, CVal):
            re = self.re * o.re - self.im * o.im
            im = self.re * o.im + self.im * o.re
            return CVal(re, im)
        return CVal(self.re * o, self.im * o)

    __rmul__ = __mul__


def parts(x):
    """The real components of a value."""
    return (x.re, x.im) if isinstance(x, CVal) else (x,)


def from_parts(p):
    return CVal(*p) if len(p) == 2 else p[0]


def where(c, x, y):
    """jnp.where for values (a real branch paired with a complex one is promoted)."""
    if isinstance(x, CVal) or isinstance(y, CVal):
        xr, xi = (x.re, x.im) if isinstance(x, CVal) else (x, 0.0)
        yr, yi = (y.re, y.im) if isinstance(y, CVal) else (y, 0.0)
        return CVal(jnp.where(c, xr, yr), jnp.where(c, xi, yi))
    return jnp.where(c, x, y)


def vsum(x, axis=None):
    if isinstance(x, CVal):
        return CVal(jnp.sum(x.re, axis=axis), jnp.sum(x.im, axis=axis))
    return jnp.sum(x, axis=axis)


def vadd(values):
    """Sum of a list of values (real or CVal)."""
    it = iter(values)
    acc = next(it)
    for v in it:
        acc = acc + v
    return acc


def abs1(x):
    """Pivot magnitude: |x| for real, |re| + |im| for complex (LAPACK's cabs1: cheaper
    than the modulus and equally valid for choosing a pivot)."""
    if isinstance(x, CVal):
        return jnp.abs(x.re) + jnp.abs(x.im)
    return jnp.abs(x)


def mag(x):
    """The modulus |x| (used for log|pivot|)."""
    if isinstance(x, CVal):
        return jnp.sqrt(x.re * x.re + x.im * x.im)
    return jnp.abs(x)


def iszero(x):
    if isinstance(x, CVal):
        return (x.re == 0.0) & (x.im == 0.0)
    return x == 0.0


def _safe_inv(x):
    """1 / x with 1 / 0 -> 0 (lax.select: jnp.where on scalars can lower badly)."""
    return lax.select(x == 0.0, jnp.zeros_like(x), 1.0 / x)


def recip(x):
    """Guarded reciprocal: 1 / x, 0 -> 0; complex as conj(x) / |x|^2."""
    if isinstance(x, CVal):
        inv = _safe_inv(x.re * x.re + x.im * x.im)
        return CVal(x.re * inv, -(x.im * inv))
    return _safe_inv(x)


def unit(x):
    """x / |x| (the sign of a pivot), 0 -> 0."""
    if isinstance(x, CVal):
        inv = _safe_inv(mag(x))
        return CVal(x.re * inv, x.im * inv)
    return jnp.sign(x)


def conj(x):
    return x.conj() if isinstance(x, CVal) else x


def transpose(x):
    return x.T


def dot(a, b, prec, three=False):
    """a @ b on register tiles: one Triton dot for real values; for complex ones four
    real dots (or Gauss' three-multiplication form with ``three=True``: 25 % fewer
    tensor-core passes, one extra rounding). ``prec`` names the fp32 dot algorithm
    ("tf32x3" / "ieee"); float64 dots always run in IEEE fp64."""
    if isinstance(a, CVal) or isinstance(b, CVal):
        ar, ai = (a.re, a.im) if isinstance(a, CVal) else (a, None)
        br, bi = (b.re, b.im) if isinstance(b, CVal) else (b, None)
        if ai is None:
            return CVal(dot(ar, br, prec), dot(ar, bi, prec))
        if bi is None:
            return CVal(dot(ar, br, prec), dot(ai, br, prec))
        rr = dot(ar, br, prec)
        ii = dot(ai, bi, prec)
        if three:
            mixed = dot(ar + ai, br + bi, prec)
            return CVal(rr - ii, mixed - rr - ii)
        return CVal(rr - ii, dot(ar, bi, prec) + dot(ai, br, prec))
    rdt = jnp.result_type(a)
    precision = PREC[prec] if rdt == jnp.float32 else lax.Precision.HIGHEST
    return jnp.dot(a, b, precision=precision, preferred_element_type=rdt)


# ------------------------------------------------------------------ the field
@dataclasses.dataclass(frozen=True)
class Field:
    """Static description of the scalar field of one call (user dtype, real component
    dtype, number of components) with the typed constructors the kernels need."""

    dtype: jnp.dtype

    @staticmethod
    def of(dtype):
        return Field(jnp.dtype(dtype))

    @property
    def kind(self):
        return KINDS[self.dtype]

    @property
    def cplx(self):
        return jnp.issubdtype(self.dtype, jnp.complexfloating)

    @property
    def k(self):
        return 2 if self.cplx else 1

    @property
    def real(self):
        """The real component dtype (float32 / float64)."""
        return jnp.dtype(jnp.finfo(self.dtype).dtype)

    @property
    def regs(self):
        """Register cost of one value relative to a float32 (1, 2 or 4)."""
        return self.dtype.itemsize // 4

    def rscalar(self, x):
        """A real scalar constant of the component dtype (a NumPy scalar: lowers to an
        in-kernel constant independently of the x64 flag)."""
        return self.real.type(x)

    def zeros(self, shape):
        z = jnp.zeros(shape, self.real)
        return CVal(z, jnp.zeros(shape, self.real)) if self.cplx else z

    def eye(self, m):
        ib = lax.broadcasted_iota(jnp.int32, (m, m), 0)
        jb = lax.broadcasted_iota(jnp.int32, (m, m), 1)
        e = jnp.where(ib == jb, 1.0, 0.0).astype(self.real)
        return CVal(e, jnp.zeros((m, m), self.real)) if self.cplx else e

    def one(self):
        one = self.rscalar(1.0)
        return CVal(one, self.rscalar(0.0)) if self.cplx else one

    def split(self, A):
        """A jnp array of the field -> its parts (real arrays)."""
        if self.cplx:
            return (jnp.real(A), jnp.imag(A))
        return (A,)

    def join(self, p):
        """Parts -> a jnp array of the user dtype."""
        if self.cplx:
            return lax.complex(p[0], p[1])
        return p[0]

    def value(self, A):
        """A jnp array of the field -> a value (CVal for complex)."""
        return from_parts(self.split(A))

    def structs(self, shape):
        """Parts-shaped output spec: k ShapeDtypeStructs of the component dtype."""
        return tuple(jax.ShapeDtypeStruct(shape, self.real) for _ in range(self.k))

    def default_prec(self):
        return "tf32x3" if self.real == jnp.float32 else "ieee"

    def check_prec(self, prec):
        if prec is None:
            return self.default_prec()
        if prec not in PREC:
            raise ValueError(f"prec must be 'tf32x3' or 'ieee', got {prec!r}")
        if prec == "tf32x3" and self.real != jnp.float32:
            raise ValueError(
                "prec='tf32x3' applies to float32 / complex64 only; float64 and "
                "complex128 run their block updates in IEEE fp64 (prec='ieee')"
            )
        return prec


# ------------------------------------------------------------ ref access
def ld(refs, idx):
    """refs[idx] for the k component refs of a value."""
    return from_parts(tuple(r[idx] for r in refs))


def st(refs, idx, val):
    for r, p in zip(refs, parts(val)):
        r[idx] = p


def mld(refs, idx, mask, other=0.0):
    """Masked load through the k component refs."""
    return from_parts(
        tuple(plgpu.load(r.at[idx], mask=mask, other=other) for r in refs)
    )


def mst(refs, idx, val, mask):
    for r, p in zip(refs, parts(val)):
        plgpu.store(r.at[idx], p, mask=mask)


def read_sl(sl_ref, k):
    """(sign, logabs) from a (k + 1)-vector ref: k sign components, then logabs."""
    sign = from_parts(tuple(sl_ref[c] for c in range(k)))
    return sign, sl_ref[k]


def write_sl(sl_ref, sign, logabs):
    p = parts(sign)
    for c, v in enumerate(p):
        sl_ref[c] = v
    sl_ref[len(p)] = logabs


def init_sl(sl_ref, fld):
    """sign = 1, logabs = 0."""
    write_sl(sl_ref, fld.one(), fld.rscalar(0.0))


# ------------------------------------------------------------ pallas_call
def _regroup(flat, groups):
    out, i = [], 0
    for g in groups:
        if g == 0:
            out.append(flat[i])
            i += 1
        else:
            out.append(tuple(flat[i : i + g]))
            i += g
    return out


def _flatten(items):
    """[(array_or_parts, spec), ...] -> flat arrays, flat specs, group sizes (0 for a
    plain array, k for a parts tuple)."""
    flat, specs, groups = [], [], []
    for x, spec in items:
        if isinstance(x, tuple):
            flat.extend(x)
            specs.extend([spec] * len(x))
            groups.append(len(x))
        else:
            flat.append(x)
            specs.append(spec)
            groups.append(0)
    return flat, specs, groups


def _pcall(kernel, ins, outs, grid, *, aliases=None, num_warps, num_stages=1):
    """pallas_call over logical arguments. ``ins`` = [(array | parts, BlockSpec), ...],
    ``outs`` = [(ShapeDtypeStruct | tuple of them, BlockSpec), ...]; a parts tuple
    becomes one flat array / output per component and reaches ``kernel`` as a tuple of
    refs, a plain array as a single ref. ``aliases`` maps logical input index ->
    logical output index (the components alias pairwise). Returns the outputs regrouped
    the same way (a list)."""
    flat_in, in_specs, in_groups = _flatten(ins)
    flat_out, out_specs, out_groups = _flatten(outs)
    in_off = np.cumsum([0] + [max(g, 1) for g in in_groups])
    out_off = np.cumsum([0] + [max(g, 1) for g in out_groups])
    io_alias = {}
    for i, o in (aliases or {}).items():
        for c in range(max(in_groups[i], 1)):
            io_alias[int(in_off[i]) + c] = int(out_off[o]) + c
    nin = len(flat_in)

    def wrapped(*refs):
        kernel(*_regroup(refs[:nin], in_groups), *_regroup(refs[nin:], out_groups))

    res = pl.pallas_call(
        wrapped,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=flat_out,
        input_output_aliases=io_alias,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps, num_stages=num_stages
        ),
    )(*flat_in)
    return _regroup(list(res), out_groups)


__all__ = [
    "CVal",
    "Field",
    "PREC",
    "KINDS",
    "parts",
    "from_parts",
    "where",
    "vsum",
    "vadd",
    "abs1",
    "mag",
    "iszero",
    "recip",
    "unit",
    "conj",
    "dot",
    "ld",
    "st",
    "mld",
    "mst",
    "read_sl",
    "write_sl",
    "init_sl",
    "_pcall",
]
