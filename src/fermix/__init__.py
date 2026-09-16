"""fermix: fast batched fp32 determinants and Pfaffians on GPU (JAX / Pallas), with
singular-safe gradients.

Public functions: :func:`slogdet`, :func:`slogpf`, :func:`det`, :func:`pf`.
"""

from .api import FermixFallbackWarning, det, pf, slogdet, slogpf

__version__ = "0.1.0"
__all__ = ["slogdet", "slogpf", "det", "pf", "FermixFallbackWarning", "__version__"]
