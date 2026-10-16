"""Normwise comparison of spynach results against oracle fixtures.

The acceptance test is ``||actual - expected||_F <= atol + rtol * ||expected||_F``.
A normwise test is used rather than an elementwise one because propagators
and Hamiltonians contain many elements near zero, where relative elementwise
errors are meaningless. Shapes must match exactly; dense and sparse inputs
may be mixed. Tolerances are always explicit: what is acceptable depends on
the stage and on the working precision, and is decided by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sps


@dataclass(frozen=True)
class Comparison:
    name: str
    shape: tuple[int, ...]
    err_norm: float
    ref_norm: float
    max_abs_err: float
    rtol: float
    atol: float

    @property
    def rel_err(self) -> float:
        if self.ref_norm > 0:
            return self.err_norm / self.ref_norm
        return 0.0 if self.err_norm == 0 else float("inf")

    @property
    def passed(self) -> bool:
        return bool(self.err_norm <= self.atol + self.rtol * self.ref_norm)

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} {self.name or '<unnamed>'} {self.shape}: "
            f"rel_err={self.rel_err:.3e} max_abs_err={self.max_abs_err:.3e} "
            f"ref_norm={self.ref_norm:.3e} (rtol={self.rtol:g}, atol={self.atol:g})"
        )


def _as_numeric(x: Any) -> np.ndarray | sps.sparray:
    if sps.issparse(x):
        return sps.csr_array(x)
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError("cell (object) arrays must be compared element by element")
    return arr


def compare(actual: Any, expected: Any, *, rtol: float, atol: float, name: str = "") -> Comparison:
    """Compare two arrays normwise; NaN or Inf anywhere makes the comparison fail."""
    a, e = _as_numeric(actual), _as_numeric(expected)
    if tuple(a.shape) != tuple(e.shape):
        raise ValueError(f"{name or 'comparison'}: shape {a.shape} != expected {e.shape}")
    if sps.issparse(a) and sps.issparse(e):
        diff = (a - e).data
        ref = e.data
    else:
        a_dense = a.toarray() if sps.issparse(a) else a
        e_dense = e.toarray() if sps.issparse(e) else e
        diff = (a_dense - e_dense).ravel()
        ref = e_dense.ravel()
    if diff.size and not (np.all(np.isfinite(diff)) and np.all(np.isfinite(ref))):
        err_norm = max_abs = float("inf")
    else:
        err_norm = float(np.linalg.norm(diff)) if diff.size else 0.0
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    ref_norm = float(np.linalg.norm(ref)) if ref.size else 0.0
    return Comparison(
        name=name,
        shape=tuple(e.shape),
        err_norm=err_norm,
        ref_norm=ref_norm,
        max_abs_err=max_abs,
        rtol=rtol,
        atol=atol,
    )


def assert_matches(actual: Any, expected: Any, *, rtol: float, atol: float, name: str = "") -> Comparison:
    """Raise AssertionError with a one-line report unless ``compare`` passes."""
    result = compare(actual, expected, rtol=rtol, atol=atol, name=name)
    if not result.passed:
        raise AssertionError(str(result))
    return result
