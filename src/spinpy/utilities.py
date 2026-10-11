"""Small numerical helpers shared across the kernel (ports of ``clean_up.m``, ``hdot.m``)."""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sps

from spinpy.tolerances import Tolerances

Matrix = Union[np.ndarray, sps.csr_array]


def nnz(matrix: Matrix) -> int:
    return int(matrix.count_nonzero()) if sps.issparse(matrix) else int(np.count_nonzero(matrix))
    # GPU: count_nonzero exists for cupyx sparse matrices and cupy arrays; route through the backend then


def matlab_round(values: np.ndarray) -> np.ndarray:
    """Round half away from zero, separately on real and imaginary parts, as MATLAB ``round`` does."""
    if np.iscomplexobj(values):
        return matlab_round(values.real) + 1j * matlab_round(values.imag)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def clean_up(tols: Tolerances, matrix: Matrix, nonzero_tol: float) -> Matrix:
    """Round elements to a grid of ``nonzero_tol`` and choose sparse or dense storage.

    Rounding removes elements smaller than half the tolerance. Matrices with
    any dimension below ``tols.small_matrix``, or with density above
    ``tols.dense_matrix``, are returned dense (``np.ndarray``); large matrices
    below that density are returned sparse (``csr_array``).
    """
    if nonzero_tol == 0 or np.isnan(nonzero_tol):
        return matrix
    if nonzero_tol < 0:
        raise ValueError("nonzero_tol parameter must be a positive real number.")
    # GPU: this is called inside the propagator Taylor loop on device arrays; the rounding
    # works through the array module and the sparse/dense switch through the backend's csr()
    scale = 1 / nonzero_tol
    if sps.issparse(matrix):
        matrix = sps.csr_array(matrix, copy=True)
        matrix.data = nonzero_tol * matlab_round(scale * matrix.data)
        matrix.eliminate_zeros()
    else:
        matrix = nonzero_tol * matlab_round(scale * np.asarray(matrix))
    density = nnz(matrix) / (matrix.shape[0] * matrix.shape[1])
    if sps.issparse(matrix):
        if (nnz(matrix) > 0 and min(matrix.shape) < tols.small_matrix) or density > tols.dense_matrix:
            return matrix.toarray()
        return matrix
    if density < tols.dense_matrix and min(matrix.shape) > tols.small_matrix:
        return sps.csr_array(matrix)
    return matrix


def hdot(a: Matrix, b: Matrix) -> complex:
    """Hilbert-Schmidt inner product ``sum(conj(a) * b)``."""
    if a.shape != b.shape:
        raise ValueError("the two inputs must have identical dimensions.")
    if sps.issparse(a):
        return complex(a.conj().multiply(b).sum())
    if sps.issparse(b):
        return complex(b.multiply(np.conj(a)).sum())
    return complex(np.sum(np.conj(a) * b))


def dagger(matrix: Matrix) -> Matrix:
    return matrix.conj().T
