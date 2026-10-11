"""Spin operators in the Zeeman product basis.

Ports ``pauli.m``, ``irr_sph_ten.m``, ``human2opspec.m``, and the Hilbert-space
branch of ``operator.m``. As in Spinach, every operator is assembled from
single-spin irreducible spherical tensors T(l,m), which keeps one code path
for any multiplicity; an operator specification ("opspec") lists the linear
tensor index ``l*l + l - m`` for each spin, 0 meaning the unit operator.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache

import numpy as np
import scipy.sparse as sps

from spinpy.system import SpinSystem

SQRT2 = np.sqrt(2)


@cache
def pauli(mult: int) -> dict[str, sps.csr_array]:
    """Spin operators ``'u'`` (unit), ``'p'``, ``'m'``, ``'z'``, ``'x'``, ``'y'`` for a multiplicity.

    Basis order is descending projection, m = s, s-1, ..., -s.
    """
    if mult < 1:
        raise ValueError("mult must be a positive real integer.")
    s = (mult - 1) / 2
    prjs = np.arange(mult - 1, -1, -1) - s
    plus = np.sqrt(s * (s + 1) - prjs[1:] * (prjs[1:] + 1))
    minus = np.sqrt(s * (s + 1) - prjs[:-1] * (prjs[:-1] - 1))
    ops = {
        "u": sps.eye_array(mult, dtype=complex, format="csr"),
        "p": sps.csr_array(sps.diags_array(plus.astype(complex), offsets=1, shape=(mult, mult))),
        "m": sps.csr_array(sps.diags_array(minus.astype(complex), offsets=-1, shape=(mult, mult))),
        "z": sps.csr_array(sps.diags_array(prjs.astype(complex), offsets=0, shape=(mult, mult))),
    }
    ops["x"] = (ops["p"] + ops["m"]) / 2
    ops["y"] = (ops["p"] - ops["m"]) / 2j
    return ops


@cache
def irr_sph_ten(mult: int, rank: int) -> tuple[sps.csr_array, ...]:
    """Irreducible spherical tensor operators T(rank, m) for m = rank, rank-1, ..., -rank."""
    if not 0 <= rank < mult:
        raise ValueError("k must be an integer between 0 and mult-1.")
    if rank == 0:
        return (sps.eye_array(mult, dtype=complex, format="csr"),)
    ops = pauli(mult)
    tensors = [((-1) ** rank) * (2 ** (-rank / 2)) * _matrix_power(ops["p"], rank)]
    for n in range(1, 2 * rank + 1):
        q = rank - n + 1
        prev = tensors[-1]
        tensors.append((1 / np.sqrt((rank + q) * (rank - q + 1))) * (ops["m"] @ prev - prev @ ops["m"]))
    return tuple(sps.csr_array(t) for t in tensors)


def lin2lm(index):
    """Linear spherical tensor index to (l, m): index = l*l + l - m. Works on ints and arrays."""
    if isinstance(index, (int, np.integer)):
        rank = int(np.sqrt(index))
        return rank, rank * rank + rank - index
    index = np.asarray(index)
    rank = np.floor(np.sqrt(index)).astype(np.int64)
    return rank, rank * rank + rank - index


def lm2lin(rank: int, proj: int) -> int:
    if rank < 0 or abs(proj) > rank:
        raise ValueError("unacceptable projection number.")
    return rank * rank + rank - proj


def human2opspec(
    spin_system: SpinSystem, operators: str | Sequence[str], spins: str | int | Sequence[int]
) -> tuple[list[np.ndarray], np.ndarray]:
    """Parse a human-readable operator description into opspecs and coefficients.

    Three forms, as in Spinach (but with 0-based spin indices):

    * ``('Lz', '13C')``: sum of single-spin operators over all spins of that
      isotope; ``'all'``, ``'electrons'``, and ``'nuclei'`` are also accepted.
    * ``('Lz', [0, 2])``: sum over the listed spins.
    * ``(['Lz', 'L+'], [0, 1])``: product operator Lz(0) L+(1).

    Operator labels: ``'E'``, ``'Lz'``, ``'Lx'``, ``'Ly'``, ``'L+'``, ``'L-'``, ``'Tl,m'``.
    """
    if isinstance(operators, str):
        specs: list[np.ndarray] = []
        coeffs: list[np.ndarray] = []
        for n in _spin_numbers(spin_system, spins):
            sub_specs, sub_coeffs = human2opspec(spin_system, [operators], [n])
            specs += sub_specs
            coeffs.append(sub_coeffs)
        return specs, np.concatenate(coeffs)
    operators = list(operators)
    if isinstance(spins, (str, int, np.integer)):
        raise TypeError("a product operator needs a sequence of spin indices, one per operator.")
    spins = [int(n) for n in spins]
    if len(operators) != len(spins):
        raise ValueError("spins and operators lists should have the same number of elements.")
    _check_spin_indices(spin_system, spins)

    stack = np.zeros((1, spin_system.nspins), dtype=np.int64)
    coeffs_arr = np.ones(1, dtype=complex)
    for label, n in zip(operators, spins):
        if spin_system.types[n] != "S":
            raise NotImplementedError("operators on bosonic modes are not ported yet.")
        if label == "E":
            stack[:, n] = 0
        elif label == "L+":
            stack[:, n] = 1
            coeffs_arr = -SQRT2 * coeffs_arr
        elif label == "L-":
            stack[:, n] = 3
            coeffs_arr = SQRT2 * coeffs_arr
        elif label == "Lz":
            stack[:, n] = 2
        elif label in ("Lx", "Ly"):
            stack_a, stack_b = stack.copy(), stack.copy()
            stack_a[:, n], stack_b[:, n] = 1, 3
            stack = np.vstack([stack_a, stack_b])
            factors = np.array([-SQRT2, SQRT2]) / 2 if label == "Lx" else np.array([-SQRT2, -SQRT2]) / 2j
            coeffs_arr = np.kron(factors, coeffs_arr)
        elif match := re.fullmatch(r"T([+-]?\d+),([+-]?\d+)", label):
            rank, proj = int(match[1]), int(match[2])
            if rank < 0 or abs(proj) > rank or lm2lin(rank, proj) + 1 > spin_system.mults[n] ** 2:
                raise ValueError("invalid irreducible spherical tensor indices.")
            stack[:, n] = lm2lin(rank, proj)
        elif label.startswith(("CT", "ZL")):
            raise NotImplementedError(f"operator {label!r} not ported yet.")
        else:
            raise ValueError(f"unrecognized operator or state specification {label!r}.")
    return list(stack), coeffs_arr


def operator(
    spin_system: SpinSystem, operators: str | Sequence[str], spins: str | int | Sequence[int]
) -> sps.csr_array:
    """Hilbert-space operator from a human-readable description; see :func:`human2opspec`."""
    _require_hilbert(spin_system)
    opspecs, coeffs = human2opspec(spin_system, operators, spins)
    dim = int(np.prod(spin_system.mults))
    total = sps.csr_array((dim, dim), dtype=complex)
    for spec, coeff in zip(opspecs, coeffs):
        term = sps.csr_array(np.array([[coeff]], dtype=complex))
        for mult, index in zip(spin_system.mults, spec):
            rank, proj = lin2lm(int(index))
            term = sps.kron(term, irr_sph_ten(int(mult), rank)[rank - proj], format="csr")
        total = total + term
    return total


def _require_hilbert(spin_system: SpinSystem) -> None:
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if spin_system.basis.formalism != "zeeman-hilb":
        raise NotImplementedError(f"formalism {spin_system.basis.formalism!r} not ported yet.")


def _spin_numbers(spin_system: SpinSystem, spins: str | int | Sequence[int]) -> list[int]:
    if isinstance(spins, str):
        if spins == "all":
            numbers = list(range(spin_system.nspins))
        elif spins == "electrons":
            numbers = [n for n, iso in enumerate(spin_system.isotopes) if iso.startswith("E")]
        elif spins == "nuclei":
            numbers = [n for n, iso in enumerate(spin_system.isotopes) if not iso.startswith("E")]
        else:
            numbers = [n for n, iso in enumerate(spin_system.isotopes) if iso == spins]
        if not numbers:
            raise ValueError("no such spins in the system.")
        return numbers
    numbers = [int(spins)] if isinstance(spins, (int, np.integer)) else [int(n) for n in spins]
    if not numbers:
        raise ValueError("spin list cannot be empty.")
    _check_spin_indices(spin_system, numbers)
    return numbers


def _check_spin_indices(spin_system: SpinSystem, numbers: list[int]) -> None:
    if any(n < 0 or n >= spin_system.nspins for n in numbers):
        raise ValueError(f"spin indices must be between 0 and {spin_system.nspins - 1}.")
    if len(set(numbers)) != len(numbers):
        raise ValueError("spin list must not have any repetitions.")


def _matrix_power(matrix: sps.csr_array, power: int) -> sps.csr_array:
    result = matrix
    for _ in range(power - 1):
        result = result @ matrix
    return result
