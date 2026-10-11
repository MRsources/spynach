"""Relaxation superoperators and thermal equilibrium (ports of ``relaxation.m``, ``equilibrium.m``, ``thermalize.m``).

Ported theories (``inter['relaxation']``):

* ``'redfield'``: Bloch-Redfield-Wangsness theory for isotropic,
  axial or rhombic rotational diffusion, in the diagonalisation-free form
  of Kuprov (2011): every pair of irreducible components of the
  anisotropic Hamiltonian contributes ``-w A int_0^T exp(iL0 t) C exp(-iD t) dt``,
  evaluated as a block of a single matrix exponential (auxiliary matrix
  formalism of Goodwin and Kuprov, 2015, :func:`expmint`);
* ``'t1_t2'``: the extended T1/T2 model, a diagonal superoperator with
  per-spin longitudinal and transverse rates summed over the spins each
  basis state correlates;
* ``'damp'``: non-selective damping of every state except the unit state.

Then the retention policy ``rlx_keep`` (``'diagonal'``, ``'kite'``,
``'secular'``, ``'labframe'``), the dynamic frequency shift policy, and the
thermalisation ``equilibrium`` (``'zero'``, ``'IME'``, ``'dibari'``) are
applied. Everything is ``sphten-liouv`` only, except ``'damp'`` and
:func:`equilibrium`, which also work in Hilbert space.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps

from spinpy.assume import assume
from spinpy.conventions import euler2dcm
from spinpy.hamiltonian import hamiltonian, orientation
from spinpy.operators import lin2lm
from spinpy.propagation import propagator, step
from spinpy.system import SpinSystem
from spinpy.utilities import Matrix, clean_up

log = logging.getLogger("spinpy")


def relaxation(spin_system: SpinSystem, euler_angles: Sequence[float] | None = None) -> Matrix:
    """Relaxation superoperator ``R`` (Hz) for the theories in ``inter['relaxation']``.

    ``euler_angles`` select the orientation for anisotropic rate tensors
    and for the thermal equilibrium of anisotropic systems. The result is
    a dissipative generator: the Liouvillian is ``H + 1j * R + 1j * K``.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    rlx = spin_system.rlx
    formalism = spin_system.basis.formalism
    dim = spin_system.basis.nstates if formalism == "sphten-liouv" else int(np.prod(spin_system.mults))
    R: Matrix = sps.csr_array((dim, dim), dtype=complex)
    if not rlx.theories:
        return R
    if formalism != "sphten-liouv" and set(rlx.theories) != {"damp"}:
        raise NotImplementedError("Redfield and T1/T2 relaxation theories are only available in the sphten-liouv formalism.")

    if "t1_t2" in rlx.theories:
        R = R + _t1_t2(spin_system, euler_angles)
    if "redfield" in rlx.theories:
        R = R + _redfield(spin_system)
        max_rate = np.abs(_diag(R)).max() if dim else 0.0
        for tau in rlx.tau_c:
            if max_rate > 0 and 1 / max_rate < tau.max():
                raise ValueError("T1,2>>tau_c validity condition violation in Redfield theory")

    if rlx.dfs == "ignore":
        R = R.real if sps.issparse(R) else np.real(R)
    R = _keep(spin_system, R)

    if "damp" in rlx.theories:
        if euler_angles is not None:
            ort = np.array([0, 0, 1.0]) @ euler2dcm(euler_angles)
            rate = float(ort @ np.atleast_2d(rlx.damp_rate) @ ort) if rlx.damp_rate.ndim == 2 else float(rlx.damp_rate)
        else:
            rate = float(np.mean(np.diag(rlx.damp_rate))) if rlx.damp_rate.ndim == 2 else float(rlx.damp_rate)
            if rlx.damp_rate.ndim == 2:
                log.warning("damping rate anisotropy ignored")
        if formalism == "zeeman-hilb":
            R = R - (rate / 2) * sps.eye_array(dim, dtype=complex, format="csr")
        else:
            # Liouville space: every state but the unit state is damped
            damping = -rate * np.ones(dim)
            damping[0] = 0
            R = R + sps.diags_array(damping.astype(complex), format="csr")

    if rlx.equilibrium == "IME":
        if formalism != "sphten-liouv":
            raise NotImplementedError("IME thermalisation is only available in Liouville space.")
        H_left = _lab_frame_left(spin_system, euler_angles)
        R = thermalize(spin_system, R, None, None, equilibrium(spin_system, H_left), "IME")
    elif rlx.equilibrium == "dibari":
        if rlx.temperature == 0:
            raise ValueError("DiBari-Levitt thermalization cannot be used with the high-temperature approximation.")
        R = thermalize(spin_system, R, _lab_frame_left(spin_system, euler_angles), rlx.temperature, None, "dibari")
    return _complex(clean_up(spin_system.tols, R, spin_system.tols.rlx_zero))


def equilibrium(
    spin_system: SpinSystem, I: Matrix | None = None, Q: list | None = None, euler_angles: Sequence[float] | None = None
) -> np.ndarray | Matrix:
    """Thermal equilibrium state at ``inter['temperature']`` (port of ``equilibrium.m``).

    ``I`` is the lab-frame Hamiltonian as a *left-side product*
    superoperator in Liouville space (``hamiltonian(assume(ss, 'labframe'),
    'left')``) or the Hamiltonian matrix in Hilbert space; with ``Q`` and
    ``euler_angles`` the anisotropic part at that orientation is added.
    Without ``I`` the isotropic lab-frame Hamiltonian is built here. The
    Liouville-space result has unit coefficient on the unit state; the
    Hilbert-space density matrix has unit trace.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    temperature = spin_system.rlx.temperature
    if temperature == 0:
        raise ValueError("absolute zero temperature is not supported.")
    formalism = spin_system.basis.formalism
    if I is None:
        I = hamiltonian(assume(spin_system, "labframe"), "left" if formalism == "sphten-liouv" else "comm")
    if Q is not None:
        if euler_angles is None:
            raise ValueError("euler_angles must be given with Q.")
        I = I + orientation(Q, euler_angles)
    beta = spin_system.tols.hbar / (spin_system.tols.kbol * temperature)
    if formalism == "sphten-liouv":
        unit = np.zeros(I.shape[1], dtype=complex)
        unit[0] = 1
        if np.abs(I @ unit).sum() < 1e-10:
            raise ValueError("I and Q must be left side product superoperators, not commutation superoperators.")
        rho = step(spin_system, I, unit, -1j * beta)
        if np.any(np.isnan(rho)):
            raise ValueError("numerical accuracy issues, temperature too low - switch to Hilbert space.")
        return rho / rho[0]
    if formalism != "zeeman-hilb":
        raise NotImplementedError(f"formalism {formalism!r} not ported yet.")
    norm = float(abs(beta) * abs(I).sum(axis=0).max())
    n_squarings = max(0, int(np.ceil(np.log2(norm)))) if norm > 0 else 0
    rho = propagator(spin_system, I, -1j * beta / 2**n_squarings)
    rho = rho / _trace(rho)
    for _ in range(n_squarings):
        rho = rho @ rho
        rho = clean_up(spin_system.tols, rho / _trace(rho), spin_system.tols.prop_chop)
    return rho


def thermalize(
    spin_system: SpinSystem, R: Matrix, HLSPS: Matrix | None, T: float | None, rho_eq: np.ndarray | None, method: str
) -> Matrix:
    """Make ``rho_eq`` the stationary state of ``R`` (port of ``thermalize.m``).

    ``'IME'`` (inhomogeneous master equation): ``R -> R - (R rho_eq) u^T``
    with ``u`` the unit state, so that ``R rho_eq = 0``. ``'dibari'``
    (DiBari-Levitt): ``R -> R expm(-H beta)`` with ``H`` the left-side
    product superoperator of the lab-frame Hamiltonian ``HLSPS`` and
    ``beta = hbar / (k T)``.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("thermalize() is only ported for the sphten-liouv formalism.")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix.")
    dim = R.shape[0]
    unit = np.zeros(dim, dtype=complex)
    unit[0] = 1
    if np.linalg.norm(R @ unit) > 1e-10:
        raise ValueError("R appears to be thermalized already.")
    if method == "IME":
        if rho_eq is None:
            raise ValueError("rho_eq cannot be empty for IME formalism.")
        drive = np.asarray(R @ np.asarray(rho_eq, dtype=complex)).reshape(-1)
        correction = sps.csr_array((drive, (np.arange(dim), np.zeros(dim, dtype=int))), shape=(dim, dim))
        return R - correction
    if method == "dibari":
        if HLSPS is None or T is None:
            raise ValueError("HLSPS and T cannot be empty for DiBari-Levitt formalism.")
        if np.abs(HLSPS @ unit).sum() < 1e-8:
            raise ValueError("HLSPS appears to be a commutation superoperator.")
        beta = spin_system.tols.hbar / (spin_system.tols.kbol * T)
        return R @ propagator(spin_system, HLSPS, 1j * beta)
    raise ValueError("method must be 'IME' or 'dibari'.")


def expmint(spin_system: SpinSystem, A: Matrix, B: Matrix, C: Matrix, T: float) -> Matrix:
    """``int_0^T exp(-iA t) B exp(-iC t) dt`` evaluated as a block of one exponential (port of ``expmint.m``).

    The auxiliary matrix ``[[-A, iB], [0, -C]]`` is exponentiated with
    :func:`propagator` over ``T``; the result is ``P^H Q`` with ``P`` the
    upper-left and ``Q`` the upper-right block.
    """
    dim = B.shape[0]
    if T == 0 or _nnz(B) == 0:
        return sps.csr_array((dim, dim), dtype=complex)
    aux = sps.block_array([[-sps.csr_array(A), 1j * sps.csr_array(B)], [None, -sps.csr_array(C)]], format="csr")
    aux = propagator(spin_system, aux, T)
    aux = sps.csr_array(aux) if sps.issparse(aux) else np.asarray(aux)
    P = aux[:dim, :dim].conj().T
    Q = aux[:dim, dim:]
    return clean_up(spin_system.tols, P @ Q, spin_system.tols.prop_chop)


def _redfield(spin_system: SpinSystem) -> Matrix:
    """The BRW relaxation integral (``redfield_integral_serial.m``) for rotational diffusion."""
    tols = spin_system.tols
    labframe = assume(spin_system, "labframe")
    L0, Q = hamiltonian(labframe, "comm", aniso=True)
    dim = L0.shape[0]
    R: Matrix = sps.csr_array((dim, dim), dtype=complex)
    basis = spin_system.basis.basis
    species_states = [basis[:, list(part)].any(axis=1) for part in spin_system.chem.parts]
    for rank, Q_rank in enumerate(Q, start=1):
        size = 2 * rank + 1
        for k in range(size):
            for m in range(size):
                if _nnz(Q_rank[k][m]) == 0:
                    continue
                for p in range(size):
                    for q in range(size):
                        if _nnz(Q_rank[p][q]) == 0:
                            continue
                        for s, (weights, rates) in enumerate(_corrfun(spin_system, rank, k, m, p, q)):
                            mask = species_states[s]
                            for weight, rate in zip(weights, rates):
                                if abs(weight) == 0:
                                    continue
                                upper_limit = -1.5 * (1 / rate) * np.log(1 / tols.rlx_integration)
                                B = clean_up(tols, L0, tols.rlx_integration / abs(upper_limit))
                                A = Q_rank[k][m]
                                C = Q_rank[p][q].conj().T
                                D = B - 1j * rate * sps.eye_array(dim, dtype=complex, format="csr")
                                A, B, C, D = (_restrict(X, mask) for X in (A, B, C, D))
                                R_int = -weight * (A @ expmint(spin_system, B, C, D, upper_limit))
                                R = R + clean_up(tols, R_int, 1e-2 * tols.rlx_zero)
    return R


def _corrfun(spin_system: SpinSystem, rank: int, k: int, m: int, p: int, q: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Weights and decay rates of the rotational diffusion correlation functions per chemical subsystem (``corrfun.m``).

    Indices ``k, m, p, q`` are 0-based positions in the ``(2*rank + 1)``
    component lists (``+rank`` first).
    """
    out = []
    for tau in spin_system.rlx.tau_c:
        if tau.size == 1:
            D = 1 / (6 * tau[0])
            weights = np.array([(1 / (2 * rank + 1)) * (k == p) * (m == q)], dtype=float)
            rates = np.array([-rank * (rank + 1) * D])
        elif tau.size == 2:
            D_ax, D_eq = 1 / (6 * tau[0]), 1 / (6 * tau[1])
            weights = np.array([(1 / (2 * rank + 1)) * (k == p) * (m == q)], dtype=float)
            proj = rank - m          # projection quantum number of component m
            rates = np.array([-(rank * (rank + 1) * D_eq + proj**2 * (D_ax - D_eq))])
        else:
            Dxx, Dyy, Dzz = 1 / (6 * tau)
            mean = np.mean([Dxx, Dyy, Dzz])
            if min(abs(Dxx - Dyy), abs(Dyy - Dzz), abs(Dzz - Dxx)) < 1e-6 * mean:
                raise ValueError("the three rotational correlation times must be different.")
            if rank != 2:
                raise ValueError("rhombic rotational diffusion only implemented for rank 2.")
            delta = np.sqrt(Dxx**2 + Dyy**2 + Dzz**2 - Dxx * Dyy - Dxx * Dzz - Dyy * Dzz)
            rates = -np.array([
                4 * Dxx + Dyy + Dzz, Dxx + 4 * Dyy + Dzz, Dxx + Dyy + 4 * Dzz,
                2 * Dxx + 2 * Dyy + 2 * Dzz - 2 * delta, 2 * Dxx + 2 * Dyy + 2 * Dzz + 2 * delta,
            ])
            lam_p = np.sqrt(2 / 3) * (Dxx + Dyy - 2 * Dzz + 2 * delta) / (Dxx - Dyy)
            lam_m = np.sqrt(2 / 3) * (Dxx + Dyy - 2 * Dzz - 2 * delta) / (Dxx - Dyy)
            h = np.zeros((5, 5))
            h[0, 1] = h[0, 3] = 1 / np.sqrt(2)
            h[1, 1], h[1, 3] = -1 / np.sqrt(2), 1 / np.sqrt(2)
            h[2, 4], h[2, 0] = 1 / np.sqrt(2), -1 / np.sqrt(2)
            h[3, 0] = h[3, 4] = 1 / np.sqrt(2 + lam_m**2)
            h[3, 2] = lam_m / np.sqrt(2 + lam_m**2)
            h[4, 0] = h[4, 4] = 1 / np.sqrt(2 + lam_p**2)
            h[4, 2] = lam_p / np.sqrt(2 + lam_p**2)
            weights = np.array([(1 / 5) * (k == p) * h[j, m] * h[j, q] for j in range(5)])
        out.append((weights, rates))
    return out


def _t1_t2(spin_system: SpinSystem, euler_angles: Sequence[float] | None) -> sps.csr_array:
    """Extended T1/T2 model: ``-diag(sum of R1 over longitudinal spins + sum of R2 over transverse spins)``."""
    rlx = spin_system.rlx
    rates = []
    for tensors in (rlx.r1_rates, rlx.r2_rates):
        values = np.zeros(spin_system.nspins)
        for n, rate in enumerate(tensors):
            if rate.ndim == 2:
                if euler_angles is None:
                    raise ValueError("Euler angles must be specified with anisotropic T1/T2 relaxation theory.")
                ort = np.array([0, 0, 1.0]) @ euler2dcm(euler_angles)
                values[n] = ort @ rate @ ort
            else:
                values[n] = rate
        rates.append(values)
    ranks, projs = lin2lm(spin_system.basis.basis)
    longitudinal = (ranks != 0) & (projs == 0)
    transverse = projs != 0
    diagonal = longitudinal @ rates[0] + transverse @ rates[1]
    return sps.diags_array((-diagonal).astype(complex), format="csr")


def _keep(spin_system: SpinSystem, R: Matrix) -> Matrix:
    """Retention policy ``rlx_keep``."""
    keep = spin_system.rlx.keep
    if keep == "labframe":
        return R
    dim = R.shape[0]
    if keep == "diagonal":
        diagonal = _diag(R).copy()
        diagonal[0] = 0          # the unit state is never relaxed
        return sps.diags_array(diagonal.astype(complex), format="csr")
    rows, cols, vals = _find(R)
    _, projs = lin2lm(spin_system.basis.basis)
    if keep == "kite":
        longitudinal = np.flatnonzero(np.abs(projs).sum(axis=1) == 0)
        mask = (np.isin(rows, longitudinal) & np.isin(cols, longitudinal)) | (rows == cols)
    elif keep == "secular":
        frequencies = projs @ spin_system.basefrqs
        mask = frequencies[rows] == frequencies[cols]
    else:
        raise ValueError("invalid value of the inter.rlx_keep parameter.")
    return sps.csr_array((vals[mask], (rows[mask], cols[mask])), shape=(dim, dim))


def _lab_frame_left(spin_system: SpinSystem, euler_angles: Sequence[float] | None) -> Matrix:
    labframe = assume(spin_system, "labframe")
    if euler_angles is None:
        return hamiltonian(labframe, "left")
    H, Q = hamiltonian(labframe, "left", aniso=True)
    return H + orientation(Q, euler_angles)


def _restrict(X: Matrix, mask: np.ndarray) -> Matrix:
    """Zero the rows and columns outside ``mask`` (states of other chemical subsystems)."""
    if mask.all():
        return X
    keep = sps.diags_array(mask.astype(float), format="csr")
    return keep @ X @ keep


def _diag(R: Matrix) -> np.ndarray:
    return np.asarray(R.diagonal()).reshape(-1)


def _find(R: Matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = sps.coo_array(R)
    return coo.row, coo.col, coo.data


def _nnz(X: Matrix) -> int:
    return int(X.count_nonzero()) if sps.issparse(X) else int(np.count_nonzero(X))


def _trace(rho: Matrix) -> complex:
    return complex(rho.trace()) if sps.issparse(rho) else complex(np.trace(rho))


def _complex(R: Matrix) -> Matrix:
    return R.astype(complex) if sps.issparse(R) else np.asarray(R, dtype=complex)
