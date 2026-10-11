"""Spin system description and its construction (port of ``create.m``).

Spinach keeps everything in one nested ``spin_system`` struct that functions
copy, modify, and return. spinpy keeps that functional style with frozen
dataclasses: :func:`create` builds a :class:`SpinSystem`, and later stages
(:func:`spinpy.basis`, :func:`spinpy.assume`) return modified copies.

Spin indices are 0-based throughout spinpy (Spinach uses 1-based indices).

Ported so far: isotopes, magnet, user tolerances, isotropic and tensor
Zeeman interactions, scalar and tensor couplings, and the chemical
subsystem description (``inter['chem']``: parts, concentrations, first-order
exchange rates). Every other ``sys`` or ``inter`` field raises
``NotImplementedError`` rather than being ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from spinpy.isotopes import particle_type, spin
from spinpy.tolerances import Tolerances, tolerances

FREE_ELECTRON_G = 2.00231930436092  # Spinach tols.freeg

SYS_FIELDS = {"isotopes", "magnet", "tols", "labels"}
INTER_FIELDS = {"zeeman", "coupling", "chem"}
ZEEMAN_FIELDS = {"scalar", "matrix"}
COUPLING_FIELDS = {"scalar", "matrix"}
CHEM_FIELDS = {"parts", "concs", "rates"}


@dataclass(frozen=True, eq=False)
class Basis:
    """Basis set information (Spinach ``spin_system.bas``)."""

    formalism: str
    approximation: str
    levels: np.ndarray  # (dim, nspins) Zeeman level number (1 = highest projection) of each basis state


@dataclass(frozen=True, eq=False)
class Assumptions:
    """Interaction truncation levels (Spinach ``inter.*.strength``), set by :func:`spinpy.assume`."""

    name: str
    zeeman: tuple[str, ...]                 # per spin: 'full', 'secular', or 'ignore'
    coupling: tuple[tuple[str, ...], ...]   # per spin pair: 'strong', 'secular', 'zz', 'ignore', ...


@dataclass(frozen=True, eq=False)
class Chemistry:
    """Chemical subsystems (Spinach ``spin_system.chem``).

    Every spin belongs to exactly one part; a system without ``inter['chem']``
    is one part holding all spins at unit concentration. ``rates`` is the
    first-order exchange matrix in Hz, ``dc/dt = rates @ c``: ``rates[i, j]``
    is the flow from part ``j`` into part ``i`` and every column sums to zero.
    """

    parts: tuple[tuple[int, ...], ...]   # 0-based spin indices of each part, ascending
    concs: np.ndarray                    # (nparts,) initial concentrations
    rates: np.ndarray | None             # (nparts, nparts) exchange rates, or None

    @property
    def nparts(self) -> int:
        return len(self.parts)


@dataclass(frozen=True, eq=False)
class SpinSystem:
    isotopes: tuple[str, ...]
    labels: tuple[str | None, ...]
    types: tuple[str, ...]            # 'S' for spins; 'C', 'V', 'T' for bosonic modes
    mults: np.ndarray                 # (nspins,) int multiplicities
    gammas: np.ndarray                # (nspins,) magnetogyric ratios, rad/(s*T)
    magnet: float                     # Tesla
    basefrqs: np.ndarray              # (nspins,) carrier frequencies -gamma*B0, rad/s
    zeeman_matrix: tuple[np.ndarray, ...]                        # 3x3 lab-frame Zeeman tensors, rad/s
    coupling_matrix: tuple[tuple[np.ndarray | None, ...], ...]   # 3x3 coupling tensors, rad/s; None if absent
    tols: Tolerances
    chem: Chemistry
    basis: Basis | None = None
    assumptions: Assumptions | None = None

    @property
    def nspins(self) -> int:
        return len(self.isotopes)


def create(sys: dict[str, Any], inter: dict[str, Any]) -> SpinSystem:
    """Build a spin system from Spinach-style ``sys`` and ``inter`` dictionaries.

    ``sys['isotopes']``: list of isotope names; ``sys['magnet']``: field in Tesla;
    optional ``sys['tols']``: tolerance overrides; optional ``sys['labels']``.

    ``inter['zeeman']['scalar']``: per-spin chemical shift in ppm (nuclei) or
    isotropic g-value (electrons), ``None`` for none. ``inter['zeeman']['matrix']``:
    per-spin 3x3 tensors in the same units. ``inter['coupling']['scalar']``:
    nspins x nspins nested list of couplings in Hz (``None`` or 0 for none);
    ``inter['coupling']['matrix']``: same layout with 3x3 tensors in Hz.
    """
    _reject_unported(sys, SYS_FIELDS, "sys")
    _reject_unported(inter, INTER_FIELDS, "inter")
    isotopes = tuple(sys["isotopes"])
    nspins = len(isotopes)
    if nspins == 0:
        raise ValueError("sys['isotopes'] must not be empty.")
    tols = tolerances(sys.get("tols"))
    labels = tuple(sys.get("labels", [None] * nspins))
    if len(labels) != nspins:
        raise ValueError("sys['labels'] must have one entry per spin.")

    gammas = np.empty(nspins)
    mults = np.empty(nspins, dtype=np.int64)
    for n, name in enumerate(isotopes):
        gammas[n], mults[n] = spin(name)
    magnet = float(sys["magnet"])
    basefrqs = -gammas * magnet

    zeeman = _zeeman_tensors(isotopes, basefrqs, inter.get("zeeman"))
    coupling = _coupling_tensors(mults, tols, inter.get("coupling"))
    chem = _chemistry(isotopes, inter.get("chem"))
    _check_no_couplings_between_parts(chem, coupling)

    return SpinSystem(
        isotopes=isotopes,
        labels=labels,
        types=tuple(particle_type(name) for name in isotopes),
        mults=mults,
        gammas=gammas,
        magnet=magnet,
        basefrqs=basefrqs,
        zeeman_matrix=zeeman,
        coupling_matrix=coupling,
        tols=tols,
        chem=chem,
    )


def _chemistry(isotopes: tuple[str, ...], chem: dict[str, Any] | None) -> Chemistry:
    """Chemical subsystems, validated as in the chemistry section of create.m."""
    nspins = len(isotopes)
    if chem is None:
        return Chemistry(parts=(tuple(range(nspins)),), concs=np.ones(1), rates=None)
    _reject_unported(chem, CHEM_FIELDS, "inter['chem']")
    if "rates" in chem and "parts" not in chem:
        raise ValueError("subsystem identifiers (inter['chem']['parts']) must be provided.")
    if "rates" in chem and "concs" not in chem:
        raise ValueError("initial concentrations (inter['chem']['concs']) must be provided.")

    if "parts" in chem:
        parts = tuple(tuple(sorted(int(n) for n in part)) for part in chem["parts"])
        for part in parts:
            if not part or len(set(part)) != len(part) or min(part) < 0 or max(part) >= nspins:
                raise ValueError(
                    f"elements of inter['chem']['parts'] must be non-empty lists of unique spin indices between 0 and {nspins - 1}."
                )
        assigned = [n for part in parts for n in part]
        if len(set(assigned)) != len(assigned):
            raise ValueError("a given spin can only belong to one chemical subsystem in inter['chem']['parts'].")
        if set(assigned) != set(range(nspins)):
            raise ValueError("every spin must belong to one of the chemical subsystems in inter['chem']['parts'].")
    else:
        parts = (tuple(range(nspins)),)

    if "concs" in chem:
        concs = np.atleast_1d(np.asarray(chem["concs"], dtype=float))
        if concs.ndim != 1 or np.any(concs < 0):
            raise ValueError("inter['chem']['concs'] must be a vector of non-negative real numbers.")
        if concs.size != len(parts):
            raise ValueError("the number of initial concentrations must be equal to the number of chemical species.")
    elif len(parts) == 1:
        concs = np.ones(1)
    else:
        raise ValueError("concentrations (inter['chem']['concs']) must be provided.")

    rates = None
    if "rates" in chem:
        rates = np.asarray(chem["rates"], dtype=float)
        if rates.ndim != 2 or rates.shape[0] != rates.shape[1]:
            raise ValueError("inter['chem']['rates'] must be a real square matrix.")
        if rates.shape[0] != len(parts):
            raise ValueError("both dimensions of inter['chem']['rates'] must be equal to the number of chemical subsystems.")
        if np.any(np.abs(rates.sum(axis=0)) > 10 * np.finfo(float).eps * np.linalg.norm(rates, 1)):
            raise ValueError("inter['chem']['rates'] violates conservation of matter: column sums must be negligible.")
        # First-order exchange moves spin states between parts, so the parts must match spin for spin
        sequences = {tuple(isotopes[n] for n in part) for part in parts}
        if len({len(part) for part in parts}) != 1:
            raise ValueError("exchange mode: all chemical subsystems must have the same number of spins.")
        if len(sequences) != 1:
            raise ValueError("exchange mode: isotope sequences in all chemical subsystems must be the same.")
    return Chemistry(parts=parts, concs=concs, rates=rates)


def _check_no_couplings_between_parts(
    chem: Chemistry, coupling: tuple[tuple[np.ndarray | None, ...], ...]
) -> None:
    for a, part_a in enumerate(chem.parts):
        for b, part_b in enumerate(chem.parts):
            if a != b and any(coupling[n][k] is not None for n in part_a for k in part_b):
                raise ValueError("couplings detected between spins in different chemical species.")


def _reject_unported(spec: dict[str, Any], supported: set[str], name: str) -> None:
    unported = set(spec) - supported
    if unported:
        raise NotImplementedError(
            f"{name} field(s) {sorted(unported)} not ported yet; supported: {sorted(supported)}"
        )


def _zeeman_tensors(
    isotopes: tuple[str, ...], basefrqs: np.ndarray, zeeman: dict[str, Any] | None
) -> tuple[np.ndarray, ...]:
    """Lab-frame Zeeman tensors in rad/s, as in the Zeeman section of create.m."""
    nspins = len(isotopes)
    if zeeman is None:
        return tuple(np.eye(3) * basefrqs[n] for n in range(nspins))
    _reject_unported(zeeman, ZEEMAN_FIELDS, "inter['zeeman']")
    tensors = [np.zeros((3, 3)) for _ in range(nspins)]
    if "matrix" in zeeman:
        _check_length(zeeman["matrix"], nspins, "inter['zeeman']['matrix']")
        for n, user in enumerate(zeeman["matrix"]):
            if user is not None and np.linalg.norm(np.asarray(user, dtype=float), 2) > 0:
                tensors[n] = tensors[n] + np.asarray(user, dtype=float)
    if "scalar" in zeeman:
        _check_length(zeeman["scalar"], nspins, "inter['zeeman']['scalar']")
        for n, user in enumerate(zeeman["scalar"]):
            if user is not None and abs(user) > 0:
                tensors[n] = tensors[n] + np.eye(3) * float(user)
    out = []
    for n, name in enumerate(isotopes):
        if name[0] == "E":
            # Electrons: g-tensor in Bohr magneton units
            out.append(tensors[n] * basefrqs[n] / FREE_ELECTRON_G)
        else:
            # Nuclei: chemical shielding in ppm
            out.append((np.eye(3) + 1e-6 * tensors[n]) * basefrqs[n])
    return tuple(out)


def _coupling_tensors(
    mults: np.ndarray, tols: Tolerances, coupling: dict[str, Any] | None
) -> tuple[tuple[np.ndarray | None, ...], ...]:
    """Coupling tensors in rad/s, as in the coupling section of create.m."""
    nspins = len(mults)
    tensors = [[np.zeros((3, 3)) for _ in range(nspins)] for _ in range(nspins)]
    if coupling is not None:
        _reject_unported(coupling, COUPLING_FIELDS, "inter['coupling']")
        if "matrix" in coupling:
            user = _check_square(coupling["matrix"], nspins, "inter['coupling']['matrix']")
            for n in range(nspins):
                for k in range(nspins):
                    value = user[n][k]
                    if value is not None and np.linalg.norm(np.asarray(value, dtype=float), 2) > tols.inter_cutoff:
                        tensors[n][k] = tensors[n][k] + 2 * np.pi * np.asarray(value, dtype=float)
        if "scalar" in coupling:
            user = _check_square(coupling["scalar"], nspins, "inter['coupling']['scalar']")
            for n in range(nspins):
                for k in range(nspins):
                    value = user[n][k]
                    if value is not None and abs(value) > tols.inter_cutoff:
                        tensors[n][k] = tensors[n][k] + 2 * np.pi * np.eye(3) * float(value)
    # Drop insignificant tensors and couplings to spin-zero particles
    return tuple(
        tuple(
            None
            if (np.linalg.norm(tensors[n][k], 2) < 2 * np.pi * tols.inter_cutoff or mults[n] == 1 or mults[k] == 1)
            else tensors[n][k]
            for k in range(nspins)
        )
        for n in range(nspins)
    )


def _check_length(values: Sequence[Any], nspins: int, name: str) -> None:
    if len(values) != nspins:
        raise ValueError(f"{name} must have one entry per spin ({nspins}).")


def _check_square(values: Sequence[Sequence[Any]], nspins: int, name: str) -> Sequence[Sequence[Any]]:
    if len(values) != nspins or any(len(row) != nspins for row in values):
        raise ValueError(f"{name} must be a {nspins} x {nspins} nested list.")
    return values
