"""Interaction truncation assumptions (port of ``assume.m``).

Ported: the ``'nmr'`` assumption set and the ``'zeeman'``/``'couplings'``
retention options.
"""

from __future__ import annotations

from dataclasses import replace

from spinpy.system import Assumptions, SpinSystem


def assume(spin_system: SpinSystem, assumptions: str, retention: str | None = None) -> SpinSystem:
    """Return a copy of ``spin_system`` with interaction strengths set for ``assumptions``.

    ``'nmr'``: rotating frame for all spins, secular Zeeman terms, secular
    couplings between spins of the same isotope, and weak (zz) couplings
    between different isotopes.

    ``retention='zeeman'`` drops all couplings; ``'couplings'`` drops all
    Zeeman interactions.
    """
    if any(t != "S" for t in spin_system.types):
        raise ValueError("Cavities, phonons, and transmons are not allowed under this assumption set.")
    if assumptions != "nmr":
        raise NotImplementedError(f"assumption set {assumptions!r} not ported yet; only 'nmr'")
    isotopes = spin_system.isotopes
    nspins = spin_system.nspins
    zeeman = ["secular"] * nspins
    coupling = [["secular" if isotopes[n] == isotopes[k] else "zz" for k in range(nspins)] for n in range(nspins)]
    if retention == "couplings":
        zeeman = ["ignore"] * nspins
    elif retention == "zeeman":
        coupling = [["ignore"] * nspins for _ in range(nspins)]
    elif retention is not None:
        raise ValueError("retention must be 'zeeman', 'couplings', or None.")
    return replace(
        spin_system,
        assumptions=Assumptions(
            name=assumptions, zeeman=tuple(zeeman), coupling=tuple(tuple(row) for row in coupling)
        ),
    )
