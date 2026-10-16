"""Fixture-free tests of basis(): state counts, order, filters, chemistry, and refusals."""

from math import comb

import numpy as np
import pytest

import respinach
from respinach.basis import dfpt, prune_subgraphs

NONE = {"formalism": "sphten-liouv", "approximation": "none"}


def system(isotopes, **inter):
    return respinach.create({"isotopes": isotopes, "magnet": 14.1}, inter)


def chain(nspins, J=10.0):
    scalar = np.zeros((nspins, nspins))
    for n in range(nspins - 1):
        scalar[n, n + 1] = J
    return system(["1H"] * nspins, coupling={"scalar": scalar.tolist()})


def test_complete_basis_is_sorted_with_unit_state_first():
    spin_system = respinach.basis(system(["1H", "2H"]), NONE)
    bas = spin_system.basis
    assert bas.nstates == 4 * 9
    assert bas.basis[0].tolist() == [0, 0]
    assert bas.basis.tolist() == sorted(bas.basis.tolist())
    assert bas.basis[1].tolist() == [0, 1]
    # (l, m) labels: row [1, 5] is T(1,+1) x T(2,+1): coherence order 2, two spins correlated
    row = np.flatnonzero((bas.basis == [1, 5]).all(axis=1))[0]
    assert bas.tot_proj[row] == 2 and bas.tot_cord[row] == 2


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_ik0_state_count(level):
    spin_system = respinach.basis(system(["1H"] * 4), {**NONE, "approximation": "IK-0", "inter_level": level})
    assert spin_system.basis.nstates == sum(comb(4, j) * 3**j for j in range(level + 1))
    assert spin_system.basis.tot_cord.max() == level


def test_ik1_and_ik2_on_a_chain():
    spin_system = chain(6)
    spin_system_ik2 = respinach.basis(
        spin_system, {**NONE, "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 1}
    )
    spin_system_ik1 = respinach.basis(
        spin_system,
        {**NONE, "approximation": "IK-1", "connectivity": "scalar_couplings", "inter_level": 3, "prox_level": 1},
    )
    # States supported on up to three consecutive spins: 1 + 6*3 + 9*9 + 4*27
    assert spin_system_ik2.basis.nstates == spin_system_ik1.basis.nstates == 208
    spin_system_ik1_2 = respinach.basis(
        spin_system,
        {**NONE, "approximation": "IK-1", "connectivity": "scalar_couplings", "inter_level": 2, "prox_level": 1},
    )
    assert spin_system_ik1_2.basis.nstates == 1 + 6 * 3 + 5 * 9
    # Uncoupled spins only get single-spin states under IK-2
    spin_system_isolated = respinach.basis(
        system(["1H"] * 3), {**NONE, "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 2}
    )
    assert spin_system_isolated.basis.nstates == 1 + 3 * 3


def test_full_tensor_connectivity_sees_traceless_couplings():
    tensor = np.diag([1.0, 1.0, -2.0]).tolist()
    spin_system = system(["1H", "1H"], coupling={"matrix": [[None, tensor], [None, None]]})
    spin_system_scalar = respinach.basis(
        spin_system, {**NONE, "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 1}
    )
    spin_system_full = respinach.basis(
        spin_system, {**NONE, "approximation": "IK-2", "connectivity": "full_tensors", "prox_level": 1}
    )
    assert spin_system_scalar.basis.nstates == 7 and spin_system_full.basis.nstates == 16


def test_filters():
    spin_system = system(["1H", "13C", "1H"])
    spin_system_plus = respinach.basis(spin_system, {**NONE, "projections": [[1]]})
    assert spin_system_plus.basis.tot_proj[0] == 0 and set(spin_system_plus.basis.tot_proj[1:]) == {1}
    spin_system_longitudinal = respinach.basis(spin_system, {**NONE, "longitudinal": [["13C"]]})
    assert set(spin_system_longitudinal.basis.basis[:, 1]) == {0, 2} and spin_system_longitudinal.basis.nstates == 32
    spin_system_by_number = respinach.basis(spin_system, {**NONE, "longitudinal": [[1]]})
    assert spin_system_by_number.basis.basis.tolist() == spin_system_longitudinal.basis.basis.tolist()
    spin_system_zq = respinach.basis(spin_system, {**NONE, "zero_quantum": [["1H"]]})
    _, m = respinach.operators.lin2lm(spin_system_zq.basis.basis[:, [0, 2]])
    assert np.all(m.sum(axis=1) == 0) and spin_system_zq.basis.nstates == 6 * 4
    spin_system_manual = respinach.basis(
        spin_system, {**NONE, "approximation": "IK-0", "inter_level": 1, "manual": [[True, False, True]]}
    )
    assert spin_system_manual.basis.nstates == 1 + 3 * 3 + 9


def test_chemical_subsystems_share_only_the_unit_state():
    spin_system = system(["1H", "1H"], zeeman={"scalar": [0.0, 3.0]}, chem={"parts": [[0], [1]], "concs": [1, 1]})
    bas = respinach.basis(spin_system, NONE).basis
    assert bas.basis.tolist() == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]]
    filtered = respinach.basis(spin_system, {**NONE, "projections": [[1], []]}).basis
    assert filtered.basis.tolist() == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]]
    with pytest.raises(ValueError, match="outside substance"):
        respinach.basis(spin_system, {**NONE, "longitudinal": [[1], []]})
    with pytest.raises(ValueError, match="crosses chemical substance"):
        respinach.basis(spin_system, {**NONE, "manual": [[True, True]]})
    with pytest.raises(ValueError, match="only available for sphten-liouv"):
        respinach.basis(spin_system, {"formalism": "zeeman-hilb", "approximation": "none"})


def test_dfpt_and_prune():
    con = np.eye(4, dtype=bool)
    con[0, 1] = con[1, 0] = con[1, 2] = con[2, 1] = True  # chain 0-1-2, spin 3 isolated
    two = dfpt(con, 2)
    assert sorted(map(tuple, two.astype(int))) == [(0, 0, 0, 1), (0, 1, 1, 0), (1, 1, 0, 0)]
    three = dfpt(con, 3)
    assert sorted(map(tuple, three.astype(int))) == [(0, 0, 0, 1), (1, 1, 1, 0)]
    pruned = prune_subgraphs(np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=bool))
    assert pruned.tolist() == [[True, True, False], [False, True, True]]


@pytest.mark.parametrize(
    "bas, exc, message",
    [
        (
            {"formalism": "sphten-liouv", "approximation": "IK-0", "level": 2},
            ValueError,
            "renamed bas\\['inter_level'\\]",
        ),
        (
            {
                "formalism": "sphten-liouv",
                "approximation": "IK-2",
                "connectivity": "scalar_couplings",
                "space_level": 1,
            },
            ValueError,
            "renamed bas\\['prox_level'\\]",
        ),
        ({"formalism": "sphten-liouv", "approximation": "IK-0"}, ValueError, "inter_level"),
        ({"formalism": "sphten-liouv", "approximation": "IK-0", "inter_level": 7}, ValueError, "greater than"),
        ({"formalism": "sphten-liouv", "approximation": "IK-2", "prox_level": 1}, ValueError, "connectivity"),
        ({"formalism": "sphten-liouv", "approximation": "none", "inter_level": 1}, ValueError, "only applicable"),
        ({"formalism": "sphten-liouv", "approximation": "ESR-1"}, ValueError, "unrecognized approximation"),
        (
            {"formalism": "sphten-liouv", "approximation": "IK-DNP", "inter_level": [1, 1, 1]},
            NotImplementedError,
            "IK-DNP",
        ),
        (
            {"formalism": "sphten-liouv", "approximation": "none", "sym_group": ["S2"], "sym_spins": [[0, 1]]},
            NotImplementedError,
            "symmetry",
        ),
        ({"formalism": "zeeman-liouv", "approximation": "none"}, NotImplementedError, "zeeman-liouv"),
        (
            {"formalism": "sphten-liouv", "approximation": "none", "projections": [[1], [1]]},
            ValueError,
            "one element per",
        ),
        ({"formalism": "sphten-liouv", "approximation": "none", "longitudinal": [["19F"]]}, ValueError, "not present"),
    ],
)
def test_refusals(bas, exc, message):
    with pytest.raises(exc, match=message):
        respinach.basis(system(["1H", "1H", "1H"]), bas)
