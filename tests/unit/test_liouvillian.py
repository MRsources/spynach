"""Fixture-free tests of the sphten-liouv Hamiltonian, kinetics and filters against Hilbert-space algebra."""

import numpy as np
import pytest

import spinpy
from spinpy.oracle import assert_matches

from .test_superop import basis_operators, reference_superop

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
CASES = {
    "ab_2spin": (["1H", "1H"], {"zeeman": {"scalar": [1.00, 1.20]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}}),
    "hc_2spin": (["1H", "13C"], {"zeeman": {"scalar": [2.00, 30.0]}, "coupling": {"scalar": [[0, 145.0], [0, 0]]}}),
    "hd_2spin": (["1H", "2H"], {"zeeman": {"scalar": [4.50, 4.50]}, "coupling": {"scalar": [[0, 43.0], [0, 0]]}}),
}


def system(isotopes, inter=None, **kwargs):
    return spinpy.create({"isotopes": isotopes, "magnet": 14.1}, inter if inter is not None else kwargs)


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("operator_type", ["left", "right", "comm", "acomm"])
def test_hamiltonian_superoperators_match_hilbert_space(case, operator_type):
    """Each formalism rounds its own elements to liouv_zero (1e-10 rad/s), hence an elementwise 1e-10 bound."""
    ss = system(*CASES[case])
    hs = spinpy.assume(spinpy.basis(ss, HILB), "nmr")
    ls = spinpy.assume(spinpy.basis(ss, LIOUV), "nmr")
    B = basis_operators(ls)
    H = dense(spinpy.hamiltonian(hs))
    L = spinpy.hamiltonian(ls, operator_type)
    np.testing.assert_allclose(dense(L), reference_superop(B, H, operator_type), rtol=0, atol=1.1e-10)
    if operator_type == "comm":
        pars = {"spins": ["1H"], "offset": [1200.0]}
        np.testing.assert_allclose(dense(spinpy.frqoffset(ls, L, pars)),
                                   reference_superop(B, dense(spinpy.frqoffset(hs, H, pars)), "comm"), rtol=0, atol=1.1e-10)


def test_commutation_superoperator_spectrum():
    """[H, .] has the differences of the Hamiltonian eigenvalues as its eigenvalues, and is traceless."""
    ss = system(*CASES["ab_2spin"])
    H = dense(spinpy.hamiltonian(spinpy.assume(spinpy.basis(ss, HILB), "nmr")))
    L = dense(spinpy.hamiltonian(spinpy.assume(spinpy.basis(ss, LIOUV), "nmr")))
    E = np.linalg.eigvalsh(H)
    assert_matches(np.sort(np.linalg.eigvals(L).real), np.sort((E[:, None] - E[None, :]).ravel()), rtol=1e-9, atol=1e-6)
    assert abs(np.trace(L)) < 1e-6


def test_restricted_hamiltonian_is_a_projection():
    ss = system(["1H", "1H", "1H"], zeeman={"scalar": [1.0, 1.15, 3.0]},
                coupling={"scalar": [[0, 15.0, 7.0], [0, 0, 3.0], [0, 0, 0]]})
    full = spinpy.assume(spinpy.basis(ss, LIOUV), "nmr")
    small = spinpy.assume(spinpy.basis(ss, {**LIOUV, "approximation": "IK-0", "inter_level": 2}), "nmr")
    kept = [np.flatnonzero((full.basis.basis == row).all(axis=1))[0] for row in small.basis.basis]
    assert_matches(spinpy.hamiltonian(small), dense(spinpy.hamiltonian(full))[np.ix_(kept, kept)], rtol=0, atol=1e-12)
    assert small.basis.nstates == 1 + 9 + 27


def test_kinetics_exchange_superoperator():
    rates = np.array([[-5e2, 2e3], [5e2, -2e3]])
    ss = system(["1H", "1H"], zeeman={"scalar": [0.0, 3.0]},
                chem={"parts": [[0], [1]], "concs": [2e3, 5e2], "rates": rates.tolist()})
    ls = spinpy.basis(ss, LIOUV)
    K = spinpy.kinetics(ls)
    assert K.shape == (7, 7) and np.allclose(dense(K).sum(axis=0), 0)
    basis = ls.basis.basis
    site = [np.flatnonzero(basis[:, s] > 0) for s in (0, 1)]
    Kd = dense(K)
    assert Kd[0].sum() == 0 and Kd[:, 0].sum() == 0
    for i in range(2):
        for j in range(2):
            block = Kd[np.ix_(site[i], site[j])]
            assert_matches(block, rates[i, j] * np.eye(3), rtol=0, atol=0)
    # Matter is conserved along the state vector too: K maps the single-spin states into each other
    rho = spinpy.state(ls, "L+", "1H", "chem")
    assert np.isclose((Kd @ rho).sum(), 0)
    # No rates: zero superoperator; Hilbert space: refused
    assert spinpy.kinetics(spinpy.basis(system(["1H"]), LIOUV)).count_nonzero() == 0
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        spinpy.kinetics(spinpy.basis(system(["1H"]), HILB))


def test_filters():
    ss = system(["1H", "13C", "1H"])
    ls = spinpy.basis(ss, LIOUV)
    rho = (spinpy.state(ls, "L+", "all") + spinpy.state(ls, ["Lz", "L+"], [0, 1])
           + spinpy.state(ls, ["L+", "L-", "Lz"], [0, 1, 2]))
    assert np.count_nonzero(spinpy.correlation(ls, rho, 1)) == 3
    assert np.count_nonzero(spinpy.correlation(ls, rho, [2, 3])) == 2
    assert np.count_nonzero(spinpy.correlation(ls, rho, 1, "13C")) == 3    # every state with a 13C operator
    assert np.count_nonzero(spinpy.correlation(ls, rho, [0, 1], [0])) == 5    # one spin: nothing excluded
    assert np.count_nonzero(spinpy.correlation(ls, rho, [0], [0])) == 2    # L+ on 13C and on the second 1H
    assert np.count_nonzero(spinpy.coherence(ls, rho, [("13C", [1])])) == 2
    assert np.count_nonzero(spinpy.coherence(ls, rho, [("1H", 1), ("13C", 0)])) == 2
    assert np.count_nonzero(spinpy.coherence(ls, rho, [("all", 0)])) == 1   # L+ L- Lz
    assert np.count_nonzero(spinpy.coherence(ls, rho, [("nuclei", [-1])])) == 0
    stack = np.stack([rho, 2 * rho])
    out = spinpy.correlation(ls, stack, [1])
    assert out.shape == stack.shape and np.count_nonzero(out) == 6
    assert np.array_equal(rho, rho)      # inputs are not modified
    with pytest.raises(ValueError, match="last dimension"):
        spinpy.correlation(ls, rho[:5], [1])
    with pytest.raises(ValueError, match="non-negative"):
        spinpy.correlation(ls, rho, [-1])
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        spinpy.coherence(spinpy.basis(ss, HILB), np.eye(16), [("1H", 1)])


def test_liquid_builds_the_kinetics_superoperator():
    """liquid() hands K = kinetics(...) to the sequence in Liouville space."""
    ss = system(["1H", "1H"], zeeman={"scalar": [0.0, 3.0]},
                chem={"parts": [[0], [1]], "concs": [1, 1], "rates": [[-1e3, 1e3], [1e3, -1e3]]})
    ls = spinpy.basis(ss, LIOUV)
    seen = {}

    def grab(spin_system, parameters, H, R, K):
        seen["K"] = K
        return None

    spinpy.liquid(ls, grab, {"spins": ["1H"]}, "nmr")
    assert_matches(seen["K"], spinpy.kinetics(ls), rtol=0, atol=0)
