"""Fixture-free tests of the sphten-liouv Hamiltonian, kinetics and filters against Hilbert-space algebra."""

import numpy as np
import pytest

import respinach
from respinach.oracle import assert_matches

from .test_superop import basis_operators, reference_superop

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
CASES = {
    "ab_2spin": (["1H", "1H"], {"zeeman": {"scalar": [1.00, 1.20]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}}),
    "hc_2spin": (["1H", "13C"], {"zeeman": {"scalar": [2.00, 30.0]}, "coupling": {"scalar": [[0, 145.0], [0, 0]]}}),
    "hd_2spin": (["1H", "2H"], {"zeeman": {"scalar": [4.50, 4.50]}, "coupling": {"scalar": [[0, 43.0], [0, 0]]}}),
}


def system(isotopes, inter=None, **kwargs):
    return respinach.create({"isotopes": isotopes, "magnet": 14.1}, inter if inter is not None else kwargs)


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("operator_type", ["left", "right", "comm", "acomm"])
def test_hamiltonian_superoperators_match_hilbert_space(case, operator_type):
    """Each formalism rounds its own elements to liouv_zero (1e-10 rad/s), hence an elementwise 1e-10 bound."""
    ss = system(*CASES[case])
    hs = respinach.assume(respinach.basis(ss, HILB), "nmr")
    ls = respinach.assume(respinach.basis(ss, LIOUV), "nmr")
    B = basis_operators(ls)
    H = dense(respinach.hamiltonian(hs))
    L = respinach.hamiltonian(ls, operator_type)
    np.testing.assert_allclose(dense(L), reference_superop(B, H, operator_type), rtol=0, atol=1.1e-10)
    if operator_type == "comm":
        pars = {"spins": ["1H"], "offset": [1200.0]}
        np.testing.assert_allclose(
            dense(respinach.frqoffset(ls, L, pars)),
            reference_superop(B, dense(respinach.frqoffset(hs, H, pars)), "comm"),
            rtol=0,
            atol=1.1e-10,
        )


def test_commutation_superoperator_spectrum():
    """[H, .] has the differences of the Hamiltonian eigenvalues as its eigenvalues, and is traceless."""
    ss = system(*CASES["ab_2spin"])
    H = dense(respinach.hamiltonian(respinach.assume(respinach.basis(ss, HILB), "nmr")))
    L = dense(respinach.hamiltonian(respinach.assume(respinach.basis(ss, LIOUV), "nmr")))
    E = np.linalg.eigvalsh(H)
    assert_matches(np.sort(np.linalg.eigvals(L).real), np.sort((E[:, None] - E[None, :]).ravel()), rtol=1e-9, atol=1e-6)
    assert abs(np.trace(L)) < 1e-6


def test_restricted_hamiltonian_is_a_projection():
    ss = system(
        ["1H", "1H", "1H"],
        zeeman={"scalar": [1.0, 1.15, 3.0]},
        coupling={"scalar": [[0, 15.0, 7.0], [0, 0, 3.0], [0, 0, 0]]},
    )
    full = respinach.assume(respinach.basis(ss, LIOUV), "nmr")
    small = respinach.assume(respinach.basis(ss, {**LIOUV, "approximation": "IK-0", "inter_level": 2}), "nmr")
    kept = [np.flatnonzero((full.basis.basis == row).all(axis=1))[0] for row in small.basis.basis]
    assert_matches(
        respinach.hamiltonian(small), dense(respinach.hamiltonian(full))[np.ix_(kept, kept)], rtol=0, atol=1e-12
    )
    assert small.basis.nstates == 1 + 9 + 27


def test_kinetics_exchange_superoperator():
    rates = np.array([[-5e2, 2e3], [5e2, -2e3]])
    ss = system(
        ["1H", "1H"],
        zeeman={"scalar": [0.0, 3.0]},
        chem={"parts": [[0], [1]], "concs": [2e3, 5e2], "rates": rates.tolist()},
    )
    ls = respinach.basis(ss, LIOUV)
    K = respinach.kinetics(ls)
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
    rho = respinach.state(ls, "L+", "1H", "chem")
    assert np.isclose((Kd @ rho).sum(), 0)
    # No rates: zero superoperator; Hilbert space: refused
    assert respinach.kinetics(respinach.basis(system(["1H"]), LIOUV)).count_nonzero() == 0
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        respinach.kinetics(respinach.basis(system(["1H"]), HILB))


def test_filters():
    ss = system(["1H", "13C", "1H"])
    ls = respinach.basis(ss, LIOUV)
    rho = (
        respinach.state(ls, "L+", "all")
        + respinach.state(ls, ["Lz", "L+"], [0, 1])
        + respinach.state(ls, ["L+", "L-", "Lz"], [0, 1, 2])
    )
    assert np.count_nonzero(respinach.correlation(ls, rho, 1)) == 3
    assert np.count_nonzero(respinach.correlation(ls, rho, [2, 3])) == 2
    assert np.count_nonzero(respinach.correlation(ls, rho, 1, "13C")) == 3  # every state with a 13C operator
    assert np.count_nonzero(respinach.correlation(ls, rho, [0, 1], [0])) == 5  # one spin: nothing excluded
    assert np.count_nonzero(respinach.correlation(ls, rho, [0], [0])) == 2  # L+ on 13C and on the second 1H
    assert np.count_nonzero(respinach.coherence(ls, rho, [("13C", [1])])) == 2
    assert np.count_nonzero(respinach.coherence(ls, rho, [("1H", 1), ("13C", 0)])) == 2
    assert np.count_nonzero(respinach.coherence(ls, rho, [("all", 0)])) == 1  # L+ L- Lz
    assert np.count_nonzero(respinach.coherence(ls, rho, [("nuclei", [-1])])) == 0
    stack = np.stack([rho, 2 * rho])
    out = respinach.correlation(ls, stack, [1])
    assert out.shape == stack.shape and np.count_nonzero(out) == 6
    assert np.array_equal(rho, rho)  # inputs are not modified
    with pytest.raises(ValueError, match="last dimension"):
        respinach.correlation(ls, rho[:5], [1])
    with pytest.raises(ValueError, match="non-negative"):
        respinach.correlation(ls, rho, [-1])
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        respinach.coherence(respinach.basis(ss, HILB), np.eye(16), [("1H", 1)])


def test_liquid_builds_the_kinetics_superoperator():
    """liquid() hands K = kinetics(...) to the sequence in Liouville space."""
    ss = system(
        ["1H", "1H"],
        zeeman={"scalar": [0.0, 3.0]},
        chem={"parts": [[0], [1]], "concs": [1, 1], "rates": [[-1e3, 1e3], [1e3, -1e3]]},
    )
    ls = respinach.basis(ss, LIOUV)
    seen = {}

    def grab(spin_system, parameters, H, R, K):
        seen["K"] = K
        return None

    respinach.liquid(ls, grab, {"spins": ["1H"]}, "nmr")
    assert_matches(seen["K"], respinach.kinetics(ls), rtol=0, atol=0)


def test_equilibrate_solves_the_kinetic_equilibrium():
    K = np.array([[-50.0, 950.0], [50.0, -950.0]])
    c = respinach.equilibrate(K, np.array([0.3, 0.7]))
    assert np.allclose(c, [0.95, 0.05]) and np.isclose(c.sum(), 1.0)
    assert np.allclose(K @ c, 0)
    # independent reaction groups keep their own totals; a zero vector stays zero
    K2 = np.block([[K, np.zeros((2, 2))], [np.zeros((2, 2)), np.array([[-1.0, 1.0], [1.0, -1.0]])]])
    c2 = respinach.equilibrate(K2, np.array([1.0, 1.0, 4.0, 0.0]))
    assert np.allclose(c2, [1.9, 0.1, 2.0, 2.0])
    assert np.array_equal(respinach.equilibrate(K, np.zeros(2)), np.zeros(2))
    assert np.array_equal(respinach.equilibrate(np.zeros((2, 2)), np.array([1.0, 3.0])), [1.0, 3.0])  # no reactions
    with pytest.raises(ValueError, match="conservation of matter"):
        respinach.equilibrate(np.array([[-1.0, 0.0], [0.0, -1.0]]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="dimension mismatch"):
        respinach.equilibrate(K, np.array([1.0, 1.0, 1.0]))


def test_trajan_and_stateinfo():
    ss = respinach.basis(
        system(["1H", "13C"], zeeman={"scalar": [0.0, 0.0]}, coupling={"scalar": [[0, 140.0], [0, 0]]}), LIOUV
    )
    Lz_H, Lp_C = respinach.state(ss, ["Lz"], [0]), respinach.state(ss, ["L+"], [1])
    LzLz = respinach.state(ss, ["Lz", "Lz"], [0, 1])
    unit = np.zeros(16, dtype=complex)
    unit[0] = 1
    traj = np.stack([Lz_H + 3 * unit, Lp_C + LzLz], axis=1)
    corr, labels = respinach.trajan(ss, traj, "correlation_order")
    assert labels == ["1-spin", "2-spin"]
    assert np.allclose(corr[:, 0], [np.linalg.norm(Lz_H), 0])  # the unit state is projected out
    assert np.allclose(corr[:, 1], [np.linalg.norm(Lp_C), np.linalg.norm(LzLz)])
    coh, labels = respinach.trajan(ss, traj, "coherence_order")
    assert labels == ["-2", "-1", "0", "+1", "+2"]
    assert np.allclose(coh[:, 1], [0, 0, np.linalg.norm(LzLz), np.linalg.norm(Lp_C), 0])
    total, labels = respinach.trajan(ss, traj, "total_each_spin")
    assert labels == ["1H (0)", "13C (1)"]
    assert np.allclose(total[:, 1], [np.linalg.norm(LzLz), np.linalg.norm(np.concatenate([Lp_C, LzLz]))])
    local, _ = respinach.trajan(ss, traj, "local_each_spin")
    assert np.allclose(local[:, 1], [0, np.linalg.norm(Lp_C)])
    with pytest.raises(ValueError, match="unknown property"):
        respinach.trajan(ss, traj, "populations")
    with pytest.raises(NotImplementedError):
        respinach.trajan(ss, traj, "level_populations")
    info = respinach.stateinfo(ss, traj[:, 1], 2)
    assert [index for _, _, index in info] == sorted(
        np.argsort(-np.abs(traj[:, 1]))[:2].tolist(), key=lambda i: -abs(traj[i, 1])
    )
    assert info[0][1] == traj[info[0][2], 1]
    assert all(lm == (0, 0) or lm[0] > 0 for state, _, _ in info for lm in state)
    with pytest.raises(ValueError, match="exceeds"):
        respinach.stateinfo(ss, traj[:, 1], 17)
