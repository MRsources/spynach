"""Fixture-free tests of sphten-liouv operators and states against Hilbert-space algebra."""

import numpy as np
import pytest

import respinach
from respinach.operators import irr_sph_ten, ist_product_table, lin2lm
from respinach.oracle import assert_matches

TIGHT = dict(rtol=1e-14, atol=1e-15)
HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}


def system(isotopes, **inter):
    return respinach.create({"isotopes": isotopes, "magnet": 14.1}, inter)


def basis_operators(ss):
    """Normalised product operators of every basis state of a sphten-liouv system."""
    ops = []
    for row in ss.basis.basis:
        t = np.array([[1.0 + 0j]])
        for mult, index in zip(ss.mults, row):
            rank, proj = lin2lm(int(index))
            t = np.kron(t, irr_sph_ten(int(mult), rank)[rank - proj].toarray())
        ops.append(t / np.sqrt(np.vdot(t, t).real))
    return np.array(ops)


def reference_superop(B, Q, operator_type):
    act = {
        "left": lambda b: Q @ b,
        "right": lambda b: b @ Q,
        "comm": lambda b: Q @ b - b @ Q,
        "acomm": lambda b: Q @ b + b @ Q,
    }[operator_type]
    return np.array([[np.vdot(B[k], act(B[j])) for j in range(len(B))] for k in range(len(B))])


def test_spin_half_product_tables_match_spinach():
    """The hard-coded tables of ist_product_table.m, pages are the output state k."""
    left, right = ist_product_table(2)
    assert left.dtype == float and left.shape == (4, 4, 4)
    np.testing.assert_array_equal(left[:, :, 0], [[1, 0, 0, 0], [0, 0, 0, -0.5], [0, 0, 0.5, 0], [0, -0.5, 0, 0]])
    np.testing.assert_array_equal(left[:, :, 1], [[0, 1, 0, 0], [0.5, 0, -0.5, 0], [0, 0.5, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(left[:, :, 2], [[0, 0, 1, 0], [0, 0, 0, -0.5], [0.5, 0, 0, 0], [0, 0.5, 0, 0]])
    np.testing.assert_array_equal(left[:, :, 3], [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -0.5], [0.5, 0, 0.5, 0]])
    np.testing.assert_array_equal(right[:, :, 0], left[:, :, 0])
    np.testing.assert_array_equal(right[:, :, 1], [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, -0.5, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(right[:, :, 2], [[0, 0, 1, 0], [0, 0, 0, 0.5], [0.5, 0, 0, 0], [0, -0.5, 0, 0]])
    np.testing.assert_array_equal(right[:, :, 3], [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0.5], [0.5, 0, -0.5, 0]])


@pytest.mark.parametrize("mult", [2, 3, 4])
def test_product_tables_are_structure_coefficients(mult):
    """Tn Bm = sum_k left[n, m, k] Bk and Bm Tn = sum_k right[n, m, k] Bk."""
    tensors = np.array([t.toarray() for rank in range(mult) for t in irr_sph_ten(mult, rank)])
    norms = np.sqrt(np.einsum("nij,nij->n", tensors.conj(), tensors).real)
    normalised = tensors / norms[:, None, None]
    left, right = ist_product_table(mult)
    for n in range(mult * mult):
        for m in range(mult * mult):
            assert_matches(tensors[n] @ normalised[m], np.einsum("k,kij->ij", left[n, m], normalised), **TIGHT)
            assert_matches(normalised[m] @ tensors[n], np.einsum("k,kij->ij", right[n, m], normalised), **TIGHT)


@pytest.mark.parametrize("ops, spins", [(["Lz", "L+"], [0, 1]), ("Lx", "2H"), (["T2,1"], [1]), ("Ly", "all")])
@pytest.mark.parametrize("operator_type", ["left", "right", "comm", "acomm"])
def test_superoperators_in_a_complete_basis(ops, spins, operator_type):
    ss = system(["1H", "2H"])
    hs, ls = respinach.basis(ss, HILB), respinach.basis(ss, LIOUV)
    B = basis_operators(ls)
    Q = respinach.operator(hs, ops, spins).toarray()
    A = respinach.operator(ls, ops, spins, operator_type)
    assert A.shape == (36, 36)
    assert_matches(A, reference_superop(B, Q, operator_type), **TIGHT)


def test_restricted_superoperator_is_a_projection():
    """In a truncated basis the superoperator is the complete one restricted to the kept states."""
    ss = system(["1H", "1H", "13C"], coupling={"scalar": [[0, 5.0, 100.0], [0, 0, 20.0], [0, 0, 0]]})
    full = respinach.basis(ss, LIOUV)
    small = respinach.basis(ss, {**LIOUV, "approximation": "IK-0", "inter_level": 2})
    kept = [np.flatnonzero((full.basis.basis == row).all(axis=1))[0] for row in small.basis.basis]
    for ops, spins in (("Lz", "all"), (["L+", "L-"], [0, 2]), (["Lz", "Lz", "L+"], [0, 1, 2])):
        A_full = respinach.operator(full, ops, spins).toarray()
        A_small = respinach.operator(small, ops, spins)
        assert_matches(A_small, A_full[np.ix_(kept, kept)], **TIGHT)
    # A three-spin operator has no matrix elements among states of order <= 2 except via the unit state
    assert respinach.operator(small, ["L+", "L-", "Lz"], [0, 1, 2], "left").count_nonzero() > 0


def test_states_in_liouville_space():
    ss = system(["1H", "13C"], zeeman={"scalar": [1.0, 10.0]})
    ls = respinach.basis(ss, LIOUV)
    B = basis_operators(ls)
    hs = respinach.basis(ss, HILB)
    dim = 4
    for ops, spins in (("Lz", "1H"), ("L+", "all"), (["Lz", "L-"], [0, 1])):
        Q = respinach.state(hs, ops, spins).toarray()
        exact = respinach.state(ls, ops, spins)
        assert exact.shape == (16,) and exact.dtype == complex
        assert_matches(exact, np.array([np.vdot(b, Q) for b in B]) / np.sqrt(dim), **TIGHT)
    lz_exact, lz_cheap = respinach.state(ls, "Lz", "1H"), respinach.state(ls, "Lz", "1H", "cheap")
    row = np.flatnonzero((ls.basis.basis == [2, 0]).all(axis=1))[0]
    assert lz_exact[row] == 0.5 and lz_cheap[row] == 1.0
    assert np.count_nonzero(lz_exact) == 1
    assert_matches(
        respinach.state(ls, ["Lz", "L-"], [0, 1], "cheap"), 4 * respinach.state(ls, ["Lz", "L-"], [0, 1]), **TIGHT
    )
    assert_matches(respinach.state(ls, "Lz", "all", "chem"), lz_exact + respinach.state(ls, "Lz", "13C"), **TIGHT)
    with pytest.raises(ValueError, match="not present in the basis"):
        respinach.state(respinach.basis(ss, {**LIOUV, "projections": [[1]]}), "Lz", "1H", "cheap")


def test_chem_state_weights_by_concentration():
    ss = system(
        ["1H", "1H"],
        zeeman={"scalar": [0.0, 3.0]},
        chem={"parts": [[0], [1]], "concs": [2.0, 0.5], "rates": [[-1.0, 1.0], [1.0, -1.0]]},
    )
    ls = respinach.basis(ss, LIOUV)
    exact, chem = respinach.state(ls, "L+", "1H"), respinach.state(ls, "L+", "1H", "chem")
    expected = exact.copy()
    expected[ls.basis.basis[:, 0] > 0] *= 2.0
    expected[ls.basis.basis[:, 1] > 0] *= 0.5
    assert_matches(chem, expected, **TIGHT)
    with pytest.raises(ValueError, match="crosses chemical species"):
        respinach.state(ls, ["Lz", "Lz"], [0, 1], "chem")
    # The two-spin state across species is absent from the basis
    assert not (ls.basis.basis == [2, 2]).all(axis=1).any()


def test_operator_and_state_refusals():
    ls = respinach.basis(system(["1H"]), LIOUV)
    with pytest.raises(ValueError, match="operator_type"):
        respinach.operator(ls, "Lz", "1H", "commutator")
    with pytest.raises(ValueError, match="unknown state generation method"):
        respinach.state(ls, "Lz", "1H", "fast")
