"""Fixture-free tests of the ported kernel: algebra, analytic results, and refusals."""

import numpy as np
import pytest
import scipy.sparse as sps

import spinpy
from spinpy.isotopes import spin
from spinpy.operators import irr_sph_ten, pauli
from spinpy.oracle import assert_matches
from spinpy.tolerances import Tolerances
from spinpy.utilities import clean_up, matlab_round

TIGHT = dict(rtol=1e-14, atol=1e-14)
HILB = {"formalism": "zeeman-hilb", "approximation": "none"}


def hilb_system(isotopes, **inter):
    return spinpy.basis(spinpy.create({"isotopes": isotopes, "magnet": 14.1}, inter), HILB)


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def test_isotope_table():
    assert spin("1H") == (2.6752218744e8, 2)
    assert spin("2H")[1] == 3
    assert spin("E") == (-1.76085963023e11, 2)
    assert spin("E3") == (-1.76085963023e11, 3)
    assert spin("G") == (0.0, 1)
    with pytest.raises(ValueError, match="unknown isotope"):
        spin("1X")
    with pytest.raises(ValueError, match="no data"):
        spin("222Rn")


@pytest.mark.parametrize("mult", [2, 3, 4, 5, 6])
def test_spin_algebra(mult):
    ops = {k: dense(v) for k, v in pauli(mult).items()}
    s = (mult - 1) / 2
    assert_matches(ops["x"] @ ops["y"] - ops["y"] @ ops["x"], 1j * ops["z"], **TIGHT)
    casimir = ops["x"] @ ops["x"] + ops["y"] @ ops["y"] + ops["z"] @ ops["z"]
    assert_matches(casimir, s * (s + 1) * np.eye(mult), **TIGHT)


@pytest.mark.parametrize("mult", [2, 3, 4, 5])
def test_spherical_tensors(mult):
    """T(k,m)^dagger = (-1)^m T(k,-m); tensors of different (k,m) are orthogonal; T(1,0) = Lz."""
    tensors = {}
    for rank in range(mult):
        for n, t in enumerate(irr_sph_ten(mult, rank)):
            tensors[(rank, rank - n)] = dense(t)
    for (rank, proj), t in tensors.items():
        assert_matches(t.conj().T, (-1) ** proj * tensors[(rank, -proj)], **TIGHT, name=f"T({rank},{proj})")
    keys = list(tensors)
    for a in keys:
        for b in keys:
            if a != b:
                assert abs(np.vdot(tensors[a], tensors[b])) < 1e-13, (a, b)
    assert_matches(tensors[(1, 0)], dense(pauli(mult)["z"]), **TIGHT)
    assert_matches(tensors[(1, 1)], -dense(pauli(mult)["p"]) / np.sqrt(2), **TIGHT)


def test_operator_forms_and_zero_based_indices():
    ss = hilb_system(["1H", "13C", "1H"])
    single = [dense(spinpy.operator(ss, ["Lz"], [n])) for n in range(3)]
    assert_matches(spinpy.operator(ss, "Lz", "1H"), single[0] + single[2], **TIGHT)
    assert_matches(spinpy.operator(ss, "Lz", [0, 2]), single[0] + single[2], **TIGHT)
    assert_matches(spinpy.operator(ss, "Lz", "all"), sum(single), **TIGHT)
    lx = dense(spinpy.operator(ss, ["Lx"], [1]))
    lp, lm = dense(spinpy.operator(ss, ["L+"], [1])), dense(spinpy.operator(ss, ["L-"], [1]))
    assert_matches(lx, (lp + lm) / 2, **TIGHT)
    assert_matches(spinpy.operator(ss, ["Lz", "Lx"], [0, 1]), single[0] @ lx, **TIGHT)
    assert_matches(spinpy.operator(ss, ["T1,0"], [2]), single[2], **TIGHT)
    with pytest.raises(ValueError, match="between 0 and 2"):
        spinpy.operator(ss, ["Lz"], [3])
    with pytest.raises(ValueError, match="repetitions"):
        spinpy.operator(ss, ["Lz", "Lx"], [0, 0])


def test_matlab_round_is_half_away_from_zero():
    np.testing.assert_array_equal(matlab_round(np.array([0.5, 1.5, 2.5, -0.5, -2.5])), [1, 2, 3, -1, -3])
    np.testing.assert_array_equal(matlab_round(np.array([0.5 - 2.5j])), [1 - 3j])


def test_clean_up_rounding_and_storage():
    tols = Tolerances()
    small = sps.csr_array(np.diag([1.0, 4e-11, 0.0]))
    out = clean_up(tols, small, 1e-10)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.diag([1.0, 0.0, 0.0]))
    big_sparse = sps.eye_array(300, format="csr")
    assert sps.issparse(clean_up(tols, big_sparse, 1e-10))
    assert sps.issparse(clean_up(tols, np.eye(300), 1e-10))
    assert isinstance(clean_up(tols, np.ones((300, 300)), 1e-10), np.ndarray)


def test_single_spin_fid_is_analytic():
    """Spin-1/2 under H = w*Lz: <L+|rho(t)> = exp(-1j*w*t) for rho0 = L+."""
    ss = hilb_system(["1H"])
    w, dt, nsteps = 2 * np.pi * 125.0, 1e-3, 64
    H = w * spinpy.operator(ss, "Lz", "1H")
    rho = spinpy.state(ss, "L+", "1H")
    fid = spinpy.evolution(ss, H, rho, rho, dt, nsteps, "observable")
    assert_matches(fid, np.exp(-1j * w * dt * np.arange(nsteps + 1)), rtol=1e-12, atol=0)


@pytest.mark.parametrize("norm_target", [0.5, 40.0])
@pytest.mark.parametrize("prop_chop, rtol", [(1e-14, 1e-10), (1e-10, 1e-6)])
def test_taylor_propagator_large_dimension(norm_target, prop_chop, rtol):
    """Above tols.small_matrix the scaled-and-squared Taylor path is used; check it against eigh.

    Each Taylor term and each squaring is rounded to a prop_chop grid, so the
    error scales linearly with prop_chop and grows with the number of
    squarings: at the Spinach default 1e-10 and 7 squarings it is ~7e-8.
    The fine grid checks the algorithm; the default grid checks the accuracy
    users actually get.
    """
    rng = np.random.default_rng(1)
    dim = 256
    A = sps.random_array((dim, dim), density=0.02, rng=rng) + 1j * sps.random_array((dim, dim), density=0.02, rng=rng)
    H = (A + A.conj().T).toarray()
    w, V = np.linalg.eigh(H)
    dt = norm_target / np.abs(w).max()
    exact = (V * np.exp(-1j * w * dt)) @ V.conj().T
    ss = spinpy.basis(spinpy.create({"isotopes": ["1H"], "magnet": 14.1, "tols": {"prop_chop": prop_chop}}, {}), HILB)
    P = spinpy.propagator(ss, sps.csr_array(H), dt)
    assert_matches(P, exact, rtol=rtol, atol=0, name=f"Taylor P, ||H dt||={norm_target}, prop_chop={prop_chop}")


def test_unported_inputs_are_refused():
    with pytest.raises(NotImplementedError, match="coordinates"):
        spinpy.create({"isotopes": ["1H"], "magnet": 14.1}, {"coordinates": [[0, 0, 0]]})
    with pytest.raises(NotImplementedError, match="enable"):
        spinpy.create({"isotopes": ["1H"], "magnet": 14.1, "enable": ["gpu"]}, {})
    ss = spinpy.create({"isotopes": ["1H"], "magnet": 14.1}, {})
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        spinpy.basis(ss, {"formalism": "sphten-liouv", "approximation": "none"})
    with pytest.raises(NotImplementedError, match="esr"):
        spinpy.assume(ss, "esr")
    hs = spinpy.basis(ss, HILB)
    with pytest.raises(NotImplementedError, match="final"):
        spinpy.evolution(hs, np.zeros((2, 2)), np.eye(2), np.eye(2), 1.0, 1, "final")
