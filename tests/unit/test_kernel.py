"""Fixture-free tests of the ported kernel: algebra, analytic results, and refusals."""

import numpy as np
import pytest
import scipy.sparse as sps

import spynach
from spynach.isotopes import spin
from spynach.operators import irr_sph_ten, pauli
from spynach.oracle import assert_matches
from spynach.tolerances import Tolerances
from spynach.utilities import clean_up, matlab_round

TIGHT = dict(rtol=1e-14, atol=1e-14)
HILB = {"formalism": "zeeman-hilb", "approximation": "none"}


def hilb_system(isotopes, **inter):
    return spynach.basis(spynach.create({"isotopes": isotopes, "magnet": 14.1}, inter), HILB)


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
    spin_system = hilb_system(["1H", "13C", "1H"])
    single = [dense(spynach.operator(spin_system, ["Lz"], [n])) for n in range(3)]
    assert_matches(spynach.operator(spin_system, "Lz", "1H"), single[0] + single[2], **TIGHT)
    assert_matches(spynach.operator(spin_system, "Lz", [0, 2]), single[0] + single[2], **TIGHT)
    assert_matches(spynach.operator(spin_system, "Lz", "all"), sum(single), **TIGHT)
    lx = dense(spynach.operator(spin_system, ["Lx"], [1]))
    lp, lm = dense(spynach.operator(spin_system, ["L+"], [1])), dense(spynach.operator(spin_system, ["L-"], [1]))
    assert_matches(lx, (lp + lm) / 2, **TIGHT)
    assert_matches(spynach.operator(spin_system, ["Lz", "Lx"], [0, 1]), single[0] @ lx, **TIGHT)
    assert_matches(spynach.operator(spin_system, ["T1,0"], [2]), single[2], **TIGHT)
    with pytest.raises(ValueError, match="between 0 and 2"):
        spynach.operator(spin_system, ["Lz"], [3])
    with pytest.raises(ValueError, match="repetitions"):
        spynach.operator(spin_system, ["Lz", "Lx"], [0, 0])


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
    spin_system = hilb_system(["1H"])
    w, dt, nsteps = 2 * np.pi * 125.0, 1e-3, 64
    H = w * spynach.operator(spin_system, "Lz", "1H")
    rho = spynach.state(spin_system, "L+", "1H")
    fid = spynach.evolution(spin_system, H, rho, dt, nsteps, "observable", coil=rho)
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
    spin_system = spynach.basis(
        spynach.create({"isotopes": ["1H"], "magnet": 14.1, "tols": {"prop_chop": prop_chop}}, {}), HILB
    )
    P = spynach.propagator(spin_system, sps.csr_array(H), dt)
    assert_matches(P, exact, rtol=rtol, atol=0, name=f"Taylor P, ||H dt||={norm_target}, prop_chop={prop_chop}")


def test_unported_inputs_are_refused():
    with pytest.raises(NotImplementedError, match="pbc"):
        spynach.create({"isotopes": ["1H"], "magnet": 14.1}, {"coordinates": [[0, 0, 0]], "pbc": [[10, 0, 0]]})
    with pytest.raises(NotImplementedError, match="enable"):
        spynach.create({"isotopes": ["1H"], "magnet": 14.1, "enable": ["gpu"]}, {})
    spin_system = spynach.create({"isotopes": ["1H"], "magnet": 14.1}, {})
    with pytest.raises(NotImplementedError, match="zeeman-liouv"):
        spynach.basis(spin_system, {"formalism": "zeeman-liouv", "approximation": "none"})
    spin_system_liouv = spynach.basis(spin_system, {"formalism": "sphten-liouv", "approximation": "none"})
    with pytest.raises(NotImplementedError, match="total"):
        spynach.evolution(spin_system_liouv, np.zeros((4, 4)), np.ones(4), 1.0, 1, "total", coil=np.ones(4))
    with pytest.raises(NotImplementedError, match="qnmr"):
        spynach.assume(spin_system, "qnmr")
    spin_system_hilb = spynach.basis(spin_system, HILB)
    with pytest.raises(NotImplementedError, match="final"):
        spynach.evolution(spin_system_hilb, np.zeros((2, 2)), np.eye(2), 1.0, 1, "final", coil=np.eye(2))


def test_chemistry_defaults_and_exchange():
    spin_system = spynach.create({"isotopes": ["1H", "13C"], "magnet": 14.1}, {})
    assert spin_system.chem.parts == ((0, 1),)
    assert spin_system.chem.rates is None and list(spin_system.chem.concs) == [1.0]
    rates = [[-2e3, 2e3], [2e3, -2e3]]
    spin_system = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {"zeeman": {"scalar": [0.0, 3.0]}, "chem": {"parts": [[1], [0]], "rates": rates, "concs": [1, 1]}},
    )
    assert spin_system.chem.parts == ((1,), (0,))  # parts keep their order, spins inside are sorted
    np.testing.assert_array_equal(spin_system.chem.rates, rates)


@pytest.mark.parametrize(
    "inter, message",
    [
        ({"chem": {"parts": [[0], [1]]}}, "concentrations"),
        ({"chem": {"parts": [[0], [0, 1]], "concs": [1, 1]}}, "only belong to one"),
        ({"chem": {"parts": [[0]], "concs": [1]}}, "every spin must belong"),
        ({"chem": {"parts": [[0], [1]], "concs": [1]}}, "number of initial concentrations"),
        ({"chem": {"parts": [[0], [1]], "concs": [1, 1], "rates": [[-1, 1], [2, -1]]}}, "conservation of matter"),
        ({"chem": {"rates": [[-1, 1], [1, -1]], "concs": [1, 1]}}, "parts"),
        ({"chem": {"parts": [[0], [1]], "concs": [1, 1], "flux_rate": [[0, 1], [0, 0]]}}, "not ported"),
        (
            {"coupling": {"scalar": [[0, 10.0], [0, 0]]}, "chem": {"parts": [[0], [1]], "concs": [1, 1]}},
            "different chemical species",
        ),
    ],
)
def test_chemistry_refusals(inter, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        spynach.create({"isotopes": ["1H", "1H"], "magnet": 14.1}, inter)


def test_exchange_needs_matching_isotope_sequences():
    with pytest.raises(ValueError, match="isotope sequences"):
        spynach.create(
            {"isotopes": ["1H", "13C"], "magnet": 14.1},
            {"chem": {"parts": [[0], [1]], "concs": [1, 1], "rates": [[-1, 1], [1, -1]]}},
        )


def test_backend_adapter_is_numpy_and_gpu_is_refused_with_a_pointer():
    from spynach import backend

    be = backend.get_backend()
    assert be.name == "numpy"
    A = sps.csr_array(np.eye(3))
    assert be.issparse(A) and be.to_device(A) is A and be.to_host(A) is A
    assert be.nnz(A) == 3 and be.norm1(A) == 1.0 and be.dense(A).shape == (3, 3)
    with pytest.raises(NotImplementedError, match="gpu-notes"):
        backend.use("cupy")
    with pytest.raises(NotImplementedError, match="gpu-notes"):
        spynach.create({"isotopes": ["1H"], "magnet": 14.1, "enable": ["gpu"]}, {})
