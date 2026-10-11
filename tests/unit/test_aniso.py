"""Fixture-free tests of anisotropic interactions: conventions, tensor input, Q, orientation, powder averages."""

import numpy as np
import pytest
import scipy.sparse as sps

import spinpy
from spinpy.conventions import eeqq2nqi, euler2dcm, mat2sphten, wigner
from spinpy.oracle import assert_matches

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def system(isotopes, bas=HILB, **inter):
    return spinpy.assume(spinpy.basis(spinpy.create({"isotopes": isotopes, "magnet": 14.1}, inter), bas), "nmr")


def hamiltonian_at(ss, angles):
    I, Q = spinpy.hamiltonian(ss, aniso=True)
    H = dense(I) + dense(spinpy.orientation(Q, angles))
    return (H + H.conj().T) / 2


def test_conventions():
    angles = (0.3, 0.9, 1.2)
    R = euler2dcm(angles)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(euler2dcm((0.3, 0, 0)), [[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]])
    for rank in (1, 2, 3):
        D = wigner(rank, *angles)
        assert D.shape == (2 * rank + 1,) * 2 and np.allclose(D @ D.conj().T, np.eye(2 * rank + 1))
    assert np.allclose(wigner(2, 0, 0, 0), np.eye(5))
    # A symmetric traceless tensor has no rank-1 part; its rank-2 components rotate with D^2
    M = np.array([[1.0, 0.2, 0.3], [0.2, -3.0, 0.4], [0.3, 0.4, 2.0]])
    iso, lam, phi = mat2sphten(M)
    assert iso == 0 and np.allclose(lam, 0) and np.isclose(phi[2], np.sqrt(1 / 6) * (2 * 2.0 - 1.0 + 3.0))
    assert np.isclose(np.linalg.norm(phi) ** 2, np.sum(M * M))      # Frobenius norm is preserved
    _, _, phi_rot = mat2sphten(R @ M @ R.T)
    assert np.allclose(phi_rot, wigner(2, *angles) @ phi) or np.allclose(phi_rot, wigner(2, *angles).conj() @ phi)
    nqi = eeqq2nqi(1.24e6, 0.22, 1, [0.1, 0.2, 0.3])
    assert np.isclose(np.trace(nqi), 0) and np.allclose(nqi, nqi.T)
    assert np.allclose(np.sort(np.linalg.eigvalsh(eeqq2nqi(1.24e6, 0.22, 1, [0, 0, 0]))),
                       np.sort([-1.24e6 * 0.78 / 4, -1.24e6 * 1.22 / 4, 1.24e6 / 2]))


def test_tensor_inputs():
    ss = spinpy.create({"isotopes": ["1H", "13C"], "magnet": 14.1},
                       {"zeeman": {"eigs": [[-2, -2, 4], None], "euler": [[0.3, 0.9, 1.2], None], "scalar": [1.0, 50.0]},
                        "coupling": {"eigs": [[None, [-1.0, -1.0, 2.0]], [None, None]],
                                     "euler": [[None, [0.1, 0.2, 0.3]], [None, None]], "scalar": [[0, 140.0], [0, 0]]}})
    R = euler2dcm([0.3, 0.9, 1.2])
    shift = (ss.zeeman_matrix[0] / ss.basefrqs[0] - np.eye(3)) * 1e6
    assert np.allclose(shift, R @ np.diag([-2, -2, 4]) @ R.T + np.eye(3))
    R = euler2dcm([0.1, 0.2, 0.3])
    assert np.allclose(ss.coupling_matrix[0][1] / (2 * np.pi), R @ np.diag([-1.0, -1.0, 2.0]) @ R.T + 140.0 * np.eye(3))
    assert ss.coupling_matrix[1][0] is None
    with pytest.raises(ValueError, match="euler"):
        spinpy.create({"isotopes": ["1H"], "magnet": 14.1}, {"zeeman": {"eigs": [[1, 2, 3]]}})
    with pytest.raises(ValueError, match="identical non-empty"):
        spinpy.create({"isotopes": ["1H"], "magnet": 14.1}, {"zeeman": {"eigs": [[1, 2, 3]], "euler": [None]}})


def test_dipolar_couplings_from_coordinates():
    """dipolar.m stores half the dipolar tensor in both [n][k] and [k][n]; the pair counts twice in H."""
    ss = spinpy.create({"isotopes": ["1H", "13C"], "magnet": 14.1}, {"coordinates": [[0, 0, 0], [0, 0, 1.09]]})
    r = 1.09e-10
    b = ss.gammas[0] * ss.gammas[1] * ss.tols.hbar * ss.tols.mu0 / (4 * np.pi * r**3)
    D = ss.coupling_matrix[0][1]
    assert np.allclose(D, 0.5 * b * np.diag([1, 1, -2])) and np.allclose(ss.coupling_matrix[1][0], D)
    assert ss.proxmatrix.tolist() == [[False, True], [True, False]]
    assert np.isclose(abs(b) / (2 * np.pi), 23.3e3, rtol=0.01)     # the usual 1H-13C one-bond dipolar coupling
    far = spinpy.create({"isotopes": ["1H", "13C"], "magnet": 14.1, "tols": {"prox_cutoff": 1.0}},
                        {"coordinates": [[0, 0, 0], [0, 0, 1.09]]})
    assert far.coupling_matrix[0][1] is None and not far.proxmatrix.any()
    with pytest.raises(ValueError, match="collision"):
        spinpy.create({"isotopes": ["1H", "1H"], "magnet": 14.1}, {"coordinates": [[0, 0, 0], [0, 0, 0.3]]})
    # Coordinates only couple spins within one chemical subsystem
    two = spinpy.create({"isotopes": ["1H", "1H"], "magnet": 14.1},
                        {"coordinates": [[0, 0, 0], [0, 0, 2.0]], "chem": {"parts": [[0], [1]], "concs": [1, 1]}})
    assert two.coupling_matrix[0][1] is None


def test_proximity_graph_drives_ik_bases():
    coords = [[0, 0, 0], [0, 0, 1.5], [0, 0, 3.0], [0, 0, 20.0]]
    ss = spinpy.create({"isotopes": ["1H"] * 4, "magnet": 14.1, "tols": {"prox_cutoff": 2.0}}, {"coordinates": coords})
    assert ss.proxmatrix.sum() == 4        # 0-1 and 1-2, both ways
    bas = spinpy.basis(ss, {**LIOUV, "approximation": "IK-1", "connectivity": "scalar_couplings",
                            "inter_level": 1, "prox_level": 2}).basis
    # pairs (0,1) and (1,2) correlate; spin 3 and the pair (0,2) do not
    assert bas.nstates == 1 + 4 * 3 + 2 * 9


@pytest.mark.parametrize("bas", [HILB, LIOUV])
def test_csa_hamiltonian_follows_the_rotated_tensor(bas):
    ss = system(["1H"], bas, zeeman={"eigs": [[-2, -2, 4]], "euler": [[0, 0, 0]], "scalar": [3.0]})
    Lz = dense(spinpy.operator(ss, "Lz", "1H"))
    for angles in ([0, 0, 0], [0.3, 0.9, 1.2], [1.0, 2.0, 3.0]):
        R = euler2dcm(angles)
        Z = R @ ss.zeeman_matrix[0] @ R.T
        expected = (Z[2, 2] - ss.basefrqs[0]) * Lz          # secular: the zz element of the rotated tensor
        np.testing.assert_allclose(hamiltonian_at(ss, angles), expected, rtol=0, atol=1e-6)


def test_dipolar_pair_angular_dependence():
    ss = system(["1H", "1H"], HILB, coordinates=[[0, 0, 0], [0, 0, 3.0]])
    LzSz = dense(spinpy.operator(ss, ["Lz", "Lz"], [0, 1]))
    flip = dense(spinpy.operator(ss, ["L+", "L-"], [0, 1])) + dense(spinpy.operator(ss, ["L-", "L+"], [0, 1]))
    for beta in (0.0, 0.7, np.pi / 2, np.arccos(1 / np.sqrt(3))):
        D = euler2dcm([0, beta, 0]) @ ss.coupling_matrix[0][1] @ euler2dcm([0, beta, 0]).T
        expected = 2 * D[2, 2] * (LzSz - flip / 4)        # twice: both [0][1] and [1][0] hold half the tensor
        np.testing.assert_allclose(hamiltonian_at(ss, [0, beta, 0]), expected, rtol=0, atol=1e-7)
    assert np.abs(hamiltonian_at(ss, [0, np.arccos(1 / np.sqrt(3)), 0])).max() < 1e-7   # magic angle


def test_quadrupolar_coupling_enters_through_q():
    nqi = eeqq2nqi(1.24e6, 0.22, 1, [0, 0, 0])
    ss = system(["14N"], HILB, coupling={"matrix": [[nqi.tolist()]]})
    I, Q = spinpy.hamiltonian(ss, aniso=True)
    assert np.abs(dense(I)).max() == 0                      # no isotropic part
    H = hamiltonian_at(ss, [0, 0, 0])
    T20 = dense(spinpy.operator(ss, ["T2,0"], [0]))
    _, _, phi = mat2sphten(2 * np.pi * nqi)
    np.testing.assert_allclose(H, phi[2] * T20, rtol=1e-12, atol=1e-6)   # secular: T(2,0) only under 'nmr'
    with pytest.raises(ValueError, match="traceless"):
        spinpy.hamiltonian(system(["14N"], HILB, coupling={"matrix": [[np.eye(3).tolist()]]}), aniso=True)


def test_powder_average_and_crystal():
    ss = spinpy.basis(spinpy.create({"isotopes": ["1H"], "magnet": 14.1},
                                    {"zeeman": {"eigs": [[-2, -2, 4]], "euler": [[0, 0, 0]], "scalar": [3.0]}}), LIOUV)
    pars = {"spins": ["1H"], "offset": 0.0, "sweep": 20000.0, "npoints": 64,
            "rho0": spinpy.state(ss, "L+", "1H"), "coil": spinpy.state(ss, "L+", "1H")}
    single = spinpy.powder(ss, spinpy.acquire, {**pars, "grid": "single_crystal"}, "nmr")
    assert_matches(single, spinpy.crystal(ss, spinpy.acquire, {**pars, "orientation": [0, 0, 0]}, "nmr"), rtol=1e-12, atol=0)
    fid = spinpy.powder(ss, spinpy.acquire, {**pars, "grid": "leb_2ang_rank_23"}, "nmr")
    assert fid.shape == (64,)
    # The powder-averaged FID at t = 0 is the full magnetisation, and its initial phase velocity is the
    # isotropic shift (first moment of the powder pattern); use a dwell time short against the anisotropy
    assert np.isclose(fid[0], 0.5)
    fast = spinpy.powder(ss, spinpy.acquire, {**pars, "sweep": 1e6, "npoints": 2, "grid": "leb_2ang_rank_23"}, "nmr")
    iso = ss.basefrqs[0] * 3.0e-6
    assert np.isclose(np.angle(fast[1] / fast[0]) * 1e6, -iso, rtol=1e-3)
    per_orientation = spinpy.powder(ss, spinpy.acquire, {**pars, "grid": "leb_2ang_rank_11", "sum_up": False}, "nmr")
    grid = spinpy.load_grid("leb_2ang_rank_11")
    assert len(per_orientation) == grid["weights"].size and np.isclose(grid["weights"].sum(), 1)
    with pytest.raises(ValueError, match="unknown spherical grid"):
        spinpy.load_grid("no_such_grid")
    with pytest.raises(ValueError, match="orientation"):
        spinpy.crystal(ss, spinpy.acquire, pars, "nmr")


def test_liouville_and_hilbert_q_agree():
    inter = {"zeeman": {"eigs": [[-2, -2, 4], [1, -3, 2]], "euler": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
             "coordinates": [[0, 0, 0], [0, 2.5, 0.5]]}
    hs, ls = system(["1H", "1H"], HILB, **inter), system(["1H", "1H"], LIOUV, **inter)
    from .test_superop import basis_operators, reference_superop

    B = basis_operators(ls)
    for angles in ([0, 0, 0], [0.5, 1.1, 2.0]):
        H = hamiltonian_at(hs, angles)
        L = hamiltonian_at(ls, angles)
        np.testing.assert_allclose(L, reference_superop(B, H, "comm"), rtol=0, atol=1e-6)
