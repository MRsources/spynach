"""Fixture-free tests of magic angle spinning in Fokker-Planck form: spectral derivatives, rotor echoes,
the static and fast-spinning limits, and the Fokker-Planck state layout."""

import numpy as np
import pytest

import respinach
from respinach.contexts import fourdif

LIOUV = {"formalism": "sphten-liouv", "approximation": "none", "projections": [[1]]}


def csa_pair():
    ss = respinach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {"zeeman": {"eigs": [[-7, -7, -1], [4, 2, 9]], "euler": [[0, 0, 0], [0.3, 0.9, 1.2]]}},
    )
    return respinach.basis(ss, LIOUV)


def pars(ss, **extra):
    base = {
        "spins": ["1H"],
        "offset": 0.0,
        "axis": [1, 1, 1],
        "grid": "leb_2ang_rank_11",
        "sweep": 2e4,
        "npoints": 128,
        "rho0": respinach.state(ss, "L+", "1H"),
        "coil": respinach.state(ss, "L+", "1H"),
    }
    return {**base, **extra}


@pytest.mark.parametrize("N", [8, 9, 16, 17])
def test_fourdif_is_exact_for_resolved_harmonics(N):
    x, D1 = fourdif(N, 1)
    _, D2 = fourdif(N, 2)
    _, D3 = fourdif(N, 3)
    for k in range(-(N // 2 - 1), N // 2):
        f = np.exp(1j * k * x)
        np.testing.assert_allclose(D1 @ f, 1j * k * f, atol=1e-10)
        np.testing.assert_allclose(D2 @ f, -(k**2) * f, atol=1e-9)
        np.testing.assert_allclose(D3 @ f, (1j * k) ** 3 * f, atol=1e-8)
    assert np.allclose(D1, -D1.T) and np.allclose(D2, D2.T)  # odd/even derivatives


def test_rotor_echoes():
    """A single-spin CSA Hamiltonian commutes with itself at all rotor phases, so with the initial
    state averaged over the rotor phase the FID refocuses exactly every rotor period, for every
    orientation, with the phase of the isotropic shift; a non-axial tensor is the harder case."""
    rate, sweep = 500.0, 2e4
    period_points = int(round(sweep / rate))  # 40 dwell times per rotor period
    for eigs in ([-7, -7, -1], [4, 2, 9]):
        ss = respinach.create(
            {"isotopes": ["1H"], "magnet": 14.1}, {"zeeman": {"eigs": [eigs], "euler": [[0.3, 0.9, 1.2]]}}
        )
        ss = respinach.basis(ss, LIOUV)
        p = pars(ss, rate=rate, max_rank=17, sweep=sweep, npoints=161, sum_up=False)
        fids = np.array(respinach.singlerot(ss, respinach.acquire, p, "nmr"))
        echoes = fids[:, ::period_points]
        np.testing.assert_allclose(np.abs(echoes), np.broadcast_to(np.abs(fids[:, :1]), echoes.shape), rtol=1e-6)
        iso = ss.basefrqs[0] * np.mean(eigs) * 1e-6
        phases = np.angle(echoes[:, 1] / echoes[:, 0])
        np.testing.assert_allclose(phases, np.angle(np.exp(-1j * iso / rate)), atol=1e-6)
        assert np.abs(fids[:, period_points // 2]).min() < 0.9 * abs(fids[0, 0])  # dephased between echoes
    # Two uncoupled spins: each refocuses, their sum beats at the shift difference modulo the rate
    ss = csa_pair()
    fid = respinach.singlerot(ss, respinach.acquire, pars(ss, rate=rate, max_rank=17, sweep=sweep, npoints=161), "nmr")
    delta = abs(ss.basefrqs[0]) * 10e-6 / (2 * np.pi)  # shift difference, Hz
    beat = 2 * np.pi * (delta % rate) / rate  # phase difference per period
    n = np.arange(5)
    np.testing.assert_allclose(np.abs(fid[::period_points]), abs(fid[0]) * np.abs(np.cos(n * beat / 2)), rtol=1e-5)


def test_fast_spinning_removes_the_anisotropy():
    """At a spinning rate far above the CSA width the FID is the isotropic-shift precession."""
    ss = csa_pair()
    fid = respinach.singlerot(ss, respinach.acquire, pars(ss, rate=5e5, max_rank=5, sweep=2e4, npoints=64), "nmr")
    t = np.arange(64) / 2e4
    iso = ss.basefrqs * np.array([-5.0, 5.0]) * 1e-6  # isotropic shifts of the two spins, rad/s
    expected = 0.5 * (np.exp(-1j * iso[0] * t) + np.exp(-1j * iso[1] * t)) * fid[0]
    np.testing.assert_allclose(fid, expected, rtol=0, atol=2e-3 * abs(fid[0]))


def test_zero_rate_single_crystal_is_the_static_crystal():
    ss = csa_pair()
    p = pars(ss, rate=0.0, max_rank=0, grid="single_crystal", npoints=32)
    fid = respinach.singlerot(ss, respinach.acquire, p, "nmr")
    # the molecule sits at the grid orientation (0,0,0) rotated by the lab->rotor transformation
    rotor_phi, rotor_theta = np.arctan2(1, 1), np.arccos(1 / np.sqrt(3))
    static = respinach.crystal(ss, respinach.acquire, {**p, "orientation": [rotor_phi, rotor_theta, 0]}, "nmr")
    np.testing.assert_allclose(fid, static, rtol=1e-9, atol=0)


def test_fokker_planck_layout_and_inputs():
    ss = csa_pair()
    seen = {}

    def grab(spin_system, parameters, G, R, K):
        seen.update(parameters=parameters, G=G, R=R, K=K)
        return 0.0

    p = pars(ss, rate=1000.0, max_rank=3)
    respinach.singlerot(ss, grab, p, "nmr")
    spc, spn = seen["parameters"]["spc_dim"], seen["parameters"]["spn_dim"]
    assert spc == 7 and spn == ss.basis.nstates
    assert seen["G"].shape == (spc * spn, spc * spn) and seen["R"].shape == seen["G"].shape
    rho0, coil = seen["parameters"]["rho0"], seen["parameters"]["coil"]
    assert rho0.shape == (spc * spn,)
    np.testing.assert_allclose(rho0.reshape(spc, spn), np.tile(p["rho0"] / spc, (spc, 1)))  # spin index fastest
    np.testing.assert_allclose(coil.reshape(spc, spn), np.tile(p["coil"], (spc, 1)))
    # The rotor term is antisymmetric in the phase index and vanishes without spinning
    G = seen["G"].toarray() if hasattr(seen["G"], "toarray") else np.asarray(seen["G"])
    blocks = G.reshape(spc, spn, spc, spn)
    off = blocks.copy()
    for n in range(spc):
        off[n, :, n, :] = 0
    assert np.abs(off).max() > 0
    respinach.singlerot(ss, grab, {**p, "rate": 0.0}, "nmr")
    G0 = seen["G"].toarray() if hasattr(seen["G"], "toarray") else np.asarray(seen["G"])
    off0 = G0.reshape(spc, spn, spc, spn).copy()
    for n in range(spc):
        off0[n, :, n, :] = 0
    assert np.abs(off0).max() == 0
    per_orientation = respinach.singlerot(ss, respinach.acquire, {**p, "sum_up": False, "npoints": 8}, "nmr")
    assert len(per_orientation) == respinach.load_grid("leb_2ang_rank_11")["weights"].size
    for key in ("rate", "axis", "max_rank", "grid"):
        with pytest.raises(ValueError, match=key):
            respinach.singlerot(ss, grab, {k: v for k, v in p.items() if k != key}, "nmr")
    hs = respinach.basis(
        respinach.create({"isotopes": ["1H"], "magnet": 14.1}, {}),
        {"formalism": "zeeman-hilb", "approximation": "none"},
    )
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        respinach.singlerot(hs, grab, {**p, "rho0": None, "coil": None}, "nmr")
