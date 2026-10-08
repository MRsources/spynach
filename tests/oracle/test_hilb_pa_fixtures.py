"""Checks on the zeeman-hilb pulse-acquire fixtures themselves.

These run only when the fixtures from dump_hilb_pa.m are present. They verify
the HDF5 layout through the /probe variables and check the stored data
against itself and basic physics. They test the oracle output, not spinpy.
"""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sps

from spinpy.oracle import assert_matches, find_case, load_fixture

CASES = ["ab_2spin", "abx_3spin", "hc_2spin", "hd_2spin"]
EXPECTED_MULTS = {
    "ab_2spin": [2, 2],
    "abx_3spin": [2, 2, 2],
    "hc_2spin": [2, 2],
    "hd_2spin": [2, 3],
}
TIGHT = dict(rtol=1e-12, atol=0)


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


@pytest.fixture(params=CASES)
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"hilb_pa_{case}")
    if path is None:
        pytest.skip(f"fixture hilb_pa_{case} not available")
    return load_fixture(path)


def test_layout_probes(fx):
    np.testing.assert_array_equal(
        fx["/probe/dense"],
        np.arange(1, 25).reshape((2, 3, 4), order="F") + 1j * np.arange(101, 125).reshape((2, 3, 4), order="F"),
    )
    assert fx["/probe/column"].shape == (3, 1)
    expected = np.zeros((3, 4), dtype=complex)
    expected[0, 1], expected[2, 0], expected[1, 3] = 5 + 1j, 7, -2j
    np.testing.assert_array_equal(fx["/probe/sparse"].toarray(), expected)
    cell = fx["/probe/cell"]
    np.testing.assert_array_equal(cell[0, 0], [[1.0, 2.0]])
    np.testing.assert_array_equal(dense(cell[0, 1]), np.eye(2))
    assert cell[1, 0].size == 0
    np.testing.assert_array_equal(cell[1, 1], [[7.0]])


def test_meta(fx):
    assert fx.meta["schema_version"] == 1
    assert fx.meta["formalism"] == "zeeman-hilb"
    assert fx.meta["spinach_dirty"] is False
    assert len(fx.meta["spinach_commit"]) == 40


def test_spin_operators(fx, case):
    mults = fx["/comp/mults"].ravel().astype(int).tolist()
    assert mults == EXPECTED_MULTS[case]
    dim = int(np.prod(mults))
    ops = {key: fx[f"/ops/{key}"] for key in ["E", "Lz", "Lx", "Ly", "Lp", "Lm"]}
    for k in range(len(mults)):
        lx, ly, lz = (dense(ops[key][0, k]) for key in ("Lx", "Ly", "Lz"))
        assert lx.shape == (dim, dim)
        assert_matches(lx @ ly - ly @ lx, 1j * lz, **TIGHT, name=f"[Lx,Ly]=iLz spin {k}")
        assert_matches(dense(ops["Lp"][0, k]), lx + 1j * ly, **TIGHT, name=f"L+ spin {k}")
        assert_matches(dense(ops["Lm"][0, k]), lx - 1j * ly, **TIGHT, name=f"L- spin {k}")
        assert_matches(dense(ops["E"][0, k]), np.eye(dim), **TIGHT, name=f"E spin {k}")
        s = (mults[k] - 1) / 2
        casimir = lx @ lx + ly @ ly + lz @ lz
        assert_matches(casimir, s * (s + 1) * np.eye(dim), **TIGHT, name=f"L^2 spin {k}")
    assert_matches(dense(fx["/ops/LzLp_12"]), dense(ops["Lz"][0, 0]) @ dense(ops["Lp"][0, 1]), **TIGHT)


def test_hamiltonian(fx):
    H, H_off = dense(fx["/ham/H"]), dense(fx["/ham/H_off"])
    assert_matches(H, H.conj().T, **TIGHT, name="H hermitian")
    assert_matches(H, dense(fx["/ham/H_zeeman"]) + dense(fx["/ham/H_coupling"]), rtol=1e-12, atol=1e-9, name="H split")
    offset = fx.meta["inputs"]["pars"]["offset"]
    assert_matches(H_off, H + 2 * np.pi * offset * dense(fx["/ops/Lz_det"]), rtol=1e-12, atol=1e-9, name="H_off")


def test_propagator(fx):
    H_off, P = dense(fx["/ham/H_off"]), dense(fx["/prop/P"])
    dt = fx.meta["inputs"]["timestep"]
    assert_matches(P.conj().T @ P, np.eye(P.shape[0]), rtol=1e-12, atol=1e-12, name="P unitary")
    assert_matches(P, sla.expm(-1j * dt * H_off), rtol=1e-11, atol=0, name="P = expm(-iH dt)")


def test_fids(fx):
    P, rho, coil = (dense(fx[n]) for n in ("/prop/P", "/states/rho0", "/states/coil"))
    assert_matches(rho, dense(fx["/ops/Lp_det"]), **TIGHT, name="rho0 = L+")
    npoints = fx.meta["inputs"]["pars"]["npoints"]
    fid = np.empty((npoints, 1), dtype=complex)
    for k in range(npoints):
        fid[k, 0] = np.vdot(coil, rho)
        rho = P @ rho @ P.conj().T
    explicit = fx["/fid/explicit"]
    assert_matches(fid, explicit, rtol=1e-10, atol=0, name="fid recomputed")
    assert_matches(fx["/fid/evolution"], explicit, rtol=1e-10, atol=0, name="fid evolution")
    assert_matches(fx["/fid/liquid"], explicit, rtol=1e-10, atol=0, name="fid liquid")
    zerofill = fx.meta["inputs"]["pars"]["zerofill"]
    spectrum = np.fft.fftshift(np.fft.fft(fx["/fid/liquid"], zerofill, axis=0), axes=0)
    assert_matches(spectrum, fx["/spectrum/raw"], rtol=1e-12, atol=0, name="spectrum")
