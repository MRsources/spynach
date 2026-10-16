"""Parity of anisotropic interactions against Spinach fixtures (FR-004).

Each stage of oracle/matlab/dumps/dump_aniso.m is recomputed with respinach and
compared with the stored MATLAB result; the case definitions below must
stay identical to the dump script's. Tests skip when the fixtures are not
available.

Tolerances: interaction tensors are built from the same numbers in the
same order and agree to rounding; Hamiltonians and Q components carry the
liouv_zero grid; FIDs are compared at 1e-9 as in the other slices.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import respinach
from respinach.conventions import eeqq2nqi, wigner
from respinach.oracle import assert_matches, find_case, load_fixture

NONE_L = {"formalism": "sphten-liouv", "approximation": "none"}
NONE_H = {"formalism": "zeeman-hilb", "approximation": "none"}
ORIENTATIONS = ([0, 0, 0], [0.3, 0.9, 1.2])
GRIDS = ("single_crystal", "leb_2ang_rank_11", "rep_2ang_100pts_sph")

CASES = {
    "csa_pair": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={"zeeman": {"eigs": [[-7, -7, -1], [4, 2, 9]], "euler": [[0, 0, 0], [0.3, 0.9, 1.2]]}},
        bas={**NONE_L, "projections": [[1]]},
        assumptions="nmr",
        pars={"spins": ["1H"], "offset": 0, "sweep": 15000, "npoints": 128, "zerofill": 512},
    ),
    "dip_pair": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [5.0, -2.0]}, "coordinates": [[0, 0, 0], [0, 3.9, 0.1]]},
        bas={**NONE_L, "projections": [[1]]},
        assumptions="nmr",
        pars={"spins": ["1H"], "offset": 1000, "sweep": 12000, "npoints": 128, "zerofill": 512},
    ),
    "nqi_14n": dict(
        sys={"isotopes": ["14N", "14N"], "magnet": 21.1},
        inter={
            "coupling": {
                "matrix": [
                    [eeqq2nqi(1.24e6, 0.22, 1, [0, 0, 0]).tolist(), None],
                    [None, eeqq2nqi(3.06e6, 0.40, 1, [0.2, 0.4, 0.6]).tolist()],
                ]
            }
        },
        bas=NONE_H,
        assumptions="nmr",
        pars={"spins": ["14N"], "offset": 0, "sweep": 6e6, "npoints": 128, "zerofill": 512},
    ),
    "hc_tensors": dict(
        sys={"isotopes": ["1H", "13C"], "magnet": 14.1},
        inter={
            "zeeman": {"eigs": [[1, 1, 7], [20, 60, 100]], "euler": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            "coupling": {
                "eigs": [[None, [100, 120, 200]], [None, None]],
                "euler": [[None, [0.7, 0.8, 0.9]], [None, None]],
            },
            "coordinates": [[0, 0, 0], [0, 0, 1.09]],
        },
        bas=NONE_H,
        assumptions="nmr",
        pars={"spins": ["13C"], "offset": 9000, "sweep": 1e5, "npoints": 128, "zerofill": 512},
    ),
    "esr_nitroxide": dict(
        sys={"isotopes": ["E", "14N"], "magnet": 0.33},
        inter={
            "zeeman": {"eigs": [[2.0089, 2.0061, 2.0027], None], "euler": [[0, 0, 0], None]},
            "coupling": {
                "eigs": [[None, [16e6, 16e6, 86e6]], [None, None]],
                "euler": [[None, [0, 0, 0]], [None, None]],
            },
        },
        bas=NONE_L,
        assumptions="esr",
        pars={"spins": ["E"], "offset": 0, "sweep": 1e9, "npoints": 128, "zerofill": 512},
    ),
}
TENSOR_TOL = dict(rtol=1e-13, atol=0)
HAMILTONIAN_TOL = dict(rtol=1e-12, atol=0)
FID_TOL = dict(rtol=1e-9, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"aniso_{case}")
    if path is None:
        pytest.skip(f"fixture aniso_{case} not available")
    return load_fixture(path)


@pytest.fixture
def spin_system(case):
    spec = CASES[case]
    return respinach.basis(respinach.create(spec["sys"], spec["inter"]), spec["bas"])


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def column(x):
    return dense(x).reshape(-1)


def test_case_definitions_match_fixture(fx, case):
    inputs, spec = fx.meta["inputs"], CASES[case]
    assert inputs["sys"]["isotopes"] == spec["sys"]["isotopes"]
    assert inputs["assumptions"] == spec["assumptions"]
    assert inputs["pars"] == spec["pars"]
    np.testing.assert_allclose(np.array(inputs["orientations"]), np.array(ORIENTATIONS))


def test_interaction_tensors(fx, spin_system):
    zeeman_ref, coupling_ref = fx["/inter/zeeman_matrix"], fx["/inter/coupling_matrix"]
    for n in range(spin_system.nspins):
        assert_matches(spin_system.zeeman_matrix[n], zeeman_ref[n, 0], **TENSOR_TOL, name=f"zeeman {n}")
        for k in range(spin_system.nspins):
            actual, expected = spin_system.coupling_matrix[n][k], coupling_ref[n, k]
            if expected.size == 0 or np.abs(expected).max() == 0:
                assert actual is None, (n, k)
            else:
                assert_matches(actual, expected, **TENSOR_TOL, name=f"coupling {n},{k}")
    prox = dense(fx["/inter/proxmatrix"]).astype(bool)
    np.testing.assert_array_equal(
        spin_system.proxmatrix | np.eye(spin_system.nspins, dtype=bool), prox | np.eye(spin_system.nspins, dtype=bool)
    )


def test_hamiltonian_components(fx, case, spin_system):
    spin_system_assumed = respinach.assume(spin_system, CASES[case]["assumptions"])
    I, Q = respinach.hamiltonian(spin_system_assumed, aniso=True)
    assert_matches(I, fx["/ham/I"], **HAMILTONIAN_TOL, name="I")
    for rank, key in ((0, "/ham/Q1"), (1, "/ham/Q2")):
        ref = fx[key]
        for k in range(2 * rank + 3):
            for m in range(2 * rank + 3):
                assert_matches(Q[rank][k][m], ref[k, m], **HAMILTONIAN_TOL, name=f"Q{rank + 1}[{k},{m}]")
    I_off = respinach.frqoffset(spin_system, I, CASES[case]["pars"])
    assert_matches(I_off, fx["/ham/I_off"], **HAMILTONIAN_TOL, name="I_off")
    for k, angles in enumerate(ORIENTATIONS, start=1):
        H = I_off + respinach.orientation(Q, angles)
        H = (H + H.conj().T) / 2
        assert_matches(H, fx[f"/ham/H_orient_{k}"], **HAMILTONIAN_TOL, name=f"H orientation {k}")
    for rank in (1, 2, 3):
        assert_matches(
            wigner(rank, 0.3, 0.9, 1.2), fx[f"/conv/wigner_{rank}"], rtol=1e-14, atol=1e-15, name=f"wigner {rank}"
        )


def test_grids_match_spinach(fx):
    for name in GRIDS:
        grid = respinach.load_grid(name)
        ref = fx[f"/grids/{name}"]
        np.testing.assert_array_equal(
            np.column_stack([grid["alphas"], grid["betas"], grid["gammas"], grid["weights"]]), ref
        )


def test_crystal_and_powder_fids(fx, case, spin_system):
    spec = CASES[case]
    pars = dict(spec["pars"])
    det = pars["spins"][0]
    pars["rho0"] = respinach.state(spin_system, "L+", det)
    pars["coil"] = respinach.state(spin_system, "L+", det)
    assert_matches(
        column(pars["rho0"]) if pars["rho0"].ndim == 1 else pars["rho0"],
        fx["/states/rho0"] if pars["rho0"].ndim == 2 else column(fx["/states/rho0"]),
        rtol=1e-14,
        atol=0,
    )
    fid = respinach.crystal(
        spin_system, respinach.acquire, {**pars, "orientation": ORIENTATIONS[1]}, spec["assumptions"]
    )
    assert_matches(fid, column(fx["/fid/crystal"]), **FID_TOL, name="crystal")
    fid = respinach.powder(spin_system, respinach.acquire, {**pars, "grid": "leb_2ang_rank_11"}, spec["assumptions"])
    assert_matches(fid, column(fx["/fid/powder_leb11"]), **FID_TOL, name="powder leb11")
    fid = respinach.powder(spin_system, respinach.acquire, {**pars, "grid": "rep_2ang_100pts_sph"}, spec["assumptions"])
    assert_matches(fid, column(fx["/fid/powder_rep100"]), **FID_TOL, name="powder rep100")
    spectrum = np.fft.fftshift(np.fft.fft(fid, pars["zerofill"]))
    assert_matches(spectrum, column(fx["/spectrum/powder_rep100"]), **FID_TOL, name="spectrum")
