"""Parity of magic angle spinning (singlerot, Fokker-Planck form) against Spinach fixtures (FR-006).

Each stage of oracle/matlab/dumps/dump_mas.m is recomputed with respinach and
compared with the stored MATLAB result; the case definitions below must
stay identical to the dump script's. Tests skip when the fixtures are not
available.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import respinach
from respinach.contexts import fourdif
from respinach.oracle import assert_matches, find_case, load_fixture

LIOUV = {"formalism": "sphten-liouv", "approximation": "none", "projections": [[1]]}
CSA = dict(
    sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
    inter={"zeeman": {"eigs": [[-7, -7, -1], [4, 2, 9]], "euler": [[0, 0, 0], [0.3, 0.9, 1.2]]}},
)
MAS = {"axis": [1, 1, 1], "max_rank": 7, "grid": "leb_2ang_rank_11", "sweep": 2e4, "npoints": 128, "zerofill": 512}

CASES = {
    "csa": {**CSA, "pars": {"spins": ["1H"], "offset": 0, "rate": 500, **MAS}},
    "dip": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [5.0, -2.0]}, "coordinates": [[0, 0, 0], [0, 3.9, 0.1]]},
        pars={"spins": ["1H"], "offset": 0, "rate": 1000, **MAS},
    ),
    "nqi": dict(
        sys={"isotopes": ["2H"], "magnet": 9.4},
        inter={"coupling": {"eigs": [[[-1e3, -2e3, 3e3]]], "euler": [[[0, 0, 0]]]}},
        pars={"spins": ["2H"], "offset": 0, "rate": 1000, **MAS},
    ),
    "crystal_offaxis": {
        **CSA,
        "pars": {
            "spins": ["1H"],
            "offset": 300,
            "rate": 800,
            "axis": [0.5, 0.2, 1],
            "max_rank": 5,
            "grid": "single_crystal",
            "sweep": 2e4,
            "npoints": 128,
            "zerofill": 512,
        },
    },
}
GENERATOR_TOL = dict(rtol=1e-12, atol=0)
FID_TOL = dict(rtol=1e-9, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"mas_{case}")
    if path is None:
        pytest.skip(f"fixture mas_{case} not available")
    return load_fixture(path)


@pytest.fixture
def spin_system(case):
    spec = CASES[case]
    return respinach.basis(respinach.create(spec["sys"], spec["inter"]), LIOUV)


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def column(x):
    return dense(x).reshape(-1)


def test_case_definitions_match_fixture(fx, case):
    inputs, spec = fx.meta["inputs"], CASES[case]
    assert inputs["sys"]["isotopes"] == spec["sys"]["isotopes"]
    for key in ("rate", "max_rank", "grid", "sweep", "npoints"):
        assert inputs["pars"][key] == spec["pars"][key]
    np.testing.assert_allclose(inputs["pars"]["axis"], spec["pars"]["axis"])


def test_rotor_grid_and_derivative(fx, case):
    spc_dim = 2 * CASES[case]["pars"]["max_rank"] + 1
    phases, d_dphi = fourdif(spc_dim, 1)
    np.testing.assert_allclose(phases, column(fx["/fp/rotor_phases"]), rtol=1e-15, atol=0)
    assert_matches(d_dphi, fx["/fp/d_dphi"], rtol=1e-14, atol=0, name="d/dphi")


def test_fokker_planck_generator(fx, case, spin_system):
    pars = CASES[case]["pars"]
    rotor = column(fx["/fp/rotor_angles"])
    axis = np.asarray(pars["axis"], dtype=float)
    assert np.isclose(rotor[0], np.arctan2(axis[1], axis[0]))
    assert np.isclose(rotor[1], np.pi / 2 - np.arctan2(axis[2], np.hypot(axis[0], axis[1])))
    seen = {}

    def grab(ss, parameters, G, R, K):
        if "G" not in seen:
            seen["G"] = G
        return 0.0

    respinach.singlerot(spin_system, grab, {**pars, "sum_up": False}, "nmr")
    assert_matches(seen["G"], fx["/fp/G_first_orientation"], **GENERATOR_TOL, name="G")


def test_states_and_fid(fx, case, spin_system):
    pars = dict(CASES[case]["pars"])
    det = pars["spins"][0]
    pars["rho0"] = respinach.state(spin_system, "L+", det)
    pars["coil"] = respinach.state(spin_system, "L+", det)
    seen = {}

    def grab(ss, parameters, G, R, K):
        seen.setdefault("rho0", parameters["rho0"])
        seen.setdefault("coil", parameters["coil"])
        return 0.0

    respinach.singlerot(spin_system, grab, {**pars, "grid": "single_crystal"}, "nmr")
    assert_matches(seen["rho0"], column(fx["/states/rho0_fp_crystal"]), rtol=1e-14, atol=0, name="rho0 fp")
    assert_matches(seen["coil"], column(fx["/states/coil_fp"]), rtol=1e-14, atol=0, name="coil fp")
    fid = respinach.singlerot(spin_system, respinach.acquire, pars, "nmr")
    assert_matches(fid, column(fx["/fid/singlerot"]), **FID_TOL, name="fid")
    spectrum = np.fft.fftshift(np.fft.fft(fid, pars["zerofill"]))
    assert_matches(spectrum, column(fx["/spectrum/raw"]), **FID_TOL, name="spectrum")
