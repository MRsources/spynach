"""Parity of the zeeman-hilb pulse-acquire slice against Spinach fixtures.

Each stage of oracle/matlab/dumps/dump_hilb_pa.m is recomputed with respinach
and compared with the stored MATLAB result. The case definitions below must
stay identical to the ones in the dump script. Tests skip when the fixtures
are not available.

Tolerances: spin operators agree to a few ulp (MATLAB's 2^(-1/2) is one ulp
below the correctly rounded value, so Spinach's Lz is 0.4999999999999999).

Hamiltonians are built from bit-identical interaction tensors, but clean_up()
rounds every element to a 1e-10 rad/s grid, and Spinach sums the Hamiltonian
terms in random order (randperm in hamiltonian.m). An element whose unrounded
value falls near a half-grid point can land on either neighbouring grid
point, so Spinach itself is reproducible only to about 1e-10 rad/s per
element (hilb_pa_hc_2spin shows exactly one such element). Over an
acquisition of length T this is a phase difference of about 1e-10 * T rad;
T is at most 2.05 s here, so FIDs are compared at rtol 1e-9. Propagators are
compared at a single time step, where the effect is ~1e-10 * dt relative.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import respinach
from respinach.oracle import assert_matches, find_case, load_fixture

CASES = {
    "ab_2spin": dict(
        sys={"isotopes": ["1H", "1H"]},
        inter={"zeeman": {"scalar": [1.00, 1.20]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}},
        pars={"spins": ["1H"], "offset": 660, "sweep": 500, "npoints": 1024, "zerofill": 4096},
    ),
    "abx_3spin": dict(
        sys={"isotopes": ["1H", "1H", "1H"]},
        inter={
            "zeeman": {"scalar": [1.00, 1.15, 3.00]},
            "coupling": {"scalar": [[0, 15.0, 7.0], [0, 0, 3.0], [0, 0, 0]]},
        },
        pars={"spins": ["1H"], "offset": 1200, "sweep": 2000, "npoints": 2048, "zerofill": 8192},
    ),
    "hc_2spin": dict(
        sys={"isotopes": ["1H", "13C"]},
        inter={"zeeman": {"scalar": [2.00, 30.0]}, "coupling": {"scalar": [[0, 145.0], [0, 0]]}},
        pars={"spins": ["1H"], "offset": 1200, "sweep": 500, "npoints": 1024, "zerofill": 4096},
    ),
    "hd_2spin": dict(
        sys={"isotopes": ["1H", "2H"]},
        inter={"zeeman": {"scalar": [4.50, 4.50]}, "coupling": {"scalar": [[0, 43.0], [0, 0]]}},
        pars={"spins": ["1H"], "offset": 2700, "sweep": 400, "npoints": 1024, "zerofill": 4096},
    ),
}
MAGNET = 14.1
BAS = {"formalism": "zeeman-hilb", "approximation": "none"}
OPS = {"E": "E", "Lz": "Lz", "Lx": "Lx", "Ly": "Ly", "Lp": "L+", "Lm": "L-"}

OPERATOR_TOL = dict(rtol=1e-14, atol=0)
HAMILTONIAN_TOL = dict(rtol=1e-13, atol=0)
PROPAGATOR_TOL = dict(rtol=1e-12, atol=0)
FID_TOL = dict(rtol=1e-9, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"hilb_pa_{case}")
    if path is None:
        pytest.skip(f"fixture hilb_pa_{case} not available")
    return load_fixture(path)


@pytest.fixture
def spin_system(case):
    spec = CASES[case]
    return respinach.basis(respinach.create({**spec["sys"], "magnet": MAGNET}, spec["inter"]), BAS)


def test_case_definitions_match_fixture(fx, case):
    inputs = fx.meta["inputs"]
    spec = CASES[case]
    assert inputs["sys"]["isotopes"] == spec["sys"]["isotopes"]
    assert inputs["sys"]["magnet"] == MAGNET
    assert inputs["bas"] == BAS
    assert inputs["pars"] == spec["pars"]
    np.testing.assert_array_equal(inputs["inter"]["zeeman"]["scalar"], spec["inter"]["zeeman"]["scalar"])
    # MATLAB's jsonencode flattens the cell array column by column
    coupling = np.array(spec["inter"]["coupling"]["scalar"], dtype=float)
    np.testing.assert_array_equal(inputs["inter"]["coupling"]["scalar"], coupling.ravel(order="F"))


def test_tolerances_match_spinach_defaults(fx, spin_system):
    for name, value in vars(spin_system.tols).items():
        assert fx.meta["tols"][name] == value, name


def test_create(fx, spin_system):
    np.testing.assert_array_equal(spin_system.mults, fx["/comp/mults"].ravel())
    np.testing.assert_array_equal(spin_system.basefrqs, fx["/inter/basefrqs"].ravel())
    zeeman_ref = fx["/inter/zeeman_matrix"]
    for n in range(spin_system.nspins):
        np.testing.assert_array_equal(spin_system.zeeman_matrix[n], zeeman_ref[n, 0])
    coupling_ref = fx["/inter/coupling_matrix"]
    for n in range(spin_system.nspins):
        for k in range(spin_system.nspins):
            actual, expected = spin_system.coupling_matrix[n][k], coupling_ref[n, k]
            if expected.size == 0:
                assert actual is None, (n, k)
            else:
                np.testing.assert_array_equal(actual, expected)


def test_operators(fx, spin_system):
    for key, label in OPS.items():
        ref = fx[f"/ops/{key}"]
        for n in range(spin_system.nspins):
            assert_matches(
                respinach.operator(spin_system, [label], [n]), ref[0, n], **OPERATOR_TOL, name=f"{label}({n})"
            )
    assert_matches(respinach.operator(spin_system, ["Lz", "L+"], [0, 1]), fx["/ops/LzLp_12"], **OPERATOR_TOL)
    assert_matches(respinach.operator(spin_system, "Lz", "1H"), fx["/ops/Lz_det"], **OPERATOR_TOL)
    assert_matches(respinach.operator(spin_system, "L+", "1H"), fx["/ops/Lp_det"], **OPERATOR_TOL)


def test_hamiltonian(fx, case, spin_system):
    H = respinach.hamiltonian(respinach.assume(spin_system, "nmr"))
    assert_matches(H, fx["/ham/H"], **HAMILTONIAN_TOL, name="H")
    H_zeeman = respinach.hamiltonian(respinach.assume(spin_system, "nmr", "zeeman"))
    assert_matches(H_zeeman, fx["/ham/H_zeeman"], **HAMILTONIAN_TOL, name="H_zeeman")
    H_coupling = respinach.hamiltonian(respinach.assume(spin_system, "nmr", "couplings"))
    assert_matches(H_coupling, fx["/ham/H_coupling"], **HAMILTONIAN_TOL, name="H_coupling")
    H_off = respinach.frqoffset(spin_system, H, CASES[case]["pars"])
    assert_matches(H_off, fx["/ham/H_off"], **HAMILTONIAN_TOL, name="H_off")


def test_states(fx, spin_system):
    assert_matches(respinach.state(spin_system, "L+", "1H"), fx["/states/rho0"], **OPERATOR_TOL, name="rho0")
    assert_matches(respinach.state(spin_system, "L+", "1H"), fx["/states/coil"], **OPERATOR_TOL, name="coil")


def test_propagator(fx, case, spin_system):
    P = respinach.propagator(spin_system, fx["/ham/H_off"], 1 / CASES[case]["pars"]["sweep"])
    assert_matches(P, fx["/prop/P"], **PROPAGATOR_TOL, name="P")


def test_evolution(fx, case, spin_system):
    pars = CASES[case]["pars"]
    fid = respinach.evolution(
        spin_system,
        fx["/ham/H_off"],
        fx["/states/rho0"],
        1 / pars["sweep"],
        pars["npoints"] - 1,
        "observable",
        coil=fx["/states/coil"],
    )
    assert_matches(fid[:, None], fx["/fid/evolution"], **FID_TOL, name="fid evolution")


def test_liquid_acquire_end_to_end(fx, case, spin_system):
    pars = dict(CASES[case]["pars"])
    pars["rho0"] = respinach.state(spin_system, "L+", "1H")
    pars["coil"] = respinach.state(spin_system, "L+", "1H")
    pars["decouple"] = []
    fid = respinach.liquid(spin_system, respinach.acquire, pars, "nmr")
    assert_matches(fid[:, None], fx["/fid/liquid"], **FID_TOL, name="fid liquid")
    spectrum = np.fft.fftshift(np.fft.fft(fid, pars["zerofill"]))
    assert_matches(spectrum[:, None], fx["/spectrum/raw"], **FID_TOL, name="spectrum")


def test_outputs_have_spinach_storage(fx, spin_system):
    """Operators stay sparse; small Hamiltonians are made dense by clean_up, as in Spinach."""
    assert sps.issparse(respinach.operator(spin_system, "Lz", "1H"))
    assert isinstance(respinach.hamiltonian(respinach.assume(spin_system, "nmr")), np.ndarray)
