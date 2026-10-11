"""Parity of relaxation against Spinach fixtures (FR-005).

Each stage of oracle/matlab/dumps/dump_relax.m is recomputed with spinpy and
compared with the stored MATLAB result; the case definitions below must
stay identical to the dump script's. Tests skip when the fixtures are not
available.

Tolerances: the Redfield integral is evaluated by the same auxiliary
matrix exponentials with the same truncation (rlx_integration) and the
same clean_up grids, so superoperators agree to rounding of the rlx_zero
grid (1e-10 Hz); equilibrium states and NOE trajectories to 1e-9.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import spinpy
from spinpy.oracle import assert_matches, find_case, load_fixture

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
ANGLES = [0.3, 0.9, 1.2]

CASES = {
    "noe_het": dict(
        sys={"isotopes": ["1H", "13C"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [0.0, 0.0]}, "coordinates": [[0, 0, 0], [0, 0, 1.03]],
               "relaxation": ["redfield"], "equilibrium": "dibari", "rlx_keep": "kite", "temperature": 298,
               "tau_c": [[100e-12]]},
        noe={"dt": 1e-2, "nsteps": 400},
    ),
    "noe_hom": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [0.0, 0.0]}, "coordinates": [[0, 0, 0], [0, 0, 2.00]],
               "relaxation": ["redfield"], "equilibrium": "dibari", "rlx_keep": "kite", "temperature": 298,
               "tau_c": [[1e-9]]},
        noe={"dt": 1e-2, "nsteps": 1000},
    ),
    "csa_dd_secular": dict(
        sys={"isotopes": ["1H", "1H", "13C"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [1.0, 2.5, 50.0], "eigs": [[-3, -3, 6], [-2, 1, 1], [-60, -60, 120]],
                          "euler": [[0, 0, 0], [0.3, 0.6, 0.9], [1.0, 0.5, 0.2]]},
               "coordinates": [[0, 0, 0], [1.8, 0, 0], [0, 0, 1.09]],
               "coupling": {"scalar": [[0, 0, 140], [0, 0, 0], [0, 0, 0]]},
               "relaxation": ["redfield"], "equilibrium": "IME", "rlx_keep": "secular", "temperature": 298,
               "tau_c": [[2e-10]]},
        noe={"dt": 5e-3, "nsteps": 200},
    ),
    "axial_dfs": dict(
        sys={"isotopes": ["1H", "13C"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [1.0, 50.0], "eigs": [None, [-40, -40, 80]], "euler": [None, [0.2, 0.4, 0.6]]},
               "coordinates": [[0, 0, 0], [0, 0, 1.09]],
               "relaxation": ["redfield"], "equilibrium": "zero", "rlx_keep": "labframe", "rlx_dfs": "keep",
               "temperature": 298, "tau_c": [[5e-10, 1e-10]]},
        noe={"dt": 2e-3, "nsteps": 100},
    ),
    "t1t2_damp": dict(
        sys={"isotopes": ["1H", "13C"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [1.0, 50.0]}, "coupling": {"scalar": [[0, 140], [0, 0]]},
               "relaxation": ["t1_t2", "damp"], "equilibrium": "dibari", "rlx_keep": "diagonal", "temperature": 298,
               "r1_rates": [1.0, 2.0], "r2_rates": [5.0, 6.0], "damp_rate": np.diag([0.3, 0.5, 0.7]).tolist()},
        noe={"dt": 1e-2, "nsteps": 100},
    ),
}
R_TOL = dict(rtol=1e-9, atol=0)
STATE_TOL = dict(rtol=1e-9, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"relax_{case}")
    if path is None:
        pytest.skip(f"fixture relax_{case} not available")
    return load_fixture(path)


@pytest.fixture
def spin_system(case):
    spec = CASES[case]
    return spinpy.basis(spinpy.create(spec["sys"], spec["inter"]), LIOUV)


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def column(x):
    return dense(x).reshape(-1)


def test_case_definitions_match_fixture(fx, case):
    inputs, spec = fx.meta["inputs"], CASES[case]
    assert inputs["sys"]["isotopes"] == spec["sys"]["isotopes"]
    assert inputs["inter"]["relaxation"] == spec["inter"]["relaxation"]
    assert inputs["inter"]["rlx_keep"] == spec["inter"]["rlx_keep"]
    assert inputs["inter"]["equilibrium"] == spec["inter"]["equilibrium"]
    assert inputs["noe"] == spec["noe"]


def test_relaxation_superoperators(fx, spin_system):
    assert_matches(spinpy.relaxation(spin_system), fx["/rlx/R"], **R_TOL, name="R")
    assert_matches(spinpy.relaxation(spin_system, ANGLES), fx["/rlx/R_oriented"], **R_TOL, name="R oriented")


def test_equilibrium_and_thermalisation(fx, case, spin_system):
    HL, QL = spinpy.hamiltonian(spinpy.assume(spin_system, "labframe"), "left", aniso=True)
    assert_matches(HL, fx["/ham/HL"], rtol=1e-12, atol=0, name="HL")
    rho_eq = spinpy.equilibrium(spin_system, HL)
    assert_matches(rho_eq, column(fx["/states/rho_eq"]), **STATE_TOL, name="rho_eq")
    assert_matches(spinpy.equilibrium(spin_system, HL, QL, ANGLES), column(fx["/states/rho_eq_oriented"]), **STATE_TOL)
    from dataclasses import replace
    zero = replace(spin_system, rlx=replace(spin_system.rlx, equilibrium="zero"))
    R0 = spinpy.relaxation(zero)
    assert_matches(R0, fx["/rlx/R_zero"], **R_TOL, name="R zero")
    assert_matches(spinpy.thermalize(spin_system, R0, None, None, rho_eq, "IME"), fx["/rlx/R_ime"], **R_TOL, name="IME")
    temperature = CASES[case]["inter"]["temperature"]
    assert_matches(spinpy.thermalize(spin_system, R0, HL, temperature, None, "dibari"), fx["/rlx/R_dibari"], **R_TOL)


def test_transient_noe(fx, case, spin_system):
    noe = CASES[case]["noe"]
    R = fx["/rlx/R"]
    rho_eq = column(fx["/states/rho_eq"])
    Lz1 = spinpy.state(spin_system, ["Lz"], [0])
    rho = rho_eq - 2 * Lz1 * np.vdot(Lz1, rho_eq) / np.vdot(Lz1, Lz1).real
    assert_matches(rho, column(fx["/states/rho_inverted"]), **STATE_TOL, name="rho inverted")
    coil = np.stack([spinpy.state(spin_system, ["Lz"], [k]) for k in range(spin_system.nspins)], axis=1)
    answer = spinpy.evolution(spin_system, 1j * R, coil, rho, noe["dt"], noe["nsteps"], "multichannel")
    ref = dense(fx["/noe/multichannel"])
    assert_matches(answer, ref.reshape(answer.shape), **STATE_TOL, name="NOE")
