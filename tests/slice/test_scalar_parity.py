"""Parity of scalar relaxation (SRFK, SRSK) against Spinach fixtures (FR-008).

Each stage of oracle/matlab/dumps/dump_scalar.m is recomputed with respinach and
compared with the stored MATLAB result; the case definitions below must
stay identical to the dump script's. Tests skip when the fixtures are not
available.

Tolerances: the SRFK integral is the same auxiliary matrix exponential
with the same truncation and clean_up grids as the Redfield one, and
the SRSK rates are closed-form expressions of the Redfield rates, so
superoperators agree to rounding of the rlx_zero grid (1e-10 Hz);
equilibrium states and NOE trajectories to 1e-9.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import respinach
from respinach.oracle import assert_matches, find_case, load_fixture

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}

CASES = {
    "srfk_pair": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={
            "zeeman": {"scalar": [0.0, 2.0]},
            "relaxation": ["SRFK"],
            "rlx_keep": "kite",
            "equilibrium": "zero",
            "srfk_tau_c": [[1.0, 1e-3]],
            "srfk_mdepth": [[None, 15.0], [None, None]],
        },
        noe={"dt": 1e-2, "nsteps": 100},
        base=False,
    ),
    "srsk_hn": dict(
        sys={"isotopes": ["1H", "14N"], "magnet": 11.75},
        inter={
            "zeeman": {"scalar": [0.0, 0.0]},
            "coordinates": [[0, 0, 0], [0, 0, 1.0]],
            "coupling": {
                "scalar": [[0, 50], [0, 0]],
                "eigs": [[None, None], [None, [1e6, 1e6, -2e6]]],
                "euler": [[None, None], [None, [0, 0, 0]]],
            },
            "relaxation": ["redfield", "SRSK"],
            "srsk_sources": [1],
            "rlx_keep": "secular",
            "equilibrium": "zero",
            "tau_c": [[1e-9]],
        },
        noe={"dt": 1e-3, "nsteps": 100},
        base=True,
    ),
    "srfk_srsk_3spin": dict(
        sys={"isotopes": ["1H", "1H", "14N"], "magnet": 11.75},
        inter={
            "zeeman": {
                "scalar": [1.0, 2.0, 0.0],
                "eigs": [[-3, -3, 6], None, [-100, -100, 200]],
                "euler": [[0, 0, 0], None, [0.3, 0.6, 0.9]],
            },
            "coordinates": [[0, 0, 0], [1.8, 0, 0], [0, 0, 1.0]],
            "coupling": {
                "scalar": [[0, 10, 40], [0, 0, 5], [0, 0, 0]],
                "eigs": [[None, None, None], [None, None, None], [None, None, [1e6, 1e6, -2e6]]],
                "euler": [[None, None, None], [None, None, None], [None, None, [0.3, 0.6, 0.9]]],
            },
            "relaxation": ["redfield", "SRFK", "SRSK"],
            "srsk_sources": [2],
            "rlx_keep": "kite",
            "equilibrium": "zero",
            "tau_c": [[1e-9]],
            "srfk_tau_c": [[1.0, 1e-3]],
            "srfk_mdepth": [[None, None, 15.0], [None, None, None], [None, None, None]],
        },
        noe={"dt": 1e-2, "nsteps": 100},
        base=True,
    ),
}
R_TOL = dict(rtol=1e-9, atol=0)
STATE_TOL = dict(rtol=1e-9, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"scalar_{case}")
    if path is None:
        pytest.skip(f"fixture scalar_{case} not available")
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
    assert inputs["inter"]["relaxation"] == spec["inter"]["relaxation"]
    assert inputs["inter"]["rlx_keep"] == spec["inter"]["rlx_keep"]
    assert inputs["noe"] == spec["noe"]


def test_relaxation_superoperators(fx, case, spin_system):
    assert_matches(respinach.relaxation(spin_system), fx["/rlx/R"], **R_TOL, name="R")
    if CASES[case]["base"]:
        from dataclasses import replace

        rlx = spin_system.rlx
        base = replace(
            spin_system,
            rlx=replace(
                rlx,
                theories=tuple(t for t in rlx.theories if t not in ("SRFK", "SRSK")),
                srfk_tau_c=(),
                srfk_mdepth=(),
                srsk_sources=(),
            ),
        )
        assert_matches(respinach.relaxation(base), fx["/rlx/R_base"], **R_TOL, name="R without scalar relaxation")


def test_equilibrium_and_transient_noe(fx, case, spin_system):
    HL = respinach.hamiltonian(respinach.assume(spin_system, "labframe"), "left")
    assert_matches(HL, fx["/ham/HL"], rtol=1e-12, atol=0, name="HL")
    rho_eq = respinach.equilibrium(spin_system, HL)
    assert_matches(rho_eq, column(fx["/states/rho_eq"]), **STATE_TOL, name="rho_eq")
    noe = CASES[case]["noe"]
    R = fx["/rlx/R"]
    rho_eq = column(fx["/states/rho_eq"])
    Lz1 = respinach.state(spin_system, ["Lz"], [0])
    rho = rho_eq - 2 * Lz1 * np.vdot(Lz1, rho_eq) / np.vdot(Lz1, Lz1).real
    assert_matches(rho, column(fx["/states/rho_inverted"]), **STATE_TOL, name="rho inverted")
    coil = np.stack([respinach.state(spin_system, ["Lz"], [k]) for k in range(spin_system.nspins)], axis=1)
    answer = respinach.evolution(spin_system, 1j * R, coil, rho, noe["dt"], noe["nsteps"], "multichannel")
    ref = dense(fx["/noe/multichannel"])
    assert_matches(answer, ref.reshape(answer.shape), **STATE_TOL, name="NOE")
