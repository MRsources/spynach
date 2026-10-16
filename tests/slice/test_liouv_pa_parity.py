"""Parity of sphten-liouv propagation against Spinach fixtures (FR-003).

Each stage of oracle/matlab/dumps/dump_liouv_pa.m is recomputed with respinach
and compared with the stored MATLAB result. The case definitions below
must stay identical to the ones in the dump script. Tests skip when the
fixtures are not available.

Tolerances: the Liouvillian carries the liouv_zero rounding (see
test_liouv_setup_parity.py); propagators and steps are compared at 1e-12
relative, FIDs, final states and trajectories at 1e-9 (an acquisition of
up to 0.5 s times the 1e-10 rad/s grid). Reductions are compared as sets
of kept states, since the order of subspaces differs between Tarjan's and
SciPy's component labelling.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import respinach
from respinach.oracle import assert_matches, find_case, load_fixture

from .test_liouv_setup_parity import CHAIN_J, STYRENE

NONE = {"formalism": "sphten-liouv", "approximation": "none"}
EXCHANGE = dict(
    sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
    bas=NONE,
    pars={"spins": ["1H"], "offset": 900, "sweep": 5000, "npoints": 512, "zerofill": 1024},
)
STYRENE_PARS = {"spins": ["1H"], "offset": 1200, "sweep": 2000, "npoints": 512, "zerofill": 2048}

CASES = {
    "ab": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [1.00, 1.20]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}},
        bas=NONE,
        pars={"spins": ["1H"], "offset": 660, "sweep": 500, "npoints": 256, "zerofill": 1024},
    ),
    "hd": dict(
        sys={"isotopes": ["1H", "2H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [4.50, 4.50]}, "coupling": {"scalar": [[0, 43.0], [0, 0]]}},
        bas=NONE,
        pars={"spins": ["1H"], "offset": 2700, "sweep": 400, "npoints": 256, "zerofill": 1024},
    ),
    "inad_krylov": dict(
        sys={"isotopes": ["13C", "13C", "13C"], "magnet": 9.4, "tols": {"krylov_tol": 1}},
        inter={"zeeman": {"scalar": [10, 15, 20]}, "coupling": {"scalar": [[0, 55, 0], [0, 0, 0], [0, 0, 0]]}},
        bas=NONE,
        pars={"spins": ["13C"], "offset": 1800, "sweep": 5000, "npoints": 256, "zerofill": 1024},
    ),
    "hc_pt": dict(
        sys={"isotopes": ["1H", "13C"], "magnet": 14.1, "tols": {"merge_dim": 2}},
        inter={"zeeman": {"scalar": [2.00, 30.0]}, "coupling": {"scalar": [[0, 145.0], [0, 0]]}},
        bas=NONE,
        pars={"spins": ["1H"], "offset": 1200, "sweep": 500, "npoints": 256, "zerofill": 1024},
    ),
    "styrene_ik0_2": dict(
        sys=STYRENE["sys"],
        inter=STYRENE["inter"],
        pars=STYRENE_PARS,
        bas={**NONE, "approximation": "IK-0", "inter_level": 2},
    ),
    "styrene_ik1": dict(
        sys=STYRENE["sys"],
        inter=STYRENE["inter"],
        pars=STYRENE_PARS,
        bas={**NONE, "approximation": "IK-1", "connectivity": "scalar_couplings", "inter_level": 3, "prox_level": 1},
    ),
    "chain12_ik2": dict(
        sys={"isotopes": ["1H"] * 12, "magnet": 14.1},
        inter={"zeeman": {"scalar": np.linspace(-4, 4, 12).tolist()}, "coupling": {"scalar": CHAIN_J.tolist()}},
        bas={**NONE, "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 1},
        pars={"spins": ["1H"], "offset": 0, "sweep": 8000, "npoints": 512, "zerofill": 2048},
    ),
    "exch_sym": {
        **EXCHANGE,
        "inter": {
            "zeeman": {"scalar": [0.0, 3.0]},
            "chem": {"parts": [[0], [1]], "rates": [[-2e3, 2e3], [2e3, -2e3]], "concs": [1.0, 1.0]},
        },
    },
    "exch_asym": {
        **EXCHANGE,
        "inter": {
            "zeeman": {"scalar": [0.0, 3.0]},
            "chem": {"parts": [[0], [1]], "rates": [[-5e2, 2e3], [5e2, -2e3]], "concs": [2e3, 5e2]},
        },
    },
}
NTRAJ = 32
LIOUVILLIAN_TOL = dict(rtol=1e-12, atol=0)
PROPAGATOR_TOL = dict(rtol=1e-12, atol=0)
FID_TOL = dict(rtol=1e-9, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"liouv_pa_{case}")
    if path is None:
        pytest.skip(f"fixture liouv_pa_{case} not available")
    return load_fixture(path)


@pytest.fixture
def spin_system(case):
    spec = CASES[case]
    return respinach.basis(respinach.create(spec["sys"], spec["inter"]), spec["bas"])


@pytest.fixture
def liouvillian(case, spin_system):
    spin_system_nmr = respinach.assume(spin_system, "nmr")
    H = respinach.frqoffset(spin_system, respinach.hamiltonian(spin_system_nmr), CASES[case]["pars"])
    return H + 1j * respinach.kinetics(spin_system)


def column(x):
    return (x.toarray() if sps.issparse(x) else np.asarray(x)).reshape(-1)


def kept_states(projectors):
    """Set of basis states selected by a projector or a list of projectors; a scalar 1 means all."""
    if not isinstance(projectors, (list, tuple, np.ndarray)) or sps.issparse(projectors):
        projectors = [projectors]
    states = set()
    for P in np.asarray(projectors, dtype=object).ravel() if isinstance(projectors, np.ndarray) else projectors:
        if np.ndim(P) == 0 or P.shape == (1, 1):
            return None
        rows = sps.csc_array(P).nonzero()[0] if sps.issparse(P) else np.nonzero(np.asarray(P))[0]
        states |= set(rows.tolist())
    return states


def test_case_definitions_match_fixture(fx, case):
    inputs, spec = fx.meta["inputs"], CASES[case]
    assert inputs["sys"]["isotopes"] == spec["sys"]["isotopes"]
    assert inputs["bas"]["approximation"] == spec["bas"]["approximation"]
    assert inputs["pars"] == spec["pars"]
    assert inputs["ntraj"] == NTRAJ
    for name, value in spec["sys"].get("tols", {}).items():
        assert fx.meta["tols"][name] == value


def test_liouvillian_and_states(fx, case, spin_system, liouvillian):
    det = CASES[case]["pars"]["spins"][0]
    assert_matches(liouvillian, fx["/ham/L"], **LIOUVILLIAN_TOL, name="L")
    assert_matches(respinach.state(spin_system, "L+", det, "chem"), column(fx["/states/rho0"]), rtol=1e-14, atol=0)
    assert_matches(respinach.state(spin_system, "L+", det), column(fx["/states/coil"]), rtol=1e-14, atol=0)


def test_propagator_and_step(fx, case, spin_system):
    dt = 1 / CASES[case]["pars"]["sweep"]
    L, rho0, coil = fx["/ham/L"], column(fx["/states/rho0"]), column(fx["/states/coil"])
    assert_matches(respinach.propagator(spin_system, L, dt), fx["/prop/P"], **PROPAGATOR_TOL, name="P")
    assert_matches(respinach.step(spin_system, L, rho0, dt), column(fx["/step/rho_dt"]), **PROPAGATOR_TOL, name="step")
    assert_matches(respinach.step(spin_system, L, rho0, 7.3 * dt), column(fx["/step/rho_long"]), **PROPAGATOR_TOL)
    stack = np.stack([rho0, 0.5 * coil], axis=1)
    ref = fx["/step/stack"]
    assert_matches(
        respinach.step(spin_system, L, stack, dt), ref.toarray() if sps.issparse(ref) else ref, **PROPAGATOR_TOL
    )


def test_reductions_keep_the_same_states(fx, spin_system):
    L, coil = fx["/ham/L"], column(fx["/states/coil"])
    L_dag = L.conj().T
    expected = kept_states(fx["/reduce/zte_projector"])
    actual = kept_states(respinach.zte(spin_system, L_dag, coil))
    assert actual == expected or (expected is None and len(actual) == spin_system.basis.nstates)
    expected = kept_states(fx["/reduce/projectors"])
    actual = kept_states(respinach.reduce(spin_system, L_dag, coil))
    assert actual == expected or (expected is None and len(actual) == spin_system.basis.nstates)


def test_evolution_outputs(fx, case, spin_system):
    pars = CASES[case]["pars"]
    dt = 1 / pars["sweep"]
    L, rho0, coil = fx["/ham/L"], column(fx["/states/rho0"]), column(fx["/states/coil"])
    fid = respinach.evolution(spin_system, L, rho0, dt, pars["npoints"] - 1, "observable", coil=coil)
    assert_matches(fid, column(fx["/fid/evolution"]), **FID_TOL, name="observable")
    final = respinach.evolution(spin_system, L, rho0, dt, pars["npoints"] - 1, "final")
    assert_matches(final, column(fx["/fid/final"]), **FID_TOL, name="final")
    traj = respinach.evolution(spin_system, L, rho0, dt, NTRAJ - 1, "trajectory")
    ref = fx["/fid/trajectory"]
    assert_matches(traj, ref.toarray() if sps.issparse(ref) else ref, **FID_TOL, name="trajectory")


def test_liquid_acquire_end_to_end(fx, case, spin_system):
    pars = dict(CASES[case]["pars"])
    det = pars["spins"][0]
    pars["rho0"] = respinach.state(spin_system, "L+", det, "chem")
    pars["coil"] = respinach.state(spin_system, "L+", det)
    fid = respinach.liquid(spin_system, respinach.acquire, pars, "nmr")
    assert_matches(fid, column(fx["/fid/liquid"]), **FID_TOL, name="fid liquid")
    spectrum = np.fft.fftshift(np.fft.fft(fid, pars["zerofill"]))
    assert_matches(spectrum, column(fx["/spectrum/raw"]), **FID_TOL, name="spectrum")
