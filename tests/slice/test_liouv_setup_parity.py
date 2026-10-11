"""Parity of the sphten-liouv setup slice against Spinach fixtures (FR-002).

Each stage of oracle/matlab/dumps/dump_liouv_setup.m is recomputed with
spinpy and compared with the stored MATLAB result. The case definitions
below must stay identical to the ones in the dump script (0-based spin
numbers here). Tests skip when the fixtures are not available.

Tolerances: basis descriptors, product tables for spin-1/2, kinetics and
filters are exact. Superoperators and states agree to a few ulp; computed
product tables (spin-1) to rounding. Hamiltonians are rounded to the
liouv_zero grid after summing terms in random order in Spinach, so they
agree to about 1e-10 rad/s per element (see test_hilb_pa_parity.py).
"""

import numpy as np
import pytest
import scipy.sparse as sps

import spinpy
from spinpy.operators import ist_product_table
from spinpy.oracle import assert_matches, find_case, load_fixture

STYRENE = dict(
    sys={"isotopes": ["1H"] * 8, "magnet": 7.046},
    inter={
        "zeeman": {"scalar": [6.720, 6.720, 6.587, 6.587, 6.527, 6.063, 5.085, 4.555]},
        "coupling": {"scalar": [
            [0.0, 1.9170, 7.7884, 0.6061, 1.2447, -0.5347, 0.0390, 0.1582],
            [0.0, 0.0, 0.6061, 7.7884, 1.2447, -0.5347, 0.0390, 0.1582],
            [0.0, 0.0, 0.0, 1.4184, 7.4390, 0.3693, -0.0416, 0.0169],
            [0.0, 0.0, 0.0, 0.0, 7.4390, 0.3693, -0.0416, 0.0169],
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.2279, 0.2349, 0.2905],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.6002, 10.8954],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0382],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]},
    },
    det="1H", offset=1200,
)
CHAIN_J = np.zeros((12, 12))
for _n in range(11):
    CHAIN_J[_n, _n + 1] = 10.0
NONE = {"formalism": "sphten-liouv", "approximation": "none"}
EXCHANGE = dict(sys={"isotopes": ["1H", "1H"], "magnet": 14.1}, det="1H", offset=900, bas=NONE)

CASES = {
    "ab_none": dict(
        sys={"isotopes": ["1H", "1H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [1.00, 1.20]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}},
        bas=NONE, det="1H", offset=660,
    ),
    "hd_none": dict(
        sys={"isotopes": ["1H", "2H"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [4.50, 4.50]}, "coupling": {"scalar": [[0, 43.0], [0, 0]]}},
        bas=NONE, det="1H", offset=2700,
    ),
    "inad_none": dict(
        sys={"isotopes": ["13C", "13C", "13C"], "magnet": 9.4},
        inter={"zeeman": {"scalar": [10, 15, 20]}, "coupling": {"scalar": [[0, 55, 0], [0, 0, 0], [0, 0, 0]]}},
        bas=NONE, det="13C", offset=1800,
    ),
    **{
        f"styrene_ik0_{level}": {**STYRENE, "bas": {**NONE, "approximation": "IK-0", "inter_level": level}}
        for level in (1, 2, 4)
    },
    "styrene_ik1": {**STYRENE, "bas": {**NONE, "approximation": "IK-1", "connectivity": "scalar_couplings",
                                       "inter_level": 3, "prox_level": 1}},
    "styrene_ik2_proj": {**STYRENE, "bas": {**NONE, "approximation": "IK-2", "connectivity": "scalar_couplings",
                                            "prox_level": 1, "projections": [[1]]}},
    "chain12_ik2": dict(
        sys={"isotopes": ["1H"] * 12, "magnet": 14.1},
        inter={"zeeman": {"scalar": np.linspace(-4, 4, 12).tolist()}, "coupling": {"scalar": CHAIN_J.tolist()}},
        bas={**NONE, "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 1},
        det="1H", offset=0,
    ),
    "exch_sym": {**EXCHANGE, "inter": {"zeeman": {"scalar": [0.0, 3.0]}, "chem": {
        "parts": [[0], [1]], "rates": [[-2e3, 2e3], [2e3, -2e3]], "concs": [1.0, 1.0]}}},
    "exch_asym": {**EXCHANGE, "inter": {"zeeman": {"scalar": [0.0, 3.0]}, "chem": {
        "parts": [[0], [1]], "rates": [[-5e2, 2e3], [5e2, -2e3]], "concs": [2e3, 5e2]}}},
    "hcn_filters": dict(
        sys={"isotopes": ["1H", "13C", "15N"], "magnet": 14.1},
        inter={"zeeman": {"scalar": [4.70, 55.0, 120.0]},
               "coupling": {"scalar": [[0, 140.0, -92.0], [0, 0, -15.0], [0, 0, 0]]}},
        bas={**NONE, "longitudinal": [["15N"]], "zero_quantum": [[[0, 1]]]},
        det="13C", offset=500,
    ),
}
OPS = {"E": "E", "Lz": "Lz", "Lx": "Lx", "Ly": "Ly", "Lp": "L+", "Lm": "L-"}
TYPES = ("left", "right", "comm", "acomm")

EXACT = dict(rtol=0, atol=0)
OPERATOR_TOL = dict(rtol=1e-14, atol=0)
HAMILTONIAN_TOL = dict(rtol=1e-12, atol=0)


@pytest.fixture(params=sorted(CASES))
def case(request):
    return request.param


@pytest.fixture
def fx(case):
    path = find_case(f"liouv_{case}")
    if path is None:
        pytest.skip(f"fixture liouv_{case} not available")
    return load_fixture(path)


@pytest.fixture
def spin_system(case):
    spec = CASES[case]
    return spinpy.basis(spinpy.create(spec["sys"], spec["inter"]), spec["bas"])


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def column(x):
    return dense(x).ravel()


def test_case_definitions_match_fixture(fx, case):
    inputs, spec = fx.meta["inputs"], CASES[case]
    assert inputs["sys"]["isotopes"] == spec["sys"]["isotopes"]
    assert inputs["sys"]["magnet"] == spec["sys"]["magnet"]
    assert inputs["bas"]["approximation"] == spec["bas"]["approximation"]
    assert inputs["det"] == spec["det"] and inputs["offset"] == spec["offset"]
    np.testing.assert_allclose(inputs["inter"]["zeeman"]["scalar"], spec["inter"]["zeeman"]["scalar"])
    if "chem" in spec["inter"]:
        np.testing.assert_array_equal(np.array(inputs["inter"]["chem"]["parts"]) - 1,
                                      np.array(spec["inter"]["chem"]["parts"]).ravel())
        np.testing.assert_array_equal(inputs["inter"]["chem"]["rates"], spec["inter"]["chem"]["rates"])


def test_basis_descriptor(fx, spin_system):
    bas = spin_system.basis
    assert dense(fx["/bas/basis"]).shape == bas.basis.shape
    np.testing.assert_array_equal(dense(fx["/bas/basis"]), bas.basis)
    np.testing.assert_array_equal(column(fx["/bas/tot_proj"]), bas.tot_proj)
    np.testing.assert_array_equal(column(fx["/bas/tot_cord"]), bas.tot_cord)


def test_product_tables(fx, spin_system):
    for mult in sorted(set(int(m) for m in spin_system.mults)):
        left, right = ist_product_table(mult)
        tol = EXACT if mult == 2 else dict(rtol=1e-14, atol=1e-15)
        assert_matches(left, fx[f"/bas/lpst_{mult}"], **tol, name=f"lpst_{mult}")
        assert_matches(right, fx[f"/bas/rpst_{mult}"], **tol, name=f"rpst_{mult}")


@pytest.mark.parametrize("operator_type", TYPES)
def test_operators(fx, case, spin_system, operator_type):
    det = CASES[case]["det"]
    for key, label in OPS.items():
        ref = fx[f"/ops/{key}_{operator_type}"]
        for n in range(spin_system.nspins):
            actual = spinpy.operator(spin_system, [label], [n], operator_type)
            assert_matches(actual, ref[0, n], **OPERATOR_TOL, name=f"{label}({n}) {operator_type}")
            assert actual.count_nonzero() == ref[0, n].count_nonzero(), f"{label}({n}) {operator_type} sparsity"
    if spin_system.nspins > 1:
        assert_matches(spinpy.operator(spin_system, ["Lz", "L+"], [0, 1], operator_type),
                       fx[f"/ops/LzLp_12_{operator_type}"], **OPERATOR_TOL)
    if operator_type == "comm":
        assert_matches(spinpy.operator(spin_system, "Lz", det), fx["/ops/Lz_det_comm"], **OPERATOR_TOL)
    if operator_type == "left":
        assert_matches(spinpy.operator(spin_system, "L+", det, "left"), fx["/ops/Lp_det_left"], **OPERATOR_TOL)


def test_states(fx, case, spin_system):
    det = CASES[case]["det"]
    for label, key in (("Lz", "Lz"), ("L+", "Lp")):
        assert_matches(spinpy.state(spin_system, label, det), column(fx[f"/states/{key}_exact"]), **OPERATOR_TOL)
        assert_matches(spinpy.state(spin_system, label, det, "chem"), column(fx[f"/states/{key}_chem"]), **OPERATOR_TOL)
        if f"/states/{key}_cheap" in fx:
            assert_matches(spinpy.state(spin_system, label, det, "cheap"), column(fx[f"/states/{key}_cheap"]), **OPERATOR_TOL)
    if spin_system.nspins > 1:
        assert_matches(spinpy.state(spin_system, ["Lz", "L+"], [0, 1]), column(fx["/states/LzLp_12_exact"]), **OPERATOR_TOL)


def test_hamiltonian(fx, case, spin_system):
    nmr = spinpy.assume(spin_system, "nmr")
    H = spinpy.hamiltonian(nmr)
    assert_matches(H, fx["/ham/H"], **HAMILTONIAN_TOL, name="H")
    assert_matches(spinpy.hamiltonian(spinpy.assume(spin_system, "nmr", "zeeman")), fx["/ham/H_zeeman"], **HAMILTONIAN_TOL)
    assert_matches(spinpy.hamiltonian(spinpy.assume(spin_system, "nmr", "couplings")), fx["/ham/H_coupling"], **HAMILTONIAN_TOL)
    assert_matches(spinpy.hamiltonian(nmr, "left"), fx["/ham/H_left"], **HAMILTONIAN_TOL, name="H_left")
    assert_matches(spinpy.hamiltonian(nmr, "right"), fx["/ham/H_right"], **HAMILTONIAN_TOL, name="H_right")
    pars = {"spins": [CASES[case]["det"]], "offset": [CASES[case]["offset"]]}
    assert_matches(spinpy.frqoffset(spin_system, H, pars), fx["/ham/H_off"], **HAMILTONIAN_TOL, name="H_off")


def test_kinetics(fx, spin_system):
    assert_matches(spinpy.kinetics(spin_system), fx["/kin/K"], **EXACT, name="K")


def test_filters(fx, case, spin_system):
    det = CASES[case]["det"]
    v = column(fx["/filters/input"])
    assert_matches(spinpy.correlation(spin_system, v, [1, 2]), column(fx["/filters/corr_12"]), **EXACT)
    assert_matches(spinpy.correlation(spin_system, v, 1, det), column(fx["/filters/corr_1_det"]), **EXACT)
    assert_matches(spinpy.coherence(spin_system, v, [(det, 1)]), column(fx["/filters/coh_det_p1"]), **EXACT)
    assert_matches(spinpy.coherence(spin_system, v, [("all", 0)]), column(fx["/filters/coh_all_0"]), **EXACT)
