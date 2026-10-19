"""The cogwheel search reproduces the published minimal cycles and the HCL construction."""

import numpy as np
import pytest

from ctps.engine.phase import cogwheel_selects
from ctps.model import Sequence
from ctps.optimize import coverage, find_cycle, hcl_cycle, pathway_jumps

from .conftest import load_fixture

CASES = {c["name"]: c for c in load_fixture("cogwheel_cycles.json")["cases"]}


def selects_exactly(sequence: Sequence, solution, wanted) -> bool:
    orders, dp = pathway_jumps(sequence)
    selected = cogwheel_selects(dp, solution.N, np.array(solution.nu), solution.nu_rec)
    return {tuple(int(v) for v in r) for r in orders[selected]} == {tuple(w) for w in wanted}


def test_hcl_construction_matches_levitt():
    """2004 Eq. 23-26 on 3Q-MAS: q = (1, 3), Q = 3, N = 23, Δν = (3, 1), COG23(0,3,4;14)."""
    s = hcl_cycle([3, 1], [3, 3])
    assert (s.N, s.nu, s.nu_rec) == (23, (0, 3, 4), 14)
    assert hcl_cycle([5, 1], [5, 5]).N == 57  # 5Q-MAS
    assert hcl_cycle([5, 3, 1], [5, 5, 5]).N == 201  # split-t1 5Q3QMAS


@pytest.mark.parametrize(
    "p_wanted, p_max, N_min",
    [([3, 1], 3, 23), ([5, 1], 5, 57), ([1, -1, 1, -1, 1], 1, 11)],
    ids=["3Q-MAS", "5Q-MAS", "TOSS/PASS"],
)
def test_minimal_cycles_of_levitt_2002(p_wanted, p_max, N_min):
    sequence = Sequence.simple(len(p_wanted) + 1, p_max)
    wanted = [(0, *p_wanted, -1)]
    result = find_cycle(sequence, wanted, N_min=2)
    assert result.best is not None and result.best.N == N_min
    assert result.best.survivors == 0
    assert selects_exactly(sequence, result.best, wanted)


def test_all_22_solutions_at_23():
    sequence = Sequence.simple(3, 3)
    result = find_cycle(sequence, [(0, 3, 1, -1)], N_min=23, N_max=23, all_solutions=True)
    # 22 (Δν_AB, Δν_BC) pairs; with ν_1 fixed at 0 and the orbit reduction, the search keeps one representative
    # per unit multiple, so re-expand the orbit and count
    N = 23
    orbit = set()
    for s in result.solutions:
        for c in range(1, N):
            orbit.add(tuple((c * np.array(s.nu)) % N))
    assert len(orbit) == 22


def test_hughes_2004_cycles():
    seq = Sequence.simple(4, 5)
    result = find_cycle(seq, [(0, 5, 3, 1, -1)], N_min=190, N_max=201)
    assert result.best is not None and result.best.N <= 201 and result.best.survivors == 0
    dq = Sequence.simple(5, 2)
    wanted = [(0, 2, -2, 2, 0, -1), (0, -2, 2, -2, 0, -1)]
    result = find_cycle(dq, wanted, N_min=2, N_max=36)
    assert result.best is not None and result.best.N <= 36
    assert selects_exactly(dq, result.best, wanted)


def test_jerschow_kumar_2003_cases_reproduced_or_beaten():
    for name in (
        "Jerschow & Kumar 2003 MQMAS COG11(3,1,0;4), odd and zero orders, perfect detector",
        "Jerschow & Kumar 2003 MQMAS COG19(3,8,1;9), odd and zero orders, detection -1,0,+1",
        "Jerschow & Kumar 2003 SCAM-STMAS COG21(9,1,0,0;12), perfect echo at the end",
    ):
        case = CASES[name]
        periods = {int(k): v for k, v in case.get("periods", {}).items()}
        seq = Sequence.simple(
            len(case["nu"]),
            max(abs(v) for v in case["orders"]),
            end=case["detect"],
            orders=case["orders"],
            periods=periods,
        )
        result = find_cycle(seq, case["wanted"], N_min=2, N_max=case["N"])
        assert result.best is not None, name
        assert result.best.N <= case["N"], name
        assert selects_exactly(seq, result.best, case["wanted"]), name


def test_n_opt_grid_of_hughes_2004_table():
    """Two intervals with p_max = 3: N_opt equals the HCL prediction except for {0,2,2,-1} (22, not 20)."""
    expected = {
        (0, 0): 16, (0, 1): 20, (0, 2): 24, (0, 3): 28,
        (1, 0): 20, (1, 1): 21, (1, 2): 22, (1, 3): 23,
        (2, 0): 24, (2, 1): 22, (2, 2): 22, (2, 3): 18,
        (3, 0): 28, (3, 1): 23, (3, 2): 18, (3, 3): 13,
    }  # fmt: skip
    seq = Sequence.simple(3, 3)
    for (p1, p2), n_opt in expected.items():
        result = find_cycle(seq, [(0, p1, p2, -1)], N_min=2, N_max=30)
        assert result.best is not None and result.best.N == n_opt, (p1, p2, result.best)
        predicted = hcl_cycle([p1, p2], [3, 3]).N
        assert predicted == n_opt or (p1, p2) == (2, 2)


def test_coverage_mode_slaser_landheer_2020():
    """Landheer & Juchem 2020 Table 1: sLASER, one spin, 80 unwanted pathways; COG8 cancels 79, COG16 all."""
    seq = Sequence.simple(5, 1)
    wanted = [(0, -1, 1, -1, 1, -1)]
    at8 = coverage(seq, wanted, 8)
    assert at8.best is not None and at8.best.survivors <= 1
    at16 = coverage(seq, wanted, 16)
    assert at16.best is not None and at16.best.survivors == 0


def test_impossible_pathways_are_reported():
    seq = Sequence.simple(2, 1)
    result = find_cycle(seq, [(0, 1, -1)], N_min=2, N_max=8)
    assert result.best is not None and not result.impossible


def test_workers_agree():
    seq = Sequence.simple(3, 3)
    a = find_cycle(seq, [(0, 3, 1, -1)], N_min=2, N_max=30)
    b = find_cycle(seq, [(0, 3, 1, -1)], N_min=2, N_max=30, workers=2)
    assert a.best.N == b.best.N == 23


def test_crusher_scheme_steam_and_press_reach_dotcops_min_distance():
    """DOTCOPS Table 2: STEAM and PRESS with the middle crusher allowed twice the area reach min D = sqrt(3)."""
    from ctps.optimize import crusher_scheme

    for wanted in ((0, 1, 0, -1), (0, -1, 1, -1)):  # STEAM, PRESS (orders after each of three pulses)
        seq = Sequence.simple(3, 1)
        bounds = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0]], dtype=float)
        # the three crushers follow the three pulses: periods 1, 2, 3; DOTCOPS has no crusher after pulse 3
        bounds = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 1]], dtype=float)
        scheme = crusher_scheme(seq, [wanted], bounds=bounds, starts=60)
        assert scheme["min_D"] >= np.sqrt(3) - 1e-3, (wanted, scheme["min_D"])
        # the wanted pathway is refocused
        assert np.linalg.norm(np.array(wanted) @ scheme["areas"]) < 1e-6


def test_pathway_weights_match_simulate_and_shorten_the_cycle():
    """pathway_weights gives every pathway's pre-cycle weight; with it the PFG stimulated echo needs a shorter cycle."""
    from ctps.model import Period, Sample
    from ctps.optimize import pathway_weights

    pfg = Sequence.simple(7, 1, periods={3: [0], 6: [0]})
    periods = list(pfg.periods)
    for i, (axis, amp) in {
        1: ("z", 0.25),
        2: ("z", -0.25),
        3: ("x", 0.025),
        4: ("z", 0.25),
        5: ("z", -0.25),
        6: ("y", 0.025),
    }.items():
        periods[i] = Period.with_gradient(2e-3, axis, amp, 1e-3)
    seq = Sequence(pfg.nuclei, pfg.pulses, tuple(periods), pfg.allowed, Sample(5e-3, 2.5e-3))
    orders, g = pathway_weights(seq)
    assert len(orders) == 81 and np.all((g >= 0) & (g <= 1))
    wanted = [(0, -1, 1, 0, 1, -1, 0, -1), (0, 1, -1, 0, -1, 1, 0, -1)]
    plain = find_cycle(seq, wanted)
    weighted = find_cycle(seq, wanted, weights=g, tolerance=0.05)
    assert plain.best.N == 8 and weighted.best.N < 8 and weighted.best.residual <= 0.05
