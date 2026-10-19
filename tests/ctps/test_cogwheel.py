"""Published cogwheel cycles select exactly their stated pathways under the 1998/2003 convention."""

import numpy as np
import pytest

from ctps.engine.phase import cogwheel_selects, receiver_winding
from ctps.model import PhaseCycle

from .conftest import all_pathways, load_fixture

CASES = load_fixture("cogwheel_cycles.json")["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_published_cycle_selects_its_pathways(case):
    if "skip" in case:
        pytest.skip(case["skip"])
    paths = all_pathways(case)
    dp = np.diff(paths, axis=1)
    selected = cogwheel_selects(dp, case["N"], np.array(case["nu"]), case["nu_rec"])
    got = {tuple(int(v) for v in row) for row in paths[selected]}
    assert got == {tuple(w) for w in case["wanted"]}
    # the receiver winding number follows from any wanted pathway
    assert receiver_winding(np.diff(case["wanted"][0]), case["N"], np.array(case["nu"])) == case["nu_rec"] % case["N"]
    # and the explicit cycle built from (N; nu; nu_rec) gives |s| = 1 on the wanted pathways, 0 elsewhere
    cycle = PhaseCycle.cogwheel(case["N"], case["nu"], case["nu_rec"])
    from ctps.engine.phase import phase_cycle_factor

    s = np.abs(phase_cycle_factor(dp, cycle))
    np.testing.assert_allclose(s[selected], 1.0, atol=1e-12)
    assert np.all(s[~selected] < 1e-12)


def test_levitt_3qmas_has_22_solutions_at_23():
    """Levitt 2002 p. 303: exactly 22 (Δν_AB, Δν_BC) pairs select {0,3,1,-1} at N = 23, the orbit of (3, 1)."""
    case = next(c for c in CASES if "COG23(0,3,4;14)" in c["name"])
    paths = all_pathways(case)
    dp = np.diff(paths, axis=1)
    wanted = {tuple(w) for w in case["wanted"]}
    solutions = []
    for a in range(23):
        for b in range(23):
            nu = np.array([0, a, a + b]) % 23  # Δν_AB = a, Δν_BC = b
            rec = receiver_winding(np.diff(case["wanted"][0]), 23, nu)
            got = {tuple(int(v) for v in row) for row in paths[cogwheel_selects(dp, 23, nu, rec)]}
            if got == wanted:
                solutions.append((a, b))
    assert len(solutions) == 22
    assert (3, 1) in solutions and (20, 22) in solutions  # (3, 1) and its negative
