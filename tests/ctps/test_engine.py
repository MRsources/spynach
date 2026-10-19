"""Engine against the 1998 paper: pathway count, phase-cycle, gradient and diffusion factors of Fig. 4."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ctps
from ctps.engine.gradient import diffusion_attenuation, gradient_attenuation, k_vectors
from ctps.engine.phase import phase_cycle_factor
from ctps.model import PhaseCycle, Sample
from ctps.simulate import simulate

ROOT = Path(__file__).resolve().parents[2]


def test_ctps_does_not_import_spynach():
    """In a fresh interpreter (other tests import spynach first), importing ctps must not pull spynach in."""
    code = "import sys, ctps, ctps.simulate, ctps.engine.pulse; sys.exit(1 if 'spynach' in sys.modules else 0)"
    run = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": "src"}, cwd=ROOT)
    assert run.returncode == 0


def test_ste_1998_fig4_table(ste_1998):
    fx, sequence, cycle = ste_1998
    result = simulate(sequence, cycle, wanted=fx["wanted"], epsilon=1e-9, cutoff=0)
    assert result.n_enumerated == fx["n_pathways"]
    rows = {tuple(int(v) for v in r): i for i, r in enumerate(result.orders)}
    for row in fx["rows"]:
        i = rows[tuple(row["pathway"])]
        assert abs(abs(result.pc[i]) - row["pc"]) < 1.5e-4, row
        if row["pc_phase_deg"] is not None:
            phase = np.degrees(np.angle(result.pc[i])) % 360
            assert abs((phase - row["pc_phase_deg"] + 180) % 360 - 180) < 1e-6, row
        assert abs(result.grad[i] - row["grad"]) < 1.5e-4, row
    # Diffusion: the two wanted pathways do not depend on the delay after the last gradient
    for row in fx["rows"][:2]:
        i = rows[tuple(row["pathway"])]
        assert abs(result.diff[i] - row["diff"]) < 1.5e-4, row
    assert result.wanted.sum() == 2
    assert result.table()[0]["rank"] == 1


@pytest.mark.xfail(
    reason="pathways 3-8 need the delay after the last gradient, which CCCP.m defines (source pending)",
    strict=True,
)
def test_ste_1998_fig4_diffusion_of_leaking_pathways(ste_1998):
    fx, sequence, cycle = ste_1998
    result = simulate(sequence, cycle, epsilon=1e-9, cutoff=0)
    rows = {tuple(int(v) for v in r): i for i, r in enumerate(result.orders)}
    for row in fx["rows"][2:]:
        assert abs(result.diff[rows[tuple(row["pathway"])]] - row["diff"]) < 1.5e-4


def test_diffusion_stejskal_tanner_check(ste_1998):
    """1998 Eq. 40: exp[D (γGδ)² δ/3] = 1.0096 for the listed parameters."""
    fx = ste_1998[0]["diffusion_check"]
    k = np.array([[[0.0, 0.0, fx["gamma"] * fx["G"] * fx["delta_s"]]]])
    f = diffusion_attenuation(k, np.array([fx["delta_s"] / 3]), fx["D"])
    assert abs(1 / f[0] - fx["expected_exp_D_gammaGdelta2_delta_over_3"]) < 1e-4


def test_gradient_attenuation_envelopes():
    sample = Sample(z_max=5e-3, x_max=2.5e-3)
    small = gradient_attenuation(np.array([[0.0, 0.0, 1.0 / 5e-3]]), sample)  # u = 1 < pi/2: |sinc 1|
    assert np.isclose(small[0], abs(np.sin(1.0) / 1.0))
    large = gradient_attenuation(np.array([[0.0, 0.0, 10.0 / 5e-3]]), sample)  # u = 10 > pi/2: 1/u
    assert np.isclose(large[0], 0.1)
    assert np.isclose(
        gradient_attenuation(np.array([[0.0, 0.0, 10.0 / 5e-3]]), sample, envelope=False)[0], abs(np.sin(10.0) / 10.0)
    )
    u = 16.72
    transverse = gradient_attenuation(np.array([[u / 2.5e-3, 0.0, 0.0]]), sample)
    assert np.isclose(transverse[0], 2 * np.sqrt(2 / np.pi) * u ** (-1.5))
    assert gradient_attenuation(np.zeros((1, 3)), sample)[0] == 1.0


def test_phase_cycle_integer_equals_direct_sum():
    rng = np.random.default_rng(0)
    dp = rng.integers(-3, 4, size=(200, 5))
    for base in (4, 8, 6):
        phases = [rng.integers(0, base, size=rng.integers(1, 9)).tolist() for _ in range(5)]
        cycle = PhaseCycle.explicit(phases, rng.integers(0, base, size=4).tolist(), base)
        exact = phase_cycle_factor(dp, cycle, integer=True)
        direct = phase_cycle_factor(dp, cycle, integer=False)
        np.testing.assert_allclose(exact, direct, atol=1e-12)


def test_chunking_and_workers_give_the_same_result(ste_1998):
    fx, sequence, cycle = ste_1998
    whole = simulate(sequence, cycle, epsilon=1e-9, cutoff=0)
    chunked = simulate(sequence, cycle, epsilon=1e-9, cutoff=0, chunk=5000)
    parallel = simulate(sequence, cycle, epsilon=1e-9, cutoff=0, chunk=20000, workers=2)
    for other in (chunked, parallel):
        assert len(other) == len(whole)
        key = lambda r: sorted(map(tuple, np.column_stack([r.orders, np.round(r.amplitude, 12)]).tolist()))  # noqa: E731
        assert key(other) == key(whole)


def test_k_vectors_cumulative():
    gamma_star = np.array([[1.0, 2.0, 0.0]])
    areas = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
    K = k_vectors(gamma_star, areas)
    np.testing.assert_allclose(K[0, :, 2], [1.0, 3.0, 3.0])


def test_version_and_exports():
    assert ctps.__version__ and callable(ctps.simulate)
