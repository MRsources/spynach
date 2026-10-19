"""Non-ideal pulse weights against Jerschow, J. Chem. Phys. 113, 979 (2000) and the exact rotation matrices."""

import numpy as np
import pytest

from ctps.engine.pulse import f_coefficient, rotation_matrix, transfer_matrix, wigner_d


@pytest.mark.parametrize("l", [1, 2, 3, 4])
def test_wigner_d_matches_exponential_of_jy(l):
    for beta in (0.3, 1.0, 2.5):
        np.testing.assert_allclose(np.abs(wigner_d(l, beta)), rotation_matrix(l, beta, 0.0), atol=1e-12)


def test_spin_half_elements():
    d = wigner_d(1, 0.7)
    beta = 0.7
    assert np.isclose(d[2, 2], (1 + np.cos(beta)) / 2)  # d^1_{11}
    assert np.isclose(d[2, 0], (1 - np.cos(beta)) / 2)  # d^1_{1,-1}
    assert np.isclose(d[2, 1], -np.sin(beta) / np.sqrt(2))  # d^1_{10}
    np.testing.assert_allclose((d**2).sum(axis=1), 1.0)


@pytest.mark.parametrize("l", [1, 2, 3, 4, 6, 8])
def test_f_coefficients_are_the_small_angle_limit(l):
    """Eq. 13: d^l_{m'm}(β) -> f^l_{m'm} (β/2)^|m - m'| as β -> 0."""
    beta = 1e-3
    d = wigner_d(l, beta)
    for mp in range(-l, l + 1):
        for m in range(-l, l + 1):
            expected = f_coefficient(l, mp, m) * (beta / 2) ** abs(m - mp)
            # the next term of the series is of relative order l^2 beta^2
            assert abs(d[mp + l, m + l] - expected) < 2 * l**2 * beta**2 * max(abs(expected), 1e-30) + 1e-14


def test_largest_rank_bounds_every_transfer():
    """Eq. 19: for a spin j the leading coefficient |f^l_{m'm}| is largest at l = 2j, for every l <= 2j (to l = 8)."""
    for top in range(1, 9):
        for l in range(1, top + 1):
            for mp in range(-l, l + 1):
                for m in range(-l, l + 1):
                    assert abs(f_coefficient(l, mp, m)) <= abs(f_coefficient(top, mp, m)) + 1e-12


def test_perfect_180_inverts_only():
    U = transfer_matrix("perfect_180", 3)
    for p in range(-3, 4):
        column = U[:, p + 3]
        assert column[3 - p] == 1.0 and column.sum() == 1.0


def test_imperfect_180_leading_terms_match_exact_rotation():
    """Eq. 14-16, 20: near π the transfer p -> p' scales as |f^l_{p',-p}| |u|^|p+p'|, |u|² = (β/2)² + ε²."""
    l, beta, eps = 2, 2e-3, 3e-3
    leading = transfer_matrix("imperfect_180", 2, l=l, beta=beta, epsilon=eps)
    exact = transfer_matrix("imperfect_180", 2, l=l, beta=beta, epsilon=eps, exact=True)
    for p in range(-2, 3):
        for pp in range(-2, 3):
            if pp == -p:
                assert abs(leading[pp + 2, p + 2] - 1.0) < 1e-12 and abs(exact[pp + 2, p + 2] - 1.0) < 1e-4
            else:
                assert abs(leading[pp + 2, p + 2] - exact[pp + 2, p + 2]) < 2e-3 * exact[pp + 2, p + 2] + 1e-12


def test_small_flip_leading_terms_match_exact_rotation():
    l, beta = 3, 5e-3
    leading = transfer_matrix("small_flip", 3, l=l, beta=beta)
    exact = transfer_matrix("small_flip", 3, l=l, beta=beta, exact=True)
    assert np.all(np.abs(leading - exact) < 2e-4 * exact + 1e-12)


def test_table_iv_three_spins_half():
    """JCP 2000 Table IV: four pulses on three spins-1/2, d = β/2; a few tabulated bounds."""
    d = 0.01
    U = transfer_matrix("small_flip", 3, l=1, n_spins=3, beta=2 * d)
    at = lambda pp, p: U[pp + 3, p + 3]  # noqa: E731
    assert np.isclose(at(3, 2), np.sqrt(2) * d)  # (2,3,3,3,3): sqrt(2) d
    assert np.isclose(at(2, -1) * at(3, 2), 4 * d**4)  # (-1,2,3,3,3): 4 d^4
    assert np.isclose(at(3, -1), 2 * d**4)  # (-1,3,3,3,3): 2 d^4
    assert np.isclose(at(0, 3) * at(3, 0) * at(0, 3) * at(3, 0), 64 * d**12)  # (3,0,3,0,3): 64 d^12


def test_legacy_cccp_model():
    U = transfer_matrix("cccp_1998", 2, epsilon=0.1)
    assert np.isclose(U[0, 4], (1 - 0.1**2 / 2) ** 2)  # +2 -> -2
    assert np.isclose(U[4, 4], 0.1**2)  # +2 -> +2
    assert U[2, 2] == 1.0 and U[3, 4] == 0.0


def test_pulse_factor_in_a_sequence():
    from ctps import AllowedOrders, Nucleus, Period, PhaseCycle, Pulse, Sequence, simulate

    H = Nucleus("1H")
    allowed = AllowedOrders.standard(1, 3)
    seq = Sequence(
        (H,), (Pulse("1H"), Pulse("1H", "imperfect_180", beta=0.1)), (Period(), Period(), Period()), {"1H": allowed}
    )
    result = simulate(seq, PhaseCycle.explicit([[0], [0]], [0]), epsilon=0, cutoff=0)
    rows = {tuple(int(v) for v in r): float(result.pulse[i]) for i, r in enumerate(result.orders)}
    assert rows[(0, 1, -1)] == 1.0  # inverted
    assert np.isclose(rows[(0, -1, -1)], 0.05**2)  # non-inverted spin-1/2: f = 1, |u|^2
    assert np.isclose(rows[(0, 0, -1)], np.sqrt(2) * 0.05)  # 0 -> -1 at first order
