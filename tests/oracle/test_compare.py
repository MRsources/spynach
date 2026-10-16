import numpy as np
import pytest
import scipy.sparse as sps

from spynach.oracle import assert_matches, compare


def test_identical_passes_with_zero_tolerance():
    a = np.arange(6.0).reshape(2, 3) + 1j
    result = compare(a, a.copy(), rtol=0, atol=0)
    assert result.passed and result.err_norm == 0 and result.rel_err == 0


def test_normwise_threshold():
    e = np.array([1.0, 0.0, 0.0, 0.0])
    a = e + np.array([0, 1e-9, 0, 0])
    assert compare(a, e, rtol=1e-8, atol=0).passed
    assert not compare(a, e, rtol=1e-10, atol=0).passed
    assert compare(a, e, rtol=0, atol=1e-8).passed


def test_mixed_sparse_dense():
    d = np.diag([1.0, 2.0, 3.0]) + 0j
    s = sps.csr_array(d)
    assert compare(s, d, rtol=0, atol=0).passed
    assert compare(d, s, rtol=0, atol=0).passed
    s2 = s + sps.csr_array(([1e-3], ([0], [2])), shape=(3, 3))
    result = compare(s2, s, rtol=1e-6, atol=0)
    assert not result.passed and result.max_abs_err == pytest.approx(1e-3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare(np.zeros((3, 1)), np.zeros(3), rtol=1, atol=1)


def test_nan_fails():
    a = np.array([1.0, np.nan])
    assert not compare(a, np.array([1.0, 0.0]), rtol=1, atol=1).passed


def test_zero_reference():
    assert compare(np.zeros(3), np.zeros(3), rtol=0, atol=0).rel_err == 0
    assert compare(np.ones(3), np.zeros(3), rtol=1, atol=0).rel_err == float("inf")


def test_cell_rejected():
    with pytest.raises(TypeError):
        compare(np.empty(2, dtype=object), np.empty(2, dtype=object), rtol=0, atol=0)


def test_assert_matches_message():
    with pytest.raises(AssertionError, match="FAIL H"):
        assert_matches(np.ones(2), np.zeros(2), rtol=0, atol=0, name="H")
