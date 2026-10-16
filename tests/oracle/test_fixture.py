import numpy as np
import pytest
import scipy.sparse as sps

from spynach.oracle import find_case, load_fixture
from tests.oracle.layout import write_case


def matlab_probe_dense():
    # reshape(1:24,[2 3 4])+1i*reshape(101:124,[2 3 4]) in MATLAB (column-major)
    return np.arange(1, 25).reshape((2, 3, 4), order="F") + 1j * np.arange(101, 125).reshape((2, 3, 4), order="F")


def make_cell():
    cell = np.empty((2, 2), dtype=object)
    cell[0, 0] = np.array([[1.0, 2.0]])
    cell[0, 1] = sps.csr_array(np.eye(2))
    cell[1, 0] = np.zeros((0, 0))
    cell[1, 1] = np.array([[7.0]])
    return cell


@pytest.fixture
def case(tmp_path):
    sparse = sps.csr_array(([5 + 1j, 7, -2j], ([0, 2, 1], [1, 0, 3])), shape=(3, 4))
    write_case(
        tmp_path / "tag" / "case",
        {
            "/probe/dense": matlab_probe_dense(),
            "/probe/column": np.array([[1.0], [2.0], [3.0]]),
            "/probe/sparse": sparse,
            "/probe/cell": make_cell(),
            "/empty/sparse": sps.csr_array((5, 5)),
            "/real/matrix": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
        {"schema_version": 1},
    )
    return tmp_path


def test_round_trip(case):
    fx = load_fixture(case / "tag" / "case")
    assert fx.meta["schema_version"] == 1
    assert sorted(fx.keys()) == sorted(
        ["/probe/dense", "/probe/column", "/probe/sparse", "/probe/cell", "/empty/sparse", "/real/matrix"]
    )
    dense = fx["/probe/dense"]
    assert dense.dtype == np.complex128 and dense.shape == (2, 3, 4)
    np.testing.assert_array_equal(dense, matlab_probe_dense())
    assert fx["/probe/column"].shape == (3, 1)
    assert fx["/real/matrix"].dtype == np.float64

    sparse = fx["/probe/sparse"]
    assert sps.issparse(sparse) and sparse.shape == (3, 4)
    expected = np.zeros((3, 4), dtype=complex)
    expected[0, 1], expected[2, 0], expected[1, 3] = 5 + 1j, 7, -2j
    np.testing.assert_array_equal(sparse.toarray(), expected)

    empty = fx["/empty/sparse"]
    assert sps.issparse(empty) and empty.shape == (5, 5) and empty.nnz == 0

    cell = fx["/probe/cell"]
    assert cell.shape == (2, 2) and cell.dtype == object
    np.testing.assert_array_equal(cell[0, 0], [[1.0, 2.0]])
    np.testing.assert_array_equal(cell[0, 1].toarray(), np.eye(2))
    assert cell[1, 0].shape == (0, 0)
    np.testing.assert_array_equal(cell[1, 1], [[7.0]])

    assert "/probe/dense" in fx and "/probe" not in fx
    with pytest.raises(KeyError):
        fx["/missing"]


def test_find_case(case, monkeypatch):
    monkeypatch.setenv("SPYNACH_FIXTURES", str(case))
    assert find_case("case") == case / "tag" / "case"
    assert find_case("case", tag="tag") == case / "tag" / "case"
    assert find_case("absent") is None
    write_case(case / "tag2" / "case", {}, {})
    with pytest.raises(LookupError):
        find_case("case")
