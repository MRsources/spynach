"""Oracle harness: load MATLAB/Spinach reference fixtures and compare against them."""

from respinach.oracle.compare import Comparison, assert_matches, compare
from respinach.oracle.fixture import Fixture, find_case, fixture_root, load_fixture

__all__ = [
    "Comparison",
    "Fixture",
    "assert_matches",
    "compare",
    "find_case",
    "fixture_root",
    "load_fixture",
]
