"""Tests of spinpy.molecules: the strychnine spin system of Spinach's etc/molecules/strychnine.m."""

import numpy as np
import pytest

import spinpy
from spinpy.molecules import strychnine


def test_strychnine_protons():
    sys_, inter = strychnine(["1H"])
    assert sys_["isotopes"] == ["1H"] * 22
    assert "magnet" not in sys_                          # set by the caller, as in Spinach
    assert len(inter["zeeman"]["scalar"]) == 22
    assert inter["zeeman"]["scalar"][3] == 8.092         # H4, the most deshielded proton
    J = np.asarray(inter["coupling"]["scalar"])
    assert J.shape == (22, 22)
    assert np.count_nonzero(J) == 30 and not np.tril(J).any()   # upper triangle only
    assert np.asarray(inter["coordinates"]).shape == (22, 3)
    assert np.isclose(np.linalg.norm(np.subtract(inter["coordinates"][20], inter["coordinates"][21])), 1.770, atol=5e-3)   # geminal H23a-H23b


def test_strychnine_basis_size():
    sys_, inter = strychnine(["1H"])
    sys_.update(magnet=5.9, tols={"prox_cutoff": 4.0})
    ss = spinpy.basis(spinpy.create(sys_, inter),
                      {"formalism": "sphten-liouv", "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 3})
    assert ss.basis.nstates == 12316                     # noesy_strychnine.m basis, as recorded in README.md


def test_strychnine_heteronuclei_refused():
    with pytest.raises(NotImplementedError, match="13C"):
        strychnine(["1H", "13C"])
    strychnine(("1H",))                                  # any sequence of the one isotope is fine
