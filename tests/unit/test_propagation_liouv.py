"""Fixture-free tests of Liouville-space propagation: step, reduction, evolution, exchange."""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sps

import respinach
from respinach.oracle import assert_matches
from respinach.reduction import binpack, path_trace, zte

from .test_liouvillian import CASES, dense

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
PARS = {"spins": ["1H"], "offset": [660.0], "sweep": 500.0, "npoints": 64}


def system(isotopes, inter=None, sys=None, **kwargs):
    return respinach.create(
        {"isotopes": isotopes, "magnet": 14.1, **(sys or {})}, inter if inter is not None else kwargs
    )


def liouvillian(ss, pars=PARS):
    nmr = respinach.assume(ss, "nmr")
    return respinach.frqoffset(nmr, respinach.hamiltonian(nmr), pars)


def test_step_matches_the_exponential_propagator():
    ls = respinach.basis(system(*CASES["hd_2spin"]), LIOUV)
    L = liouvillian(ls)
    rho = respinach.state(ls, "L+", "1H") + 0.3 * respinach.state(ls, ["Lz", "T2,1"], [0, 1])
    for dt in (1e-3, 2e-2):  # the second needs several Taylor substeps
        P = sla.expm(-1j * dense(L) * dt)
        assert_matches(respinach.step(ls, L, rho, dt), P @ rho, rtol=1e-12, atol=0, name=f"step dt={dt}")
        stack = np.stack([rho, 2j * rho, np.roll(rho, 3)], axis=1)
        assert_matches(respinach.step(ls, L, stack, dt), P @ stack, rtol=1e-12, atol=0, name="stack")
        assert_matches(respinach.step(ls, [L, L], rho, dt), P @ rho, rtol=1e-12, atol=0, name="two-point")
        assert_matches(respinach.step(ls, [L, L, L], rho, dt), P @ rho, rtol=1e-12, atol=0, name="three-point")
    assert respinach.step(ls, L, rho, 0.0) is rho
    with pytest.raises(ValueError, match="dt is too long"):
        respinach.step(ls, L, rho, 1e3)


def test_step_in_hilbert_space():
    hs = respinach.basis(system(*CASES["ab_2spin"]), HILB)
    H = liouvillian(hs)
    rho = respinach.state(hs, "L+", "1H")
    P = sla.expm(-1j * dense(H) * 1e-3)
    assert_matches(respinach.step(hs, H, rho, 1e-3), P @ dense(rho) @ P.conj().T, rtol=1e-12, atol=0)
    big = respinach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1, "tols": {"small_matrix": 2}}, CASES["ab_2spin"][1]
    )
    hs_big = respinach.basis(big, HILB)
    assert_matches(respinach.step(hs_big, H, rho, 1e-3), P @ dense(rho) @ P.conj().T, rtol=1e-12, atol=0)


@pytest.mark.parametrize("case", ["ab_2spin", "hc_2spin", "hd_2spin"])
def test_liouville_fid_equals_hilbert_fid(case):
    """exact states carry 1/sqrt(dim) each, so the Liouville observable is the Hilbert one over dim."""
    ss = system(*CASES[case])
    hs, ls = respinach.basis(ss, HILB), respinach.basis(ss, LIOUV)
    dim = int(np.prod(ss.mults))
    fid_h = respinach.evolution(
        hs,
        liouvillian(hs),
        respinach.state(hs, "L+", "1H"),
        respinach.state(hs, "L+", "1H"),
        1 / PARS["sweep"],
        PARS["npoints"] - 1,
        "observable",
    )
    coil, rho0 = respinach.state(ls, "L+", "1H"), respinach.state(ls, "L+", "1H")
    fid_l = respinach.evolution(ls, liouvillian(ls), coil, rho0, 1 / PARS["sweep"], PARS["npoints"] - 1, "observable")
    assert fid_l.shape == (PARS["npoints"],)
    assert_matches(dim * fid_l, fid_h, rtol=1e-9, atol=0, name="fid")
    # Same through the context, with kinetics zero
    pars = {**PARS, "rho0": rho0, "coil": coil}
    assert_matches(dim * respinach.liquid(ls, respinach.acquire, pars, "nmr"), fid_h, rtol=1e-9, atol=0, name="liquid")


def test_outputs_and_stacks_are_consistent():
    ls = respinach.basis(system(*CASES["ab_2spin"]), LIOUV)
    L = liouvillian(ls)
    coil, rho0 = respinach.state(ls, "L+", "1H"), respinach.state(ls, "L+", "1H")
    dt, nsteps = 1 / PARS["sweep"], 20
    traj = respinach.evolution(ls, L, None, rho0, dt, nsteps, "trajectory")
    final = respinach.evolution(ls, L, None, rho0, dt, nsteps, "final")
    fid = respinach.evolution(ls, L, coil, rho0, dt, nsteps, "observable")
    assert traj.shape == (16, nsteps + 1) and final.shape == (16,)
    assert_matches(final, traj[:, -1], rtol=1e-11, atol=1e-14)
    assert_matches(fid, coil.conj() @ traj, rtol=1e-11, atol=1e-14)
    P = sla.expm(-1j * dense(L) * dt)
    assert_matches(final, np.linalg.matrix_power(P, nsteps) @ rho0, rtol=1e-10, atol=1e-14)
    stack = np.stack([rho0, respinach.state(ls, "Lz", "1H")], axis=1)
    fid_stack = respinach.evolution(ls, L, coil, stack, dt, nsteps, "observable")
    assert fid_stack.shape == (nsteps + 1, 2)
    assert_matches(fid_stack[:, 0], fid, rtol=1e-11, atol=1e-14)
    assert np.allclose(fid_stack[:, 1], 0)  # Lz never reaches L+ under a secular Hamiltonian
    final_stack = respinach.evolution(ls, L, None, stack, dt, nsteps, "final")
    assert_matches(final_stack[:, 0], final, rtol=1e-10, atol=1e-14)
    with pytest.raises(ValueError, match="stacks"):
        respinach.evolution(ls, L, None, stack, dt, nsteps, "trajectory")


def test_reductions_do_not_change_the_answer():
    ss_inter = CASES["hc_2spin"][1]
    results = {}
    for label, sys in (
        ("default", {}),
        ("no zte", {"disable": ["zte"]}),
        ("no dss", {"disable": ["dss"]}),
        ("nothing", {"disable": ["trajlevel"]}),
        ("krylov", {"enable": ["krylov"]}),
        ("path tracing", {"tols": {"merge_dim": 2}}),
    ):
        ls = respinach.basis(system(["1H", "13C"], ss_inter, sys=sys), LIOUV)
        coil = respinach.state(ls, "L+", "1H")
        results[label] = respinach.evolution(ls, liouvillian(ls), coil, coil, 1 / PARS["sweep"], 32, "observable")
    for label, fid in results.items():
        assert_matches(fid, results["default"], rtol=1e-9, atol=0, name=label)


def test_zte_and_path_trace():
    ls = respinach.basis(system(["1H", "13C"], CASES["hc_2spin"][1], sys={"tols": {"merge_dim": 2}}), LIOUV)
    L = liouvillian(ls)
    coil = respinach.state(ls, "L+", "1H")
    projector = zte(ls, L, coil)
    kept = projector.indices if sps.issparse(projector) else None
    # L+(H) evolves into L+(H)Lz(C) under the weak coupling and nowhere else
    assert projector.shape == (16, 2)
    rows = sorted(ls.basis.basis[kept].tolist())
    assert rows == [[1, 0], [1, 2]]
    # Path tracing of the full Liouvillian: components of the magnetisation-flow graph
    parts = path_trace(ls, L, coil)
    assert len(parts) == 1 and sorted(parts[0].indices.tolist()) == [4, 6]
    parts_all = path_trace(ls, L, None)
    assert sum(p.shape[1] for p in parts_all) == 16
    projectors = respinach.reduce(ls, L, coil)
    assert sum(p.shape[1] for p in projectors) == 2
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        respinach.reduce(respinach.basis(system(["1H"]), HILB), np.eye(2), np.eye(2))


def test_binpack():
    assert binpack([3, 2, 5, 1, 1], 4) == [[2], [0], [1, 3, 4]]
    assert binpack([], 4) == []
    assert binpack([1, 1, 1], 1) == [[0], [1], [2]]


def test_two_site_exchange_matches_mcconnell():
    """Symmetric exchange of two uncoupled spins: the FID follows the 2x2 McConnell equations."""
    k, shifts = 2e3, [0.0, 3.0]
    ss = system(
        ["1H", "1H"],
        zeeman={"scalar": shifts},
        chem={"parts": [[0], [1]], "rates": [[-k, k], [k, -k]], "concs": [1.0, 1.0]},
    )
    ls = respinach.basis(ss, LIOUV)
    pars = {"spins": ["1H"], "offset": [900.0], "sweep": 5000.0, "npoints": 128}
    pars["rho0"] = respinach.state(ls, "L+", "1H", "chem")
    pars["coil"] = respinach.state(ls, "L+", "1H")
    fid = respinach.liquid(ls, respinach.acquire, pars, "nmr")
    # Rotating-frame offsets of the two sites in rad/s: omega = basefrq*shift*1e-6 + 2*pi*offset
    omega = ss.basefrqs * np.array(shifts) * 1e-6 + 2 * np.pi * pars["offset"][0]
    M = np.array([[-1j * omega[0] - k, k], [k, -1j * omega[1] - k]])
    t = np.arange(pars["npoints"]) / pars["sweep"]
    reference = np.array([np.sum(sla.expm(M * tn) @ np.ones(2)) for tn in t])
    # exact states carry 1/sqrt(dim) in coil and rho0; the two sites are separate dim-2 systems
    assert_matches(fid, reference * (1 / 2), rtol=1e-8, atol=0, name="McConnell")


def test_unported_evolution_options_are_refused():
    ls = respinach.basis(system(*CASES["ab_2spin"]), LIOUV)
    L = liouvillian(ls)
    rho = respinach.state(ls, "L+", "1H")
    with pytest.raises(NotImplementedError, match="total"):
        respinach.evolution(ls, L, rho, rho, 1e-3, 2, "total")
    with pytest.raises(NotImplementedError, match="not ported"):
        respinach.create({"isotopes": ["1H"], "magnet": 14.1, "disable": ["symmetry"]}, {})
