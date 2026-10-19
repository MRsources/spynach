"""Shared helpers for the ctps tests: paper fixtures and the 1998 STE example."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from ctps import AllowedOrders, Nucleus, Period, PhaseCycle, Pulse, Sample, Sequence

FIXTURES = Path(__file__).parent / "fixtures" / "papers"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def all_pathways(case: dict) -> np.ndarray:
    """Every pathway of a cogwheel fixture case: rows of orders over the periods, from its allowed sets."""
    n_pulses = len(case["nu"])
    periods = [list(case["p0"])]
    overrides = {int(k): v for k, v in case.get("periods", {}).items()}
    for period in range(1, n_pulses):
        periods.append(overrides.get(period, case["orders"]))
    periods.append(overrides.get(n_pulses, case["detect"]))
    return np.array(list(itertools.product(*periods)), dtype=np.int16)


@pytest.fixture(scope="session")
def ste_1998():
    """The sequence and cycle of the 1998 Fig. 4 example, from its fixture file."""
    fx = load_fixture("ste_1998_fig4.json")
    n = fx["n_pulses"]
    periods = [Period(0.0)]
    for i in range(1, n + 1):
        key = str(i)
        duration = fx["durations_s"].get(key, 0.0)
        if key in fx["gradients"]:
            axis, amplitude = fx["gradients"][key]
            periods.append(Period.with_gradient(duration, axis, amplitude, fx["gradient_length_s"]))
        else:
            periods.append(Period(duration))
    allowed = AllowedOrders.standard(fx["p_max"], n + 1, start=fx["start"], end=fx["end"]).only(
        1, fx["allowed_period_1"]
    )
    sequence = Sequence(
        (Nucleus("1H", gamma=fx["gamma"]),),
        tuple(Pulse("1H") for _ in range(n)),
        tuple(periods),
        {"1H": allowed},
        Sample(**fx["sample"]),
        diffusion=fx["diffusion"],
    )
    cycle = PhaseCycle.explicit(fx["cycle"]["phases"], fx["cycle"]["receiver"], fx["cycle"]["base"])
    return fx, sequence, cycle
