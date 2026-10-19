"""Figures build without error and carry the right structure; the exporters round-trip."""

import json

import numpy as np
import pytest

from ctps import PhaseCycle, Sequence, find_cycle, simulate
from ctps.io import bruker_phase_program, parse_cog, phase_table, write_csv, write_json


@pytest.fixture(scope="module")
def mqmas():
    sequence = Sequence.simple(3, 3)
    found = find_cycle(sequence, [(0, 3, 1, -1)], N_min=23, N_max=23)
    result = simulate(sequence, found.best.cycle, wanted=[(0, 3, 1, -1)], epsilon=0, cutoff=0)
    return sequence, found, result


def test_parse_and_print_cog():
    cycle = parse_cog("COG23(0,3,4;14)")
    assert cycle.winding == (23, 0, 3, 4, 14) and cycle.steps == 23 and cycle.cog_string() == "COG23(0,3,4;14)"
    assert parse_cog("COG11{0,1,0,1,0,1;6}").steps == 11
    with pytest.raises(ValueError):
        parse_cog("nested 4x4")


def test_bruker_phase_program_and_table():
    cycle = PhaseCycle.explicit([[0, 2], [0, 0, 1, 1]], [0, 2, 2, 0], base=4)
    text = bruker_phase_program(cycle)
    assert text.splitlines() == ["ph1=0 2 0 2", "ph2=0 0 1 1", "ph31=0 2 2 0"]
    rows = phase_table(cycle)
    assert rows[1] == {"step": 1, "pulse_1": 180.0, "pulse_2": 0.0, "receiver": 180.0}
    cog = PhaseCycle.cogwheel(6, [3, 1], 2)
    assert bruker_phase_program(cog).startswith("ph1=(6) 0 3 0 3 0 3")


def test_nested_cycle_expands_as_mixed_radix():
    cycle = PhaseCycle.nested([[0, 2], [0, 1, 2, 3]], receiver=[0, 2, 2, 0, 0, 2, 2, 0])
    assert cycle.steps == 8
    assert cycle.phases[0] == (0, 2, 0, 2, 0, 2, 0, 2)
    assert cycle.phases[1] == (0, 0, 1, 1, 2, 2, 3, 3)


def test_exporters_round_trip(tmp_path, mqmas):
    sequence, found, result = mqmas
    write_csv(result.top(5), tmp_path / "r.csv")
    write_json(result.top(5), tmp_path / "r.json")
    payload = json.loads((tmp_path / "r.json").read_text())
    assert len(payload["pathways"]) == 5 and payload["cycle"]["winding"] == [23, *found.best.nu, found.best.nu_rec]
    assert Sequence.from_dict(payload["sequence"]).n_pulses == 3
    assert PhaseCycle.from_dict(payload["cycle"]).steps == 23
    assert (tmp_path / "r.csv").read_text().startswith("rank,pathway,wanted,total")


def test_html_views(mqmas):
    sequence, found, result = mqmas
    for html in (sequence._repr_html_(), found._repr_html_(), result._repr_html_(), result[0]._repr_html_()):
        assert html.startswith("<div>") and html.endswith("</div>")
    assert "COG23" in found.best.cycle._repr_html_()
    assert result.leakage(0.01) == 0 and result.wanted.sum() == 1


def test_figures(mqmas):
    pytest.importorskip("plotly")
    from ctps.plot import plot_pareto, plot_pathways, plot_phase_vectors, plot_selection_diagram

    sequence, found, result = mqmas
    fig = plot_pathways(result, top=6)
    assert len(fig.data) == 6 and len(fig.layout.sliders[0].steps) == 7
    assert fig.data[0].line.dash == "solid"  # the wanted pathway ranks first
    diagram = plot_selection_diagram(found.best, sequence, [(0, 3, 1, -1)])
    assert "1 of 49 pathways pass" in diagram.layout.title.text
    phasors = plot_phase_vectors(result, 0)
    assert len(phasors.data) == 24  # 23 steps plus the mean
    assert np.isclose(abs(phasors.data[-1].x[1] + 1j * phasors.data[-1].y[1]), 1.0)
    pareto = plot_pareto([{"steps": 8, "residual": 0.1, "label": "a"}, {"steps": 16, "residual": 0.0, "label": "b"}])
    assert len(pareto.data) == 1
