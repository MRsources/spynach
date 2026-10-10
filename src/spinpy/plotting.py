"""Frequency axes and interactive plots of simulation results.

``ft_axis`` and ``axis_1d`` are ports of Spinach's ``kernel/utilities/ft_axis.m``
and ``kernel/utilities/axis_1d.m``. ``plot_1d`` follows Spinach's
``plot_1d.m`` (same parameters, defaults, and axis conventions) but draws an
interactive Plotly figure and returns it instead of drawing into the current
MATLAB axes. ``plot_fid`` has no Spinach counterpart.

The plotting functions need Plotly (``pip install -e ".[plot]"``); the axis
functions need only NumPy.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from spinpy.isotopes import spin
from spinpy.system import FREE_ELECTRON_G, SpinSystem

# Categorical series colours, in fixed order, stepped separately for light and
# dark surfaces. Adjacent pairs stay distinguishable under the common forms of
# colour blindness; the first three are distinguishable all-pairs.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

# Surface, ink, and rule colours
CHROME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)"},
}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Hover precision per axis unit
_HOVER_FORMAT = {"ppm": ".4f", "Hz": ".2f", "kHz": ".4f", "MHz": ".6f", "MHz-labframe": ".6f",
                 "GHz": ".6f", "GHz-labframe": ".6f", "Gauss": ".3f", "mT": ".4f", "gtensor": ".6f",
                 "points": "d"}


def ft_axis(offset: float, sweep: float, npoints: int) -> np.ndarray:
    """Frequency axis (Hz) of a Fourier-transformed signal, as Spinach's ``ft_axis``.

    ``npoints`` points spaced ``sweep / npoints`` apart, centred on
    ``offset``; for even ``npoints`` the axis starts at ``offset - sweep/2``.
    This is the axis of ``np.fft.fftshift(np.fft.fft(fid, npoints))``.
    """
    if not np.isscalar(offset) or np.iscomplexobj(offset):
        raise ValueError("offset must be a real scalar.")
    if not np.isscalar(sweep) or np.iscomplexobj(sweep) or sweep <= 0:
        raise ValueError("sweep must be a positive real scalar.")
    if int(npoints) != npoints or npoints <= 2:
        raise ValueError("npoints must be a real integer greater than 2")
    npoints = int(npoints)
    ax = np.linspace(-sweep / 2, sweep / 2, npoints + 1)
    if npoints % 2 == 1:
        return ax[1:] - (ax[1] - ax[0]) / 2 + offset
    return ax[:-1] + offset


def axis_1d(spin_system: SpinSystem, parameters: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Axis and axis label of a 1D spectrum, as Spinach's ``axis_1d``.

    Uses ``parameters['sweep']`` (a scalar sweep width with
    ``parameters['offset']``, or a two-element ``[low, high]`` range in Hz),
    ``parameters['zerofill']`` (number of points), ``parameters['spins']``
    (one isotope), and ``parameters['axis_units']``: ``'ppm'``, ``'Hz'``,
    ``'kHz'``, ``'MHz'``, ``'GHz'``, ``'MHz-labframe'``, ``'GHz-labframe'``,
    ``'Gauss'``, ``'mT'``, ``'gtensor'``, or ``'points'``.
    """
    sweep = np.atleast_1d(np.asarray(parameters.get("sweep"), dtype=float)) if "sweep" in parameters else None
    if sweep is None:
        raise ValueError("sweep width should be specified in parameters['sweep'].")
    if sweep.size not in (1, 2):
        raise ValueError("parameters['sweep'] should have one or two elements.")
    if sweep.size == 2:
        if sweep[0] >= sweep[1]:
            raise ValueError("the first element of parameters['sweep'] must be smaller than the second.")
        if "offset" in parameters:
            raise ValueError("offset should not be specified when parameters['sweep'] has two elements.")
        ax = np.linspace(sweep[0], sweep[1], int(parameters["zerofill"]))
    else:
        if "offset" not in parameters:
            raise ValueError("offset should be specified in parameters['offset'].")
        ax = ft_axis(float(parameters["offset"]), float(sweep[0]), parameters["zerofill"])

    spins = parameters.get("spins")
    if not isinstance(spins, (list, tuple)) or len(spins) != 1:
        raise ValueError("parameters['spins'] must be a list with exactly one isotope.")
    isotope = spins[0]
    basefrq = -spin(isotope)[0] * spin_system.magnet / (2 * np.pi)

    units = parameters["axis_units"]
    if units == "ppm":
        return -1e6 * ax / basefrq, f"{isotope} chemical shift / ppm"
    if units == "Gauss":
        return 1e4 * (spin_system.magnet - 2 * np.pi * ax / spin("E")[0]), "Magnetic induction / Gauss"
    if units == "mT":
        return 1e3 * (spin_system.magnet - 2 * np.pi * ax / spin("E")[0]), "Magnetic induction / mT"
    if units in ("Hz", "kHz", "MHz", "GHz"):
        scale = {"Hz": 1, "kHz": 1e-3, "MHz": 1e-6, "GHz": 1e-9}[units]
        return scale * ax, f"{isotope} offset frequency / {units}"
    if units in ("MHz-labframe", "GHz-labframe"):
        scale = 1e-6 if units.startswith("MHz") else 1e-9
        return -scale * (ax - basefrq), f"{isotope} frequency / {units.split('-')[0]}"
    if units == "gtensor":
        return -FREE_ELECTRON_G * (ax - basefrq) / basefrq, f"{isotope} g-tensor / Bohr magneton"
    if units == "points":
        return np.arange(1, int(parameters["zerofill"]) + 1, dtype=float), "digitisation points"
    raise ValueError("unknown axis units.")


def plot_1d(
    spin_system: SpinSystem,
    spectrum: np.ndarray | Sequence[np.ndarray],
    parameters: dict[str, Any],
    *,
    names: Sequence[str] | None = None,
    title: str | None = None,
    theme: str = "light",
    height: int = 400,
):
    """Interactive plot of one or more 1D spectra, after Spinach's ``plot_1d``.

    ``spectrum`` is a 1-D array, a 2-D array with one spectrum per column
    (as in MATLAB's ``plot``), or a list of 1-D arrays. Complex spectra are
    drawn as their real and imaginary parts, as in Spinach.

    ``parameters`` as for :func:`axis_1d`, with Spinach's defaults:
    ``axis_units='ppm'``, ``invert_axis=1`` (NMR tradition: frequency
    decreasing to the right), ``offset=0``, and ``zerofill`` taken from the
    spectrum length. ``parameters['derivative']`` is not ported.

    ``names`` labels the spectra in the legend and hover box; ``theme`` is
    ``'light'`` or ``'dark'``. Drag to zoom, double-click to zoom out; the
    hover box reads off every spectrum at the cursor. Returns a ``plotly.graph_objects.Figure``: display
    it by making it the last expression of a notebook cell, or call
    ``fig.show()`` or ``fig.write_html(path)``.
    """
    traces = _as_traces(spectrum, names)
    npts = traces[0][1].size
    if any(y.size != npts for _, y in traces):
        raise ValueError("all spectra must have the same number of points.")

    parameters = dict(parameters)
    if parameters.get("derivative"):
        raise NotImplementedError("parameters['derivative'] is not ported yet.")
    if "offset" not in parameters and np.size(parameters.get("sweep")) == 1:
        parameters["offset"] = 0
    parameters.setdefault("axis_units", "ppm")
    parameters.setdefault("invert_axis", 1)
    parameters.setdefault("zerofill", npts)
    if parameters["zerofill"] != npts:
        raise ValueError(f"parameters['zerofill'] is {parameters['zerofill']} but the spectrum has {npts} points.")
    if not isinstance(parameters["axis_units"], str):
        raise ValueError("parameters['axis_units'] must be a character string.")

    ax, label = axis_1d(spin_system, parameters)
    fig = _figure(theme, title, height, traces)
    fig.update_traces(x=ax)
    lo, hi = float(ax.min()), float(ax.max())
    fig.update_xaxes(
        title_text=_rich(label, parameters["spins"][0]),
        range=[hi, lo] if parameters["invert_axis"] else [lo, hi],
        hoverformat=_HOVER_FORMAT[parameters["axis_units"]],
    )
    fig.update_yaxes(title_text="intensity / a.u.")
    return fig


def plot_fid(
    fid: np.ndarray | Sequence[np.ndarray],
    parameters: dict[str, Any],
    *,
    names: Sequence[str] | None = None,
    title: str | None = None,
    theme: str = "light",
    height: int = 360,
):
    """Interactive plot of one or more free induction decays.

    The time axis starts at zero with dwell time ``1 / parameters['sweep']``
    and is drawn in ms when the acquisition is shorter than a second.
    Complex FIDs are drawn as their real and imaginary parts. Other
    arguments as for :func:`plot_1d`.
    """
    traces = _as_traces(fid, names)
    npts = traces[0][1].size
    if any(y.size != npts for _, y in traces):
        raise ValueError("all FIDs must have the same number of points.")
    if np.size(parameters.get("sweep")) != 1:
        raise ValueError("parameters['sweep'] must be a scalar sweep width in Hz.")
    t = np.arange(npts) / float(parameters["sweep"])
    scale, unit = (1e3, "ms") if t[-1] < 1 else (1.0, "s")

    fig = _figure(theme, title, height, traces)
    fig.update_traces(x=scale * t)
    fig.update_xaxes(title_text=f"time / {unit}", range=[0, scale * t[-1]], hoverformat=".4g")
    fig.update_yaxes(title_text="signal / a.u.")
    return fig


def _as_traces(data: np.ndarray | Sequence[np.ndarray], names: Sequence[str] | None) -> list[tuple[str, np.ndarray]]:
    """(name, real 1-D array) pairs, splitting complex data into real and imaginary parts."""
    if isinstance(data, np.ndarray) and data.ndim <= 2:
        columns = [data] if data.ndim == 1 else list(data.T)
    else:
        columns = [np.asarray(d) for d in data]
    if any(c.ndim != 1 for c in columns):
        raise ValueError("expected a 1-D array, a 2-D array of columns, or a list of 1-D arrays.")
    if names is None:
        names = [f"spectrum {n + 1}" for n in range(len(columns))] if len(columns) > 1 else [""]
    if len(names) != len(columns):
        raise ValueError(f"{len(names)} names given for {len(columns)} traces.")

    traces = []
    for name, column in zip(names, columns):
        if np.iscomplexobj(column):
            traces.append((f"{name} real".strip(), column.real))
            traces.append((f"{name} imag".strip(), column.imag))
        else:
            traces.append((name, column))
    return traces


def _rich(label: str, isotope: str) -> str:
    """Write the mass number of the isotope in an axis label as a superscript."""
    match = re.fullmatch(r"(\d+)([A-Za-z]+)", isotope)
    if not match:
        return label
    return label.replace(isotope, f"<sup>{match[1]}</sup>{match[2]}", 1)


def _figure(theme: str, title: str | None, height: int, traces: list[tuple[str, np.ndarray]]):
    """Line plot of the traces (x set by the caller) with the spinpy look.

    Thin rules and quiet axes; a vertical crosshair and one hover box listing
    every trace at the cursor; a legend only when there are several traces.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError('plotting needs Plotly: pip install -e ".[plot]"') from error
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    c = CHROME[theme]

    axis = {
        "showgrid": True, "gridcolor": c["grid"], "gridwidth": 1,
        "zeroline": False, "showline": True, "linecolor": c["axis"], "linewidth": 1, "mirror": False,
        "ticks": "outside", "tickcolor": c["axis"], "ticklen": 4,
        "tickfont": {"color": c["ink2"], "size": 12}, "title": {"font": {"color": c["ink2"], "size": 13}},
        "showspikes": True, "spikemode": "across", "spikesnap": "cursor", "spikethickness": 1,
        "spikedash": "solid", "spikecolor": c["muted"],
    }
    fig = go.Figure()
    fig.update_layout(
        height=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        margin={"l": 64, "r": 24, "t": 56 if title else 24, "b": 56},
        title={"text": title, "x": 0, "xanchor": "left", "xref": "paper",
               "font": {"size": 15, "color": c["ink"]}} if title else None,
        xaxis=axis, yaxis={**axis, "showspikes": False, "exponentformat": "power"},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom",
                "bgcolor": "rgba(0,0,0,0)", "font": {"color": c["ink2"]}},
        hoverlabel={"bgcolor": c["surface"], "bordercolor": c["border"], "font": {"color": c["ink"], "family": FONT}},
        modebar={"bgcolor": "rgba(0,0,0,0)", "color": c["muted"], "activecolor": c["ink2"],
                 "remove": ["lasso2d", "select2d"]},
        dragmode="zoom", hovermode="x unified", showlegend=len(traces) > 1,
    )
    if len(traces) > len(SERIES[theme]):
        raise ValueError(f"at most {len(SERIES[theme])} traces can be told apart by colour; plot fewer.")
    for n, (name, y) in enumerate(traces):
        fig.add_scatter(y=y, name=name, mode="lines", line={"color": SERIES[theme][n], "width": 2},
                        hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>" if name else "%{y:.4g}<extra></extra>")
    return fig
