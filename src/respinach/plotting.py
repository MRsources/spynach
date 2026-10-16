"""Frequency axes and interactive plots of simulation results.

``ft_axis`` and ``axis_1d`` are ports of Spinach's ``kernel/utilities/ft_axis.m``
and ``kernel/utilities/axis_1d.m``. ``plot_1d`` follows Spinach's
``plot_1d.m`` (same parameters, defaults, and axis conventions) but draws an
interactive Plotly figure and returns it instead of drawing into the current
MATLAB axes. ``plot_2d`` follows ``plot_2d.m`` (same
parameters, contour spacing by ``contspacing.m``, and axis conventions) as
an interactive contour figure. ``plot_fid`` has no Spinach counterpart.

The plotting functions need Plotly (``pip install -e ".[plot]"``); the axis
functions need only NumPy.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from respinach.isotopes import spin
from respinach.system import FREE_ELECTRON_G, SpinSystem

# Categorical series colours, in fixed order, stepped separately for light and
# dark surfaces. Adjacent pairs stay distinguishable under the common forms of
# colour blindness; the first three are distinguishable all-pairs.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

# Surface, ink, and rule colours
CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Hover precision per axis unit
_HOVER_FORMAT = {
    "ppm": ".4f",
    "Hz": ".2f",
    "kHz": ".4f",
    "MHz": ".6f",
    "MHz-labframe": ".6f",
    "GHz": ".6f",
    "GHz-labframe": ".6f",
    "Gauss": ".3f",
    "mT": ".4f",
    "gtensor": ".6f",
    "points": "d",
}


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


def contspacing(
    smax: float, smin: float, delta: Sequence[float], k: int, signs: str, ncont: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contour levels for a 2D spectrum (port of ``contspacing.m``).

    Positive levels run from ``delta[0] * smax`` to ``delta[1] * smax`` and
    negative ones from ``delta[2] * smin`` to ``delta[3] * smin``, spaced
    as the ``k``-th power of a linear grid of ``ncont`` points; ``signs``
    is ``'positive'``, ``'negative'`` or ``'both'``. Returns all levels in
    ascending order, then the positive and the negative sets.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.size != 4 or np.any(delta < 0) or np.any(delta > 1) or delta[0] > delta[1] or delta[2] > delta[3]:
        raise ValueError(
            "delta must give ascending minimum and maximum contour fractions between 0 and 1 for each sign."
        )
    if signs not in ("positive", "negative", "both"):
        raise ValueError("signs must be 'positive', 'negative', or 'both'.")
    if int(k) != k or k < 1 or int(ncont) != ncont or ncont < 1:
        raise ValueError("k and ncont must be positive integers.")
    grid = np.linspace(0, 1, int(ncont)) ** k
    pos = (
        (delta[1] - delta[0]) * smax * grid + smax * delta[0]
        if smax > 0 and signs in ("positive", "both")
        else np.array([])
    )
    neg = (
        (delta[3] - delta[2]) * smin * grid + smin * delta[2]
        if smin < 0 and signs in ("negative", "both")
        else np.array([])
    )
    return np.concatenate([neg[::-1], pos]), pos, neg


def axis_2d(
    spin_system: SpinSystem, spectrum: np.ndarray, parameters: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """F1 and F2 axes and labels of a 2D spectrum, as ``plot_2d.m`` derives them.

    ``spectrum`` is ``(nF2, nF1)``: rows are the direct dimension F2
    (``parameters['sweep'][1]``, ``['offset'][1]``, ``['spins'][1]``),
    columns the indirect dimension F1. Scalar ``spins``, ``offset`` and
    ``sweep`` apply to both dimensions.
    """
    spins = list(parameters["spins"]) * 2 if len(parameters["spins"]) == 1 else list(parameters["spins"])
    offset = np.atleast_1d(np.asarray(parameters.get("offset", 0.0), dtype=float))
    sweep = np.atleast_1d(np.asarray(parameters["sweep"], dtype=float))
    offset = np.repeat(offset, 2) if offset.size == 1 else offset
    sweep = np.repeat(sweep, 2) if sweep.size == 1 else sweep
    units = parameters.get("axis_units", "ppm")
    axis_f1 = ft_axis(offset[0], sweep[0], spectrum.shape[1])
    axis_f2 = ft_axis(offset[1], sweep[1], spectrum.shape[0])
    if units == "ppm":
        axis_f1 = 1e6 * 2 * np.pi * axis_f1 / (spin(spins[0])[0] * spin_system.magnet)
        axis_f2 = 1e6 * 2 * np.pi * axis_f2 / (spin(spins[1])[0] * spin_system.magnet)
        labels = (f"F1: {spins[0]} chemical shift / ppm", f"F2: {spins[1]} chemical shift / ppm")
    elif units in ("Hz", "kHz", "MHz"):
        scale = {"Hz": 1.0, "kHz": 1e-3, "MHz": 1e-6}[units]
        axis_f1, axis_f2 = scale * axis_f1, scale * axis_f2
        labels = (f"F1: {spins[0]} linear frequency / {units}", f"F2: {spins[1]} linear frequency / {units}")
    elif units == "Gauss":
        axis_f1 = 1e4 * (spin_system.magnet - 2 * np.pi * axis_f1 / spin("E")[0])
        axis_f2 = 1e4 * (spin_system.magnet - 2 * np.pi * axis_f2 / spin("E")[0])
        labels = ("F1: magnetic induction / Gauss", "F2: magnetic induction / Gauss")
    elif units == "points":
        axis_f1, axis_f2 = np.arange(1, spectrum.shape[1] + 1), np.arange(1, spectrum.shape[0] + 1)
        labels = (f"F1: {spins[0]} linear frequency / points", f"F2: {spins[1]} linear frequency / points")
    else:
        raise ValueError("unknown axis units.")
    return axis_f1, axis_f2, labels[0], labels[1]


def plot_2d(
    spin_system: SpinSystem,
    spectrum: np.ndarray,
    parameters: dict[str, Any],
    ncont: int = 20,
    delta: Sequence[float] = (0.02, 0.2, 0.02, 0.2),
    k: int = 2,
    ncol: int = 256,
    m: int = 6,
    signs: str = "both",
    *,
    title: str | None = None,
    theme: str = "light",
    height: int = 560,
    max_points: int | None = 512,
):
    """Interactive contour plot of a real 2D spectrum, after Spinach's ``plot_2d``.

    ``spectrum`` is the ``(nF2, nF1)`` array that ``fftshift(fft2(fid))``
    of a sequence's FID gives (rows along the direct dimension); its real
    part is drawn, positive contours in red and negative ones in blue.
    ``ncont``, ``delta``, ``k`` and ``signs`` set the contour levels as in
    :func:`contspacing`; ``ncol`` and ``m`` (Spinach's colour map
    parameters) are accepted for compatibility and ``m`` steepens the
    colour gradient. Both axes decrease to the right/top as in Spinach.
    Returns a ``plotly.graph_objects.Figure``.

    The figure embeds the matrix it draws, so a dimension longer than
    ``max_points`` is reduced first: consecutive points are binned and
    each bin keeps its value of largest magnitude (peaks and signs are
    preserved, the contour levels are those of the full spectrum). A
    2048 x 2048 spectrum reduces to 512 x 512 and a few MB of HTML;
    ``max_points=None`` draws every point.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError('plotting needs Plotly: pip install -e ".[plot]"') from error
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    spectrum = np.asarray(spectrum)
    if np.iscomplexobj(spectrum):
        spectrum = spectrum.real
    if spectrum.ndim != 2:
        raise ValueError("spectrum must be a 2D array.")
    c = CHROME[theme]
    axis_f1, axis_f2, label_f1, label_f2 = axis_2d(spin_system, spectrum, parameters)
    smax, smin = float(spectrum.max()), float(spectrum.min())
    _, pos, neg = contspacing(smax, smin, delta, k, signs, ncont)
    if max_points is not None:
        spectrum, axis_f2 = _bin_extremes(spectrum, axis_f2, int(max_points), axis=0)
        spectrum, axis_f1 = _bin_extremes(spectrum, axis_f1, int(max_points), axis=1)
    fig = go.Figure()
    reds, blues = ("#b3261e", "#f2b8b5"), ("#1d4ed8", "#bfdbfe")

    def shade(start: str, end: str, fraction: float) -> str:
        a, b = (int(start[i : i + 2], 16) for i in (1, 3, 5)), (int(end[i : i + 2], 16) for i in (1, 3, 5))
        return "#" + "".join(f"{int(round(x + (y - x) * fraction)):02x}" for x, y in zip(a, b))

    # One trace per sign: the spectrum is mapped through a monotone piecewise-linear function that sends
    # the n-th contour level to n, so Plotly's uniform levels 0, 1, ..., reproduce Spinach's non-uniform
    # ones exactly, and the colour scale over the level index gives each contour its own shade.
    z = spectrum.T  # Plotly draws z[y, x]; x is F2 (rows of the spectrum), y is F1
    for levels, (strong, weak) in ((pos, reds), (neg, blues)):
        if not len(levels):
            continue
        nlev = len(levels)
        colours = [shade(weak, strong, (1 - n / max(nlev - 1, 1)) ** (1 / m)) for n in range(nlev)]
        colorscale = [(n / max(nlev - 1, 1), colour) for n, colour in enumerate(colours)]
        if nlev == 1:
            colorscale = [(0.0, colours[0]), (1.0, colours[0])]
        fig.add_contour(
            x=axis_f2,
            y=axis_f1,
            z=_level_index(z, levels),
            showscale=False,
            hoverinfo="skip",
            zmin=0,
            zmax=nlev - 1,
            autocontour=False,
            contours={
                "type": "levels",
                "coloring": "lines",
                "start": 0,
                "end": nlev - 1,
                "size": 1,
                "showlabels": False,
            },
            colorscale=colorscale,
            line={"width": 1.2},
            showlegend=False,
        )
    if not len(pos) and not len(neg):
        fig.add_annotation(text="all-zero spectrum", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    axis = {
        "showgrid": True,
        "gridcolor": c["grid"],
        "zeroline": False,
        "showline": True,
        "linecolor": c["axis"],
        "ticks": "outside",
        "tickcolor": c["axis"],
        "tickfont": {"color": c["ink2"], "size": 12},
        "title": {"font": {"color": c["ink2"], "size": 13}},
        "mirror": True,
    }
    isotopes = list(parameters["spins"]) * 2 if len(parameters["spins"]) == 1 else list(parameters["spins"])
    fig.update_layout(
        height=height,
        width=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 72, "r": 24, "t": 56 if title else 24, "b": 64},
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": c["ink"]}} if title else None,
        xaxis={
            **axis,
            "title": {**axis["title"], "text": _rich(label_f2, isotopes[1])},
            "range": [float(axis_f2.max()), float(axis_f2.min())],
        },
        yaxis={
            **axis,
            "title": {**axis["title"], "text": _rich(label_f1, isotopes[0])},
            "range": [float(axis_f1.max()), float(axis_f1.min())],
        },
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": c["muted"],
            "activecolor": c["ink2"],
            "remove": ["lasso2d", "select2d"],
        },
        dragmode="zoom",
        showlegend=False,
    )
    return fig


def _level_index(z: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Monotone map sending ``levels[n]`` to ``n`` (piecewise linear between levels, extended linearly outside).

    Values more than one level spacing beyond the end levels are clamped to
    -1 and ``len(levels)``, outside the range of contours that are drawn.
    """
    order = np.argsort(levels)
    xs, ys = np.asarray(levels, dtype=float)[order], np.arange(len(levels), dtype=float)[order]
    span = xs[-1] - xs[0] if len(xs) > 1 else max(abs(xs[0]), 1.0)
    step = (xs[1] - xs[0]) if len(xs) > 1 else span
    last = (xs[-1] - xs[-2]) if len(xs) > 1 else span
    xs = np.concatenate([[xs[0] - step], xs, [xs[-1] + last]])
    ys = np.concatenate(
        [[ys[0] + (ys[0] - ys[1] if len(ys) > 1 else -1)], ys, [ys[-1] + (ys[-1] - ys[-2] if len(ys) > 1 else 1)]]
    )
    return np.interp(z, xs, ys)


def _bin_extremes(
    spectrum: np.ndarray, axis_values: np.ndarray, max_points: int, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``spectrum`` along ``axis`` to at most ``max_points`` bins, keeping each bin's largest-magnitude value.

    The axis coordinate of a bin is the mean of its points. Returns the
    spectrum unchanged when it is short enough.
    """
    n = spectrum.shape[axis]
    if max_points < 1:
        raise ValueError("max_points must be a positive integer or None.")
    if n <= max_points:
        return spectrum, axis_values
    step = -(-n // max_points)
    nbins = -(-n // step)
    pad = nbins * step - n
    moved = np.moveaxis(spectrum, axis, 0)
    padded = np.concatenate([moved, np.zeros((pad,) + moved.shape[1:], dtype=moved.dtype)]) if pad else moved
    binned = padded.reshape(nbins, step, *moved.shape[1:])
    pick = np.argmax(np.abs(binned), axis=1)
    reduced = np.take_along_axis(binned, pick[:, None, ...], axis=1)[:, 0, ...]
    coords = np.concatenate([np.asarray(axis_values, dtype=float), np.full(pad, np.nan)]).reshape(nbins, step)
    return np.moveaxis(reduced, 0, axis), np.nanmean(coords, axis=1)


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
    """Line plot of the traces (x set by the caller) with the respinach look.

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
        "showgrid": True,
        "gridcolor": c["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": c["axis"],
        "linewidth": 1,
        "mirror": False,
        "ticks": "outside",
        "tickcolor": c["axis"],
        "ticklen": 4,
        "tickfont": {"color": c["ink2"], "size": 12},
        "title": {"font": {"color": c["ink2"], "size": 13}},
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikethickness": 1,
        "spikedash": "solid",
        "spikecolor": c["muted"],
    }
    fig = go.Figure()
    fig.update_layout(
        height=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 64, "r": 24, "t": 56 if title else 24, "b": 56},
        title={"text": title, "x": 0, "xanchor": "left", "xref": "paper", "font": {"size": 15, "color": c["ink"]}}
        if title
        else None,
        xaxis=axis,
        yaxis={**axis, "showspikes": False, "exponentformat": "power"},
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": c["ink2"]},
        },
        hoverlabel={"bgcolor": c["surface"], "bordercolor": c["border"], "font": {"color": c["ink"], "family": FONT}},
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": c["muted"],
            "activecolor": c["ink2"],
            "remove": ["lasso2d", "select2d"],
        },
        dragmode="zoom",
        hovermode="x unified",
        showlegend=len(traces) > 1,
    )
    if len(traces) > len(SERIES[theme]):
        raise ValueError(f"at most {len(SERIES[theme])} traces can be told apart by colour; plot fewer.")
    for n, (name, y) in enumerate(traces):
        fig.add_scatter(
            y=y,
            name=name,
            mode="lines",
            line={"color": SERIES[theme][n], "width": 2},
            hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>" if name else "%{y:.4g}<extra></extra>",
        )
    return fig


def plot_time_series(
    time_axis: np.ndarray,
    series: np.ndarray | Sequence[np.ndarray],
    *,
    names: Sequence[str] | None = None,
    xlabel: str = "time / s",
    ylabel: str = "amplitude",
    title: str | None = None,
    theme: str = "light",
    height: int = 360,
):
    """Interactive line plot of one or more real series against a common axis (MATLAB's ``plot(t, y)``).

    ``series`` is a 1-D array, a 2-D array with one series per row, or a
    list of 1-D arrays, each as long as ``time_axis``. More than eight
    series get an evenly spaced hue cycle, as Spinach's ``trajan`` draws.
    """
    import colorsys

    rows = (
        [np.asarray(series)]
        if isinstance(series, np.ndarray) and series.ndim == 1
        else [np.asarray(r).real for r in series]
    )
    rows = [r.real for r in rows]
    time_axis = np.asarray(time_axis, dtype=float)
    if any(r.ndim != 1 or r.size != time_axis.size for r in rows):
        raise ValueError("every series must be a 1-D array as long as time_axis.")
    if names is None:
        names = [f"series {n + 1}" for n in range(len(rows))] if len(rows) > 1 else [""]
    if len(names) != len(rows):
        raise ValueError(f"{len(names)} names given for {len(rows)} series.")
    fig = _figure(theme, title, height, [])
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    palette = (
        SERIES[theme]
        if len(rows) <= len(SERIES[theme])
        else [
            "#" + "".join(f"{int(255 * v):02x}" for v in colorsys.hsv_to_rgb((n + 1) / len(rows), 0.75, 0.75))
            for n in range(len(rows))
        ]
    )
    for n, (name, y) in enumerate(zip(names, rows)):
        fig.add_scatter(
            x=time_axis,
            y=y,
            name=name,
            mode="lines",
            line={"color": palette[n], "width": 2},
            hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>" if name else "%{y:.4g}<extra></extra>",
        )
    fig.update_layout(showlegend=len(rows) > 1)
    fig.update_xaxes(title_text=xlabel, range=[float(time_axis.min()), float(time_axis.max())], hoverformat=".4g")
    fig.update_yaxes(title_text=ylabel)
    return fig


def plot_trajan(
    spin_system: SpinSystem,
    traj: np.ndarray,
    property: str,
    time_axis: np.ndarray | None = None,
    *,
    title: str | None = None,
    theme: str = "light",
    height: int = 360,
):
    """Trajectory analysis plot, after Spinach's ``trajan``: the amplitudes from :func:`respinach.trajan` against time.

    Without ``time_axis`` the x axis is the trajectory point number. The
    y range is padded by five percent of the largest amplitude, as in Spinach.
    """
    from respinach.states import trajan

    result, labels = trajan(spin_system, traj, property)
    x = np.arange(result.shape[1]) if time_axis is None else np.asarray(time_axis, dtype=float)
    ylabel = {
        "correlation_order": "correlation order amplitude",
        "coherence_order": "coherence order amplitude",
        "total_each_spin": "density touching each spin",
        "local_each_spin": "density local to each spin",
    }[property]
    heading = {
        "correlation_order": "correlation orders",
        "coherence_order": "coherence orders",
        "total_each_spin": "spin populations",
        "local_each_spin": "spin populations",
    }[property]
    fig = plot_time_series(
        x,
        result,
        names=labels,
        xlabel="time / s" if time_axis is not None else "trajectory point",
        ylabel=ylabel,
        title=title or heading,
        theme=theme,
        height=height,
    )
    top = float(result.max()) if result.size else 1.0
    fig.update_yaxes(range=[-0.05 * top, 1.05 * top])
    return fig
