"""Shared matplotlib style for every figure this project produces.

Import and call `apply_style()` once at the top of a plotting script, then use
`LINE` / `POINTS` for the data marks and `refline()` for reference lines.

The point of centralising this is that the evaluation scripts previously
disagreed: the CEM ones drew `marker="o"` at matplotlib's default size 6.0 on
125-point series, which merges into a solid blob and hides the line it is
supposed to decorate, while the TRPO ones drew a bare line. Markers on a dense
time series are decoration, not the primary encoding, so they are sized down
to just legible (`MARKER_SIZE`) and the line carries the shape.
"""

import matplotlib as mpl
import matplotlib.ticker as mticker

# Validated categorical palette, slots 1-8, in fixed order. Assign by index --
# never cycle, and never let a series change colour when another is filtered out.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

LINE_WIDTH = 1.6
MARKER_SIZE = 2.6  # decoration on a dense line, not the primary mark

#: Line with small point marks -- for step-indexed trajectories.
LINE = dict(
    linewidth=LINE_WIDTH,
    marker="o",
    markersize=MARKER_SIZE,
    markeredgewidth=0,
    color=SERIES[0],
)

#: Line only, no marks -- for very dense series (more than ~200 points).
LINE_PLAIN = dict(linewidth=LINE_WIDTH, color=SERIES[0])

#: Standalone points -- here the mark IS the encoding, so it is bigger.
POINTS = dict(linestyle="none", marker="o", markersize=4.5, markeredgewidth=0)


def apply_style():
    """Set the global rcParams. Call once, before creating any figure."""
    mpl.rcParams.update(
        {
            "figure.figsize": (6.4, 4.0),
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            # Recessive frame: keep the left and bottom rules, drop the box.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            # Recessive grid, always behind the data.
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def refline(ax, y=None, x=None, label=None, color=INK_MUTED):
    """A dashed annotation line (true value, coherence time, ...).

    Reference lines are annotation, not data, so they wear muted ink rather
    than a series colour.
    """
    kw = dict(color=color, linestyle="--", linewidth=1.0, zorder=1.5, label=label)
    return ax.axhline(y, **kw) if y is not None else ax.axvline(x, **kw)


def log_y(ax=None):
    """Switch the y axis to log and keep the minor ticks readable.

    Matplotlib labels only whole decades on a log axis. These variance curves
    span about 1.5 decades, so the default leaves a single labelled tick and
    the reader cannot recover a value from the plot.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(
        mticker.LogFormatterSciNotation(minor_thresholds=(2.0, 0.4))
    )
    ax.tick_params(axis="y", which="minor", labelsize=7.5, colors=INK_MUTED)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.7)
    return ax
