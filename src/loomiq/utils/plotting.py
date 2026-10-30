"""Shared chart style and colors, so notebooks and the dashboard look the same.

Colors come from a colorblind-checked palette. Rules of thumb:
    * Each department always keeps the same color (sewing blue, finishing orange).
    * Met / missed target use the status colors and are always labelled,
      so meaning never depends on color alone.
    * Grid and axes stay light so the data stands out.

Example:
    >>> from loomiq.utils.plotting import apply_style, DEPARTMENT_COLORS
    >>> apply_style()
"""

from pathlib import Path

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#8a8984"

DEPARTMENT_COLORS = {"sewing": "#2a78d6", "finishing": "#eb6834"}
STATUS_COLORS = {"met": "#0ca30c", "missed": "#d03b3b"}
# Diverging poles for signed values such as correlations
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"


def apply_style() -> None:
    """Apply the LoomIQ matplotlib style (light surface, recessive grid, no top/right spines)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.size": 10,
        }
    )


def save_figure(fig, name: str, folder: Path | None = None) -> Path:
    """Save a figure as PNG to ``reports/figures/`` and return the path."""
    from loomiq.config import get_path

    folder = folder or get_path("figures")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
