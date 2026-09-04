"""
Project-wide figure style. Importing this module applies the house rcParams
(8 x 6 figures, Times / mathptmx text, out-facing ticks, dashed major grid, nothing
bold); `apply(use_tex=...)` re-applies it and `save(fig, path)` writes a figure
with a mathtext fallback when LaTeX rendering fails.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

FIG_SIZE = (8.0, 6.0)
AXIS_LW = 1.1
TICK_LABEL_SIZE = 20
AXIS_LABEL_SIZE = 24

_LATEX_PREAMBLE = r"\usepackage{mathptmx}\usepackage{amssymb}"

USING_TEX = False


def latex_available() -> bool:
    """True if a usable LaTeX + dvipng toolchain is on PATH."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def _luatex_runs() -> bool:
    """True if `luatex --version` runs at all (matplotlib's font lookup depends on it)."""
    if shutil.which("luatex") is None:
        return False
    try:
        return subprocess.run(["luatex", "--version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _shim_broken_luatex() -> None:
    """
    Route matplotlib's TeX font lookup through `kpsewhich` when `luatex` is broken.

    matplotlib.dviread only falls back to `kpsewhich` when luatex is absent; a present but
    non-functional luatex.exe makes every usetex font-metric lookup fail. Replacing the
    lookup class with one that raises FileNotFoundError takes that fallback branch.
    """
    import matplotlib.dviread as _dv

    class _NoLuatex:
        """Stand-in lookup class that always raises so matplotlib falls back to kpsewhich."""
        def __new__(cls):
            raise FileNotFoundError("luatex is broken here; use kpsewhich (plotstyle shim)")

    _dv._LuatexKpsewhich = _NoLuatex


def apply(use_tex: bool | None = None) -> bool:
    """
    Install the house style into matplotlib's global rcParams.

    ``use_tex=None`` auto-detects a LaTeX toolchain. Returns the resolved flag.
    """
    global USING_TEX
    if use_tex is None:
        use_tex = latex_available()

    plt.rcParams.update({
        "figure.figsize": FIG_SIZE,
        "figure.autolayout": True,
        "axes.linewidth": AXIS_LW,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.grid.axis": "both",
        "axes.axisbelow": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
        "xtick.minor.width": AXIS_LW, "ytick.minor.width": AXIS_LW,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.labelsize": TICK_LABEL_SIZE, "ytick.labelsize": TICK_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.weight": "normal",
        "axes.unicode_minus": False,
    })

    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": _LATEX_PREAMBLE,
        })
        if not _luatex_runs():
            _shim_broken_luatex()
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
        })

    USING_TEX = use_tex
    return use_tex


def save(fig, out_path, *, dpi: int = 150, **savefig_kw) -> Path:
    """
    Save a figure to ``out_path`` (creating parent dirs) and close it.

    If LaTeX rendering fails at draw time, retry once with LaTeX disabled.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=dpi, **savefig_kw)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"(LaTeX rendering failed, retrying with mathtext: {exc})")
        apply(use_tex=False)
        fig.savefig(out_path, dpi=dpi, **savefig_kw)
    plt.close(fig)
    return out_path


apply()
