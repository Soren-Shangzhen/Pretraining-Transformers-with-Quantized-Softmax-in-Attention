"""Figure style of the paper.

Usage
-----
    import paperstyle as ps
    ps.apply()
    fig, axes = ps.figure(ncols=3, height=1.8)
    ...
    ps.panel_label(axes[0], "a", "endpoint by K")
    ps.save(fig, "fig4.pdf")        # lint + exact physical size, no bbox tightening

Only matplotlib + numpy.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# ----------------------------------------------------------------------------- sizes
TEXTWIDTH_IN = 5.5  # \textwidth of the paper
BASE_PT = 7.0       # axis labels, titles, annotations
SMALL_PT = 6.5      # tick labels, legends
MIN_PT = 6.0        # hard floor for any Text object at final print size

# ----------------------------------------------------------------------------- colors
OI = {  # Okabe-Ito, colour-blind safe
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
GRAY = "#7F7F7F"
LIGHTGRAY = "#DDDDDD"


def shade(color, frac):
    """Mix `color` with white; frac=0 -> color, frac=1 -> white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": BASE_PT,
    "axes.titlesize": BASE_PT,
    "axes.labelsize": BASE_PT,
    "xtick.labelsize": SMALL_PT,
    "ytick.labelsize": SMALL_PT,
    "legend.fontsize": SMALL_PT,
    "legend.title_fontsize": SMALL_PT,
    "figure.titlesize": BASE_PT,
    "axes.titlepad": 3.0,
    "axes.labelpad": 2.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.8,
    "errorbar.capsize": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,          # grid and reference lines under the data
    "axes.grid": False,
    "grid.linewidth": 0.4,
    "grid.color": LIGHTGRAY,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 600,              # only affects raster parts (imshow)
    "pdf.fonttype": 42,              # TrueType, never Type 3
    "ps.fonttype": 42,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
}


def apply():
    mpl.rcParams.update(RC)


def figure(width=TEXTWIDTH_IN, height=1.8, nrows=1, ncols=1, **kw):
    """Figure at its FINAL printed size. Include in LaTeX without rescaling."""
    return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kw)


def panel_label(ax, letter, title=""):
    """Bold panel letter + plain title, left aligned (matches the caption's (a)/(A))."""
    ax.set_title(rf"$\bf{{{letter}}}$  {title}".rstrip(), loc="left")


# ----------------------------------------------------------------------------- lint
def _tick_label_texts(ax):
    """Tick labels that are actually drawn (inside the view limits)."""
    out = []
    for axis in (ax.xaxis, ax.yaxis):
        try:
            ticks = axis._update_ticks()  # private but stable; returns ticks to draw
        except Exception:  # pragma: no cover
            ticks = axis.get_major_ticks()
        for tk in ticks:
            for lab in (tk.label1, tk.label2):
                if lab.get_visible() and lab.get_text().strip():
                    out.append(lab)
    return out


def _all_tick_label_objs(ax):
    objs = set()
    for axis in (ax.xaxis, ax.yaxis):
        for tk in list(axis.majorTicks) + list(axis.minorTicks):
            objs.add(id(tk.label1))
            objs.add(id(tk.label2))
    return objs


def _data_points_display(ax, step_px=2.0):
    """Display-space points of lines (densified), scatter offsets and opaque patches' boxes."""
    pts = []
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        xy = np.asarray(line.get_xydata(), float)
        if xy.size == 0:
            continue
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        d = line.get_transform().transform(xy)
        if len(d) > 1 and line.get_linestyle() not in ("None", "none", "", " "):
            dense = [d[:1]]
            for p, q in zip(d[:-1], d[1:]):
                n = max(2, int(np.hypot(*(q - p)) / step_px) + 1)
                dense.append(np.linspace(p, q, n)[1:])
            d = np.vstack(dense)
        pts.append(d)
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and coll.get_visible():
            off = np.asarray(coll.get_offsets(), float)
            if off.size:
                off = off[np.isfinite(off).all(axis=1)]
                pts.append(coll.get_offset_transform().transform(off))
    return np.vstack(pts) if pts else np.empty((0, 2))


def _opaque_patch_boxes(ax, renderer):
    boxes = []
    for p in ax.patches:
        if not p.get_visible():
            continue
        alpha = p.get_alpha()
        fc_alpha = p.get_facecolor()[3]
        if (alpha is not None and alpha < 0.5) or fc_alpha < 0.5:
            continue
        boxes.append(p.get_window_extent(renderer))
    return boxes


def _clean(s, n=40):
    return re.sub(r"\$\\bf\{(\w+)\}\$\s*", r"\1 ", s)[:n]


def _area(b):
    return max(0.0, b.width) * max(0.0, b.height)


def lint_figure(fig, min_pt=MIN_PT, overlap_px2=4.0):
    """Heuristic checks. Returns a list of human-readable problems (empty = clean).

    Checks: font size floor; text-text overlap; text/legend clipped by figure edge;
    legends and in-axes annotations covering data (line vertices, scatter points,
    opaque bars). Treat as warnings and confirm on the rendered page.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fig_box = fig.bbox
    problems = []

    legends = [a.get_legend() for a in fig.axes if a.get_legend() is not None] + list(fig.legends)
    legend_text_ids = {id(t) for lg in legends for t in lg.get_texts()}
    if any(lg.get_title().get_text() for lg in legends):
        legend_text_ids |= {id(lg.get_title()) for lg in legends}

    tick_ids = set()
    items = []  # (label, bbox, text_obj)
    for ax in fig.axes:
        tick_ids |= _all_tick_label_objs(ax)
        for t in _tick_label_texts(ax):
            items.append(("tick", t.get_window_extent(r), t))
    for t in fig.findobj(Text):
        if id(t) in tick_ids or id(t) in legend_text_ids:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        items.append(("text", t.get_window_extent(r), t))
    for lg in legends:
        items.append(("legend", lg.get_window_extent(r), lg))

    # font floor: every drawn text, legend entries included
    drawn = [obj for kind, _, obj in items if kind != "legend"]
    drawn += [t for lg in legends for t in lg.get_texts()]
    for t in drawn:
        if t.get_text().strip() and t.get_fontsize() < min_pt - 1e-6:
            problems.append(f"font {t.get_fontsize():.1f}pt < {min_pt}pt: {t.get_text()[:40]!r}")

    def name(it):
        kind, _, obj = it
        return "legend" if kind == "legend" else f"{kind} {_clean(obj.get_text())!r}"

    # clipped by figure edge
    for it in items:
        b = it[1]
        if b.x0 < fig_box.x0 - 0.5 or b.y0 < fig_box.y0 - 0.5 or b.x1 > fig_box.x1 + 0.5 or b.y1 > fig_box.y1 + 0.5:
            problems.append(f"outside figure edge: {name(it)}")

    # pairwise overlap
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            inter = Bbox.intersection(items[i][1], items[j][1])
            if inter is not None and _area(inter) > overlap_px2:
                problems.append(f"overlap: {name(items[i])} <-> {name(items[j])}")

    # legends / annotations covering data
    for ax in fig.axes:
        pts = _data_points_display(ax)
        bars = _opaque_patch_boxes(ax, r)
        covers = []
        if ax.get_legend() is not None:
            covers.append(("legend", ax.get_legend().get_window_extent(r)))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                covers.append((f"annotation {_clean(t.get_text())!r}", t.get_window_extent(r)))
        for lbl, box in covers:
            n = int(np.sum((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1) &
                           (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))) if len(pts) else 0
            nb = sum(1 for bb in bars if (inter := Bbox.intersection(bb, box)) is not None
                     and _area(inter) > overlap_px2)
            if n or nb:
                problems.append(f"{lbl} covers data in axes {_clean(ax.get_title(loc='left'))!r}: "
                                f"{n} line/scatter pts, {nb} bars")
    return problems


def save(fig, path, lint=True):
    """Lint, then write at exact figsize (no bbox tightening). Returns lint problems."""
    path = Path(path)
    problems = lint_figure(fig) if lint else []
    w, h = fig.get_size_inches()
    kw = {}
    if path.suffix.lower() == ".pdf":
        kw["metadata"] = {"Creator": None, "Producer": None, "CreationDate": None}
    fig.savefig(path, **kw)
    status = "OK" if not problems else f"{len(problems)} WARNINGS"
    print(f"[paperstyle] {path.name}: {w:.2f} x {h:.2f} in  lint={status}")
    for p in problems:
        print("    -", p)
    return problems
