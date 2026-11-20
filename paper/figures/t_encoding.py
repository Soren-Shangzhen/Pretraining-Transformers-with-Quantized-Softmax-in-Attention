"""Paper T (training paper) semantic encoding. Colour = calibration, line/fill = surrogate."""
import paperstyle as ps

EXACT = dict(color=ps.OI["black"], linestyle="-")
LERP = dict(color=ps.OI["green"], linestyle="-", marker="o", markerfacecolor=ps.OI["green"])
CAL_COLOR = {"MinMax": ps.OI["vermillion"], "FW": ps.OI["blue"]}
K_SHADE = {4: 0.0, 8: 0.2, 16: 0.35, 32: 0.5, 64: 0.6}   # only when several K of ONE family share a panel


def cond_style(cal, surrogate, k_shade=None):
    """cal in {'MinMax','FW'}; surrogate in {'Weight','Prob'}. Line/marker kwargs."""
    c = CAL_COLOR[cal] if k_shade is None else ps.shade(CAL_COLOR[cal], K_SHADE[k_shade])
    if surrogate == "Weight":
        return dict(color=c, linestyle="-", marker="o", markerfacecolor=c)
    return dict(color=c, linestyle="--", marker="o", markerfacecolor="white")


def bar_style(cal, surrogate):
    c = CAL_COLOR[cal]
    face = c if surrogate == "Weight" else ps.shade(c, 0.65)
    return dict(facecolor=face, edgecolor=c, linewidth=0.8)


def scatter_style(cal_or_lerp, surrogate=None):
    """For scatter plots (Fig 5c, S3): filled = Weight-STE, hollow = Prob-STE."""
    if cal_or_lerp == "LERP":
        return dict(color=ps.OI["green"], facecolors=ps.OI["green"])
    c = CAL_COLOR[cal_or_lerp]
    return dict(edgecolors=c, facecolors=c if surrogate == "Weight" else "white", linewidths=0.8)


# Fig 2 (detach record): colour = gradient mode, independent of the calibration colours
GRAD_MODE = {
    "full": dict(color=ps.OI["black"], linestyle="-"),
    "detach": dict(color=ps.OI["purple"], linestyle="--"),
    "detach+proj": dict(color=ps.OI["orange"], linestyle=(0, (3, 1.5, 1, 1.5))),  # dash-dot, distinct hue from detach
}
