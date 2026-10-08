"""Colours, line styles, names and the dataset order shared by every figure and table."""
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, LIGHT, HAIR = "#1A1A1A", "#767676", "#A6A6A6", "#DCDCDC"
RETENTION, SELECTION, TAIL, INFLUENCE = "#3F4BA0", "#9B5FD0", "#17868F", "#C4552B"
WIDTH = 7.0                                   # PNAS text width, inches
PAGE, NAVY, MONO = "#FFFFFF", "#0D3B66", "Latin Modern Mono"   # MONO is the paper's \texttt

KERNELS = {  # name, short name, dash, marker
    "bounded_confidence": ("Bounded Confidence", "BC", (0, (6, 4)), "s"),
    "sigmoidal_bounded_confidence": ("Sigmoidal BC", "SBC", (0, (5, 1.6, 1, 1.6)), "D"),
    "simplified_degroot": ("DeGroot", "DeGroot", (0, (1, 1.6)), "o"),
    "rzb": ("SAR", "SAR", (0, (3, 1.5)), "^"),
    "random_fourier": ("RFF", "RFF", "solid", "v"),
    "legendre": ("Legendre", "Legendre", (0, (8, 2, 1, 2, 1, 2)), "P"),
    "null": ("Null", "Null", "solid", "x"),
}
NOISE = {  # name, colour, marker, dash
    "gaussian_nogate": ("Gaussian", GREY, "o", "solid"),
    "laplace_nogate": ("Laplace", INK, "D", (0, (6, 4))),
    "gaussian_gate": ("Gaussian + mixture", RETENTION, "o", "solid"),
    "laplace_gate": ("Laplace + mixture", SELECTION, "D", (0, (6, 4))),
}
RUNG_COLORS = [GREY, INK, INK, TAIL, RETENTION, SELECTION]

GROUPS = [  # heading, [(dataset, label)]
    ("Controlled (known source exposure)", [
        ("adams_p02", "ADAMS-low"), ("adams_p05", "ADAMS-medium"), ("adams_p08", "ADAMS-high"),
        ("takacs_control", "TAKACS-S2-control"), ("takacs_disliking", "TAKACS-S2-disliking"),
        ("takacs_disliking_interior", "TAKACS-interior"), ("takacs_study1", "TAKACS-S1"),
        ("becker_hubless", "BECKER")]),
    ("Controlled (latent source exposure)", [("plos_gauging", "VK-gauging"), ("plos_counting", "VK-counting")]),
    ("Political", [("senate", "SENATE"), ("house", "HOUSE")]),
    ("Social media", [("cmv", "CMV"), ("spinos", "SPINOS"), ("kozitsin", "VKontakte")]),
    ("Prediction markets", [("markets", "MARKETS")]),
]
LABEL = {d: label for _, rows in GROUPS for d, label in rows} | {"adams": "ADAMS", "takacs": "TAKACS"}
ORDER = [d for _, rows in GROUPS for d, _ in rows]   # the 16 datasets the figures show
NETWORK = ["plos_gauging", "plos_counting", "senate", "house", "cmv", "spinos", "kozitsin", "markets"]


# The manuscript's two older rcParams sets: PAPER (Computer Modern, left titles) and LEGACY (Palatino, centred
# titles, wider margin), and PNAS, the type sizes PAPER figures take at the full text width.
_TICKS = {"xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
          "xtick.major.width": .6, "ytick.major.width": .6, "legend.frameon": False, "axes.spines.top": False,
          "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
          "font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
          "xtick.labelsize": 7, "ytick.labelsize": 7}
PAPER = _TICKS | {"font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
                  "axes.linewidth": .8, "axes.edgecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 1.3,
                  "lines.solid_capstyle": "round", "figure.dpi": 300, "savefig.pad_inches": .01}
LEGACY = _TICKS | {"font.serif": ["Palatino", "Georgia", "Times New Roman", "DejaVu Serif"], "axes.linewidth": .7,
                   "axes.edgecolor": "#333333", "lines.linewidth": 1.4, "figure.dpi": 150,
                   "axes.prop_cycle": mpl.cycler(color=[INK])}
PNAS = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 5.4}
LADDER = dict(row=.32, bar=.65, head=.8, legend=.57, label=10.6, heading=10.2, key=10.2, total=9.8, axis=10, tick=9.2)


def setup(*rc):
    """No arguments: the default style. Otherwise rcParams reset, then each dict of `rc` applied in turn."""
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mono = ROOT / "kernel_inference" / "fonts" / "lmmono10-regular.otf"
    if mono.exists():
        mpl.font_manager.fontManager.addfont(str(mono))
    if rc:
        mpl.rcdefaults()
        return [mpl.rcParams.update(d) for d in rc]
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Palatino", "CMU Serif", "DejaVu Serif"],
        "font.monospace": ["Latin Modern Mono", "DejaVu Sans Mono"], "mathtext.fontset": "cm",
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlelocation": "left", "lines.linewidth": 1.3,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
    })


def grid(n, ncols=4, height=1.9, width=WIDTH, rc=(), **kw):
    """n panels in rows of ncols; unused panels are hidden. Returns (fig, axes as a flat list)."""
    setup(*rc)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * nrows), squeeze=False,
                             **{"layout": "constrained"} | kw)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, list(axes.ravel())


def pictogram(ax, x, y, r, k, span, h, w=None, frac=.8, tr=None, box="none", fill=None, lw=.8, **line):
    """k(r) in a box h points high and w wide centred on (x, y) of `tr` (data by default): r spans frac of the
    width, k frac of the height per `span`, centred on its midrange; the box has edge `box` and face `fill`."""
    h, w = h / 144, (w or h) / 144
    tr = ax.figure.dpi_scale_trans + mpl.transforms.ScaledTranslation(x, y, tr or ax.transData)
    ax.add_artist(mpl.patches.Rectangle((-w, -h), 2 * w, 2 * h, transform=tr, fill=fill is not None, facecolor=fill,
                                        edgecolor=box, lw=.75 * lw, zorder=line.get("zorder", 3), clip_on=False))
    ax.add_artist(mpl.lines.Line2D(frac * w * (2 * (r - r[0]) / (r[-1] - r[0]) - 1), frac * h * (2 * k - k.max() - k.min())
                                   / span, transform=tr, lw=lw, solid_capstyle="round", dash_capstyle="round",
                                   clip_on=False, **{"zorder": 3} | line))


def pack(want, sp, lo, hi):
    """Positions near the ascending `want`, at least `sp` apart and inside [lo, hi]: overlapping neighbours merge
    into a cluster centred on their mean, so a crowd spreads evenly around where it belongs."""
    cl, start = [], lambda c: max(lo, min(c[0] / c[1] - (c[1] - 1) * sp / 2, hi - (c[1] - 1) * sp))
    for d in want:
        cl.append([d, 1])
        while len(cl) > 1 and start(cl[-2]) + cl[-2][1] * sp > start(cl[-1]) + 1e-12:
            cl[-2:] = [[cl[-2][0] + cl[-1][0], cl[-2][1] + cl[-1][1]]]
    return [start(c) + i * sp for c in cl for i in range(c[1])]


def save(fig, name):
    out = ROOT / "output" / "paper" / "figures" / f"{name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))
