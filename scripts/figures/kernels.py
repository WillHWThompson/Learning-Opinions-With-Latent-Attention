"""Fitted kernels with Fisher bands over the observed updates, their latent mechanisms and the alternatives.
`kernels.py export [dataset ...]` writes kernel_curves.json, kernel_events.npz and canonical_r_support.json afresh;
`kernels.py [name ...]` draws the figures (no names: all)."""
import copy
import json
import sys
from dataclasses import replace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from kernel_inference import model as M, results as R, selection as S
from kernel_inference.panels import load_panel
from kernel_inference.style import GREY, INK, KERNELS, LABEL, NETWORK, ORDER, RETENTION, SELECTION, save

FD, SC, Z, G = S.SCORES.parent, S.SCORE, 1.96, S.load_scores(starved=True)
DG, RFF, GAUSS, GATED = "simplified_degroot", "random_fourier", ["gaussian_nogate"], ["gaussian_gate", "laplace_gate"]
FREE_PHI = ["sigmoidal_bounded_confidence", "rzb"]  # elsewhere phi is pinned, a step, or a ridge solve
DYADIC = ["becker_hubless", "takacs_study1", "takacs_control", "takacs_disliking", "takacs_disliking_interior",
          "adams_p02", "adams_p05", "adams_p08"]
NETS = ["plos_counting", "plos_gauging", "kozitsin", "senate", "house", "cmv", "spinos", "markets"]

best = lambda ds, arms, **kw: max([r for a in arms if (r := S.best_model(ds, a, grain=G, **kw)) is not None],
                                  key=lambda r: r[SC], default=None)
def picks(ds):
    net, both, dgr = ds in NETWORK, [S.PW, S.MF], [DG, RFF]
    si = {k: r for k in [DG, "bounded_confidence", *FREE_PHI, RFF] if (r := best(ds, both, variants=GATED, kernels=[k])) is not None}
    return dict(best=best(ds, [S.PW], kernels=dgr), base=best(ds, [S.MF if net else S.PW], variants=GAUSS, kernels=dgr),
                rungs=[best(ds, [S.MF], variants=GAUSS), best(ds, [S.MF]), best(ds, [S.PW])] if net else
                [best(ds, [S.PW], variants=GAUSS, kernels=[RFF]), best(ds, [S.PW])],
                si=si, null=best(ds, both, variants=GATED, kernels=["null"], drop=()))

def information(em, ev):
    """(names, outer product of per-event scores) of the parameters off their bounds (a Laplace -LL has no usable Hessian)."""
    c, em = em.c, copy.deepcopy(em)
    phi = em.k.effective_phi().detach().double().tolist() if c.kernel_func in FREE_PHI else []
    bounds = {"lmbda": [c.lmbda_min], "pi": [c.pi_min, c.pi_max], "b_null": [c.b_null_min]}
    free = [p for p, on in [("lmbda", 1), ("sigma", 1), ("pi", c.infer_pi), ("b_null", c.decoupled_scale), ("mu_null", c.infer_mu_null)]
            if on and not any(b is not None and abs(getattr(em, p) - b) <= 1e-3 * max(abs(b), 1e-12) for b in bounds.get(p, []))]
    n, theta = len(phi), torch.tensor(phi + [float(getattr(em, p)) for p in free], dtype=torch.float64)
    def ll(t):
        em.__dict__.update(zip(free, t[n:]))
        em.k.__dict__.update({"effective_phi": lambda: t[:n]} if n else {})
        return em.log_likelihood(ev)
    s = torch.stack([torch.func.jvp(ll, (theta,), (e,))[1].detach() for e in torch.eye(len(theta), dtype=torch.float64)], 1)
    return [f"phi[{i}]" for i in range(n)] + free, (s.T @ s).numpy()

def kernel_band(em, names, m, r):
    """95% half-width of lmbda k(r), Monte Carlo over phi's (else lambda's) block of the inverse information."""
    w, n, k = np.linalg.eigvalsh(m), sum(p.startswith("phi") for p in names), copy.deepcopy(em.k)
    cov = np.linalg.pinv(m, rcond=1e-8) if (w < 1e-8 * max(abs(w).max(), 1)).any() else np.linalg.inv(m)
    i, theta = (list(range(n)), k.effective_phi().detach().double()) if n else ([names.index("lmbda")], [em.lmbda])
    d = np.random.default_rng(0).multivariate_normal(theta, cov[np.ix_(i, i)] + 1e-10 * np.eye(len(i)), 2000)
    k.effective_phi = lambda: torch.tensor(d.T[..., None])
    y = em.lmbda * k.gain(torch.tensor(r)).detach().reshape(len(d), -1).numpy() if n else d * np.ones_like(r)
    return np.diff(np.percentile(y[np.isfinite(y).all(1)], [2.5, 97.5], axis=0), axis=0)[0] / 2

def gpr_band(em, f, r):
    """Predictive band of ridge coefficients, sigma^2 (B + sigma^2 I)^-1, B the basis Gram over training pair distances."""
    rng, act = np.random.default_rng(0), torch.nonzero(f.train)
    (m, i, t), j = act[torch.as_tensor(rng.integers(0, len(act), 50_000))].T, torch.as_tensor(rng.integers(0, f.x.shape[1], 50_000))
    s = (f.x[m, i, t] - f.x[m, j, t]).abs()
    b = em.k.raw_basis(s[(f.adj[m, i, j] > 0) & (s > 1e-8)].float()).double()
    gram, s2 = b.T @ b * (float(f.train.sum()) / len(b)), em.sigma ** 2
    cov, q = s2 * torch.linalg.inv(gram + s2 * torch.eye(len(gram)).double()), em.k.raw_basis(torch.tensor(r).float()).double()
    return (Z * em.lmbda * torch.einsum("ri,ij,rj->r", q, cov, q).clamp(min=0).sqrt()).numpy()

def curve(rec, band, r_max):
    """lambda k(r) of a fit on its named fold, with its band and the joint information's SEs if `band`."""
    em, f, r = R.model(rec), R.fold(rec), np.linspace(0, r_max, 300)
    out = dict(kernel=em.c.kernel_func, noise_variant=rec.fit.split("/")[0].rsplit("_", 1)[0], is_gate=em.c.infer_pi,
               fold=int(rec.cv_fold_index), r=r.tolist(), curve=(em.lmbda * em.k.gain(torch.tensor(r)).detach()).tolist(),
               **{p: None if getattr(em, p) is None else float(getattr(em, p)) for p in S.PARAMS})
    if band:
        names, m = information(em, f.events(f.train))
        out.update(params=names, cond=float(np.linalg.cond(m)),
                   se=np.sqrt(np.diag(np.linalg.inv(m))).tolist() if np.linalg.eigvalsh(m).min() > 0 else None,
                   band=(gpr_band(em, f, r) if hasattr(em.k, "raw_basis") else kernel_band(em, names, m, r)).tolist())
    return out

def events(folds):
    """Held-out updates, each under its fold's fit, on the recorded opinions; per (update, candidate) gap, E-step B, attention."""
    out, n = {}, 0
    for _, rec in folds.iterrows():
        em, f = R.model(rec), R.fold(rec)
        ev = (replace(f, x=load_panel(f.c.panel, f.c.no_opinion_value).x.double()) if f.c.dequantize else f).events(f.test)
        v, B, w = ev.valid, em.e_step(ev), ev.attention * ev.cand_valid
        e, j = torch.nonzero(ev.cand_valid[v], as_tuple=True)
        new = dict(delta=ev.delta[v], gap=((w / w.sum(-1, keepdim=True) * ev.cand_x).sum(-1) - ev.x_self)[v], pair_ev=e + n,
                   pair_gap=(ev.cand_x - ev.x_self[:, None])[v][e, j], pair_B=B[v][e, j], pair_a=ev.attention[v][e, j],
                   gamma=em.gamma[v] if em.c.infer_pi else e[:0])
        out, n = {k: out.get(k, []) + [a] for k, a in new.items()}, n + int(v.sum())
    return {k: torch.cat(a).numpy() for k, a in out.items() if sum(map(len, a))}

def support(ds):
    """Histogram of the pair distances |x_j - x_i| at scored transitions, its max and 99th percentile."""
    p, out = load_panel(ds), []
    for m, nb in enumerate(p.adj > 0):
        live = p.transition_mask[m] & nb.any(-1)[:, None]
        out += [(p.x[m, None, :, t] - p.x[m, :, t, None]).abs()[nb & live[:, t, None] & p.node_mask[m, None, :, t]]
                for t in torch.nonzero(live.any(0)).ravel().tolist()]
    r = torch.cat(out)[torch.randperm(sum(map(len, out)), generator=torch.Generator().manual_seed(0))[:4_000_000]]  # bounds memory
    r, top = r[r > 0], r.max().item()
    return dict(max=top, p99=torch.quantile(r, 0.99).item(), hist=torch.histc(r, 60, 0, top).int().tolist(),
                edges=np.linspace(0, top, 61).tolist())

def export(datasets):
    fits, r_max = R.fits(), json.loads((S.SCORES.parents[1] / "config/canonical_r_max.json").read_text())
    cur, ev, sup = {}, {}, {}
    for ds in datasets:
        p = picks(ds)
        banded = {p["best"].fit, p["base"].fit, *(r.fit for r in p["rungs"])}
        for fit in banded | {r.fit for r in p["si"].values()}:  # the named fold: the fold winner with the best training LL
            folds = S.best_restarts(fits[fits.fit == fit])
            cur[fit] = dict(dataset=ds, **curve(folds.loc[folds.train.idxmax()], fit in banded, r_max[ds]))
            ev.update({f"{k}__{ds}": a for k, a in events(folds).items()} if fit == p["best"].fit else {})
        sup[ds] = support(ds)
    (FD / "kernel_curves.json").write_text(json.dumps(cur))
    np.savez_compressed(FD / "kernel_events.npz", **ev)
    (FD / "canonical_r_support.json").write_text(json.dumps(sup, indent=1) + "\n")

# ---------------------------------------------------------------- figures, styled as the committed PDFs
A = lambda p, *keys: [np.asarray(p[k], float) for k in keys]
TERRA, TERRA_INK, LIGHT, HAIR, HIST = "#C4552B", "#A8441F", "#A6A6A6", "#DCDCDC", "#D1D5DB"
DASH, CONTEXT, ZERO = (0, (6, 4)), (0, (1.4, 2.4)), (0, (2.5, 2))
SHORT = {"gaussian_nogate": "G", "laplace_nogate": "L", "gaussian_gate": "G+mix", "laplace_gate": "L+mix"}
PLATE = dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.88)  # backing for corner text
NAME = {"senate": "U.S. Senate", "house": "U.S. House", "kozitsin": "Kozitsin", "becker_hubless": "Becker", "spinos": "SPINOS",
        "markets": "Manifold Markets", "cmv": "r/CMV", "plos_gauging": "Vande Kerckhove (gauging)",
        "plos_counting": "Vande Kerckhove (counting)", "takacs_control": "Takács (control)", "takacs_disliking": "Takács (disliking)",
        "takacs_disliking_interior": "Takács (disliking, interior)", "adams_p02": "Adams (rel. 0.2)", "adams_p05": "Adams (rel. 0.5)",
        "adams_p08": "Adams (rel. 0.8)", "takacs_study1": "Takács (study 1)"}
PAPER = {"font.family": "serif", "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
         "font.monospace": ["Latin Modern Mono", "DejaVu Sans Mono"], "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
         "legend.fontsize": 7.5, "legend.frameon": False, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
         "axes.edgecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
         "lines.solid_capstyle": "round", "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42}
PNAS = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.4}
GAINS = {DG: M.DeGroot, "bounded_confidence": M.BoundedConfidence, "sigmoidal_bounded_confidence": M.SigmoidalBoundedConfidence, "rzb": M.RZB}
key = lambda c, lw=1.3, ls="-", **kw: Line2D([], [], color=c, lw=lw, ls=ls, **kw)
dot = lambda c, m, ms: key(c, 0, marker=m, ms=ms, mec="white", mew=0.5)
note = lambda ax, xy, text, **kw: ax.annotate(text, xy, xycoords="axes fraction", **kw)
se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
CI = (Patch(facecolor=GREY, alpha=0.3, lw=0), "95% CI (observed information)")
subplots = lambda nrows, ncols, size, rc=PAPER, **kw: (mpl.rcParams.update(PAPER | rc), plt.subplots(
    nrows, ncols, figsize=size, squeeze=False, facecolor="white", **kw))[1]
if (MONO := S.SCORES.parents[1] / "kernel_inference/fonts/lmmono10-regular.otf").exists():  # the paper's \texttt
    mpl.font_manager.fontManager.addfont(str(MONO))

def ribbon(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)  # hairline edges keep a thin band visible
    ax.plot(x, lo, x, hi, color=color, lw=0.45, alpha=0.75, zorder=2, solid_capstyle="butt")
    return [lo.min(), hi.max()]

def quantum(v):
    """Lattice spacing of a quantised variable (refit through the origin), 0 if continuous."""
    tol = 1e-5 * (float(np.max(np.abs(v))) or 1.0)
    u = np.unique(np.round(v / tol) * tol)
    d = np.diff(u)[np.diff(u) > 1.5 * tol]
    if u.size < 8 or not d.size or np.ptp(u) / d.min() > 5000 or (np.diff(k := np.round(u / d.min())) == 0).any():
        return 0.0
    q = float((k * u).sum() / (k * k).sum())
    return q if np.max(np.abs(u - k * q)) <= max(0.02 * q, 3 * tol) else 0.0

def means(ax, x, y, w, b, kview, color, marker, ms, z):
    """Weighted bin means (kernel view: the slope through 0) with 95% bars, of bins with 12 effective updates."""
    out = []
    for ww, xx, yy in ((w[m], x[m], y[m]) for m in ((b == i) & (w > 0) for i in range(12))):
        if ww.size and ww.sum() ** 2 / (ww ** 2).sum() >= 12:
            mean = (ww * yy).sum() / (den := (ww * xx).sum() if kview else ww.sum())
            out.append(((ww * xx).sum() / ww.sum(), mean, Z * np.sqrt((ww ** 2 * (yy - mean * xx ** kview) ** 2).sum()) / den))
    xs, ys, es = zip(*out) if out else ([], [], [])
    ax.errorbar(xs, ys, yerr=es, fmt=marker, ms=ms, color=color, mec="white", mew=0.4, lw=0, elinewidth=1, capsize=0, zorder=z)
    return [m + s * e for _, m, e in out for s in (-1, 1)]

def arms_panel(ax, ds, kview, rng):
    """Updates folded toward the source against |d|, the selected model's arms, a baseline, posterior-weighted means."""
    p, sc = picks(ds), (lambda r: 1.0) if kview else (lambda r: r)
    best, (rb, cb), (r, cv, bd) = C[p["best"].fit], A(C[p["base"].fit], "r", "curve"), A(C[p["best"].fit], "r", "curve", "band")
    d, g, gam, pe, pg, pb = (EV.get(f"{k}__{ds}") for k in ("delta", "gap", "gamma", "pair_ev", "pair_gap", "pair_B"))
    keep, (pe, pg, pb) = g != 0, (a[(g[pe] != 0) & (pg != 0)] for a in (pe, pg, pb))
    x, y, px, py = abs(g[keep]), (d * np.sign(g))[keep], abs(pg), d[pe] * np.sign(pg)
    ax.axhline(0, color=HAIR, lw=0.7, ls=ZERO, zorder=0)
    cy = py + rng.uniform(-0.8, 0.8, py.size) * quantum(py)  # the cloud, jittered within a recording lattice cell
    cx = abs(pg + rng.uniform(-0.8, 0.8, pg.size) * quantum(pg))
    cy = cy / np.maximum(cx, 1e-12) if kview else cy
    rgba = np.tile(to_rgba(GREY), (pb.size, 1))
    rgba[:, 3] = np.clip(1500 / max(pb.sum(), 1), 0.14, 0.6) * (0.22 + 0.78 * np.clip(pb, 0, 1))
    ax.scatter(cx, cy, s=3, c=rgba, lw=0, zorder=1, rasterized=True)
    ax.plot(rb, cb * sc(rb), color=GREY, lw=1.3, ls=DASH, zorder=2)
    ax.plot(r, cv * sc(r), color=TERRA, lw=2.3, zorder=6)
    shown, ok = [*np.percentile(cy, [1, 99]), (cv * sc(r)).min(), (cv * sc(r)).max()], best["cond"] <= 1e6
    if ok or best["kernel"] == RFF:  # the RFF band is a predictive band, not the information's
        shown += ribbon(ax, r, (cv - bd) * sc(r), (cv + bd) * sc(r), TERRA)
    if best["is_gate"] and not kview:
        mu, s = best["mu_null"], dict(zip(best["params"], best["se"] or [])).get("mu_null")
        ax.plot(r, 0 * r + mu, color=RETENTION, lw=2.3, zorder=6)
        shown += ribbon(ax, r, 0 * r + mu - Z * s, 0 * r + mu + Z * s, RETENTION) if ok and s else [mu]
    bin_of = lambda v: np.minimum(np.digitize(v, np.linspace(0, r[-1], 13)) - 1, 11)
    shown += means(ax, x, y, np.ones_like(y), bin_of(x), kview, INK, "o", 4.2, 7)
    if gam is not None and best["is_gate"] and not kview:
        shown += means(ax, x, y, 1 - gam[keep], bin_of(x), kview, RETENTION, "s", 4.0, 8)
    _, u = np.unique(pe * 14 + bin_of(px), return_inverse=True)  # an update's pairs within one bin count once
    w, ub = np.bincount(u, pb), np.zeros(u.max() + 1, int)
    ub[u] = bin_of(px)
    with np.errstate(all="ignore"):
        shown += means(ax, np.bincount(u, pb * px) / w, np.bincount(u, pb * py) / w, w, ub, kview, TERRA_INK, "^", 4.6, 8)
    lo, hi = min(shown), max(shown)
    ax.set(xlim=(0, r[-1]), ylim=(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)))
    ax.set_title(LABEL[ds], fontsize=8, pad=4, family="monospace")
    text = [f"$\\pi$ = {best['pi']:.2f}" if best["is_gate"] else "no gate in selected model"]
    text += [] if ok else [f"no band: cond {best['cond']:.0e}"]
    note(ax, (0.97, 0.03), "\n".join(text), ha="right", va="bottom", fontsize=6, color=INK, linespacing=1.35)

def arms(name, roster, kview=False):
    nrows, rng = -(-len(roster) // 4), np.random.default_rng(0)
    fig, axes = subplots(nrows, 4, (7.8, 2.65 * nrows + 0.5), {"axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    [ax.axis("off") for ax in axes.flat[len(roster):]]
    for i, (ax, ds) in enumerate(zip(axes.flat, roster)):
        arms_panel(ax, ds, kview, rng)
        ax.set(xlabel="opinion discrepancy  $|d_{e}|$" if i >= len(roster) - 4 else "", ylabel="" if i % 4 else  # each column's lowest
               "kernel  $\\delta_i\\,\\mathrm{sign}(d) / |d|$" if kview else "update toward the source  $\\delta_e\\,\\mathrm{sign}(d_{e})$")
    keys = [(key(GREY, 1.3, DASH), "baseline, no latent mechanism"), (key(TERRA, 2.3), "influence arm  " + (
        "$\\lambda\\,k(|d|)$" if kview else "$\\lambda\\,k(|d_{e}|)\\,d_{e}$")), (key(RETENTION, 2.3), "retention arm  $\\mu_0$"),
            (dot(INK, "o", 3.4), "empirical, $\\sum \\delta_e\\,\\mathrm{sign}(d) / \\sum |d|$ per bin" if kview else
             "binned mean, 95% interval"),
            (dot(TERRA_INK, "^", 3.8), "updates at $d_{e,s}=x_s-x_e^-$, weight $\\bar{z}_{e,s}$"),
            (dot(RETENTION, "s", 3.4), "updates, weight $1-\\Sigma_s \\bar{z}_{e,s}$"), CI]
    keys = [k for i, k in enumerate(keys) if not (kview and i in (2, 5))]
    fig.legend(*zip(*keys), loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.005), handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.5 / fig.get_figheight(), 1, 1))
    save(fig, name)

RUNGS = {True: ["ref, NMF", "NMF", "pairwise"], False: ["reference", "selected"]}
MECH = {"reference": (LIGHT, CONTEXT, "reference specification"), "baseline": (GREY, "-", "baseline unimodal"),
        "retention": (RETENTION, "-", "+ latent retention"), "selection": (SELECTION, "-", "+ latent neighbor selection"),
        "both": (SELECTION, "-", "+ both latents")}

def qq_inset(ax, pits):  # held-out PIT QQ curves with their KS distances, in the panel's empty top corner
    ins = ax.inset_axes((0.085, 0.60, 0.30, 0.36), zorder=12)
    ins.patch.set(facecolor="white", alpha=0.94)
    for i, (v, color, alpha) in enumerate(pits):
        q, step = (np.arange(v.size) + 0.5) / v.size, max(1, v.size // 400)
        ins.plot(q[::step], v[::step], color=color, lw=1.1, alpha=alpha)
        note(ins, (0.06, 0.80 - 0.20 * i), f"{np.max(np.abs(v - q)):.3f}", fontsize=4.4, color=color)
    ins.plot([0, 1], [0, 1], ls=ZERO, lw=0.7, color=LIGHT)
    ins.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ins.spines[:].set_linewidth(0.4)
    ins.set_title("PIT-QQ, KS", fontsize=4.6, pad=1.2)

def comparison(name, datasets, net):
    """Each rung's move lambda k(d) d over the binned moves and the histograms of d and of the moves; the best rung banded."""
    fig, axes = subplots(2, 4, (7 * 4 / 3, 2 * 2.55), {"font.size": 7}, layout="constrained")
    used = set()
    for j, (ax, ds) in enumerate(zip(axes.flat, datasets)):
        rows, d, g, (e, h) = picks(ds)["rungs"], EV[f"delta__{ds}"], EV[f"gap__{ds}"], A(SUP[ds], "edges", "hist")
        win, ro, yo, tops, pits = np.argmax([r[SC] for r in rows]), abs(g[g != 0]), (d * np.sign(g))[g != 0], [], []
        mid, tw, r_max = (e[1:] + e[:-1]) / 2, ax.twinx(), C[rows[0].fit]["r"][-1]  # the histogram of d, behind the curves
        tw.bar(mid[mid <= r_max], h[mid <= r_max], width=0.9 * (e[1] - e[0]), color=HIST, alpha=0.55, zorder=1)
        tw.set(ylim=(0, h[mid <= r_max].max() * 3.5), yticks=[])
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.axhline(0, color=HAIR, lw=0.7, ls=ZERO, zorder=0)
        for i, (short, row) in enumerate(zip(RUNGS[net], rows)):
            p, won, (r, cv, bd) = C[row.fit], i == win, A(C[row.fit], "r", "curve", "band")
            cell = "reference" if i == 0 and not won else ["baseline", "retention", "selection", "both"][
                p["is_gate"] + 2 * (net and row.representation == S.PW)]
            (color, ls, _), used = MECH[cell], used | {cell}
            if won:
                ribbon(ax, r, (cv - bd) * r, (cv + bd) * r, color)
            ax.plot(r, cv * r, color=color, ls=ls if cell == "reference" or p["kernel"] == RFF else DASH, zorder=6 if won else 4,
                    lw=2.3 if won else 1.1 if cell == "reference" else 1.3)
            tops.append(np.nanmax(cv * r))
            note(ax, (0.985, 0.963 - 0.058 * i), f"{short}: {KERNELS[p['kernel']][1]} {SHORT[p['noise_variant']]}", ha="right",
                 va="top", fontsize=4.9, color=color, zorder=8, bbox=PLATE)
            if p["pi"] < 0.999:  # a gated fit: pi and mu_0 with their fold SEs, and the band's typical relative width
                note(ax, (0.015, 0.035 + 0.058 * (sum(C[q.fit]["pi"] < 0.999 for q in rows[:i]))), f"$\\pi$={p['pi']:.2f}$\\pm$"
                     f"{se(row.folds_pi):.2f}  $\\mu_0$={p['mu_null']:+.3f}$\\pm${se(row.folds_mu_null):.3f}  CI $\\pm$"
                     f"{100 * np.nanmedian(bd / np.maximum(abs(cv), 1e-12)):.1f}%", fontsize=4.8, color=color, zorder=8, bbox=PLATE)
            pits += [(np.sort(PIT[ds, row.fit]), color, 1.0 if won else 0.7)] if (ds, row.fit) in PIT else []
        edges = np.linspace(0, ro.max(), 13)
        bins = [(edges[b:b + 2].mean(), v) for b in range(12) if (v := yo[np.digitize(ro, edges[1:-1]) == b]).size >= 6]
        n, ys, ses = (np.array([f(v) for _, v in bins]) for f in (len, np.mean, se))
        ax.scatter([c for c, _ in bins], ys, s=6 + 90 * n / n.max(), color=INK, zorder=7, lw=0)
        cnt, ye = np.histogram(yo, 44, range=ax.get_ylim())  # the moves' histogram, growing in from the right edge
        (ty := ax.twiny()).set_zorder(ax.get_zorder() - 1)
        ty.barh((ye[1:] + ye[:-1]) / 2, cnt, height=0.92 * (ye[1] - ye[0]), color=HIST, edgecolor="none", zorder=1)
        ty.set(xlim=(cnt.max() * 8, 0), xticks=[], yticks=[])
        ty.spines[:].set_visible(False)
        lo, hi = np.percentile(yo, [1, 99])
        ax.set(xlim=(0, r_max), ylim=(lo, lo + (max(hi, *tops, np.max(ys + Z * ses)) - lo) / 0.6), xlabel="$d = |x_j - x_i|$",
               ylabel="" if j % 4 else "$\\Delta x_i \\cdot \\mathrm{sign}(x_j - x_i)$")
        ax.set_title(f"({chr(97 + j)}) {NAME[ds]}", fontweight="bold")
        qq_inset(ax, pits)
        note(ax, (0.985, 0.02), f"$\\Delta$LL {rows[-1][SC] - rows[-2][SC]:+.2f}", ha="right", fontsize=5.2, color=GREY, zorder=8, bbox=PLATE)
    keys = [(key(c, ls=ls), lab) for k, (c, ls, lab) in MECH.items() if k in used] + [(key(GREY), "non-parametric"),
            (key(GREY, ls=DASH), "parametric"), CI, (key(INK, 0, marker="o", ms=4), "binned mean (area $\\propto$ n)")]
    fig.legend(*zip(*keys), loc="outside lower center", ncol=-(-len(keys) // 2), fontsize=5.6, handlelength=1.8, columnspacing=1.6)
    save(fig, name)

def mll_inset(ax, fits, winner, null, top):  # held-out MLL per transition, the gap to the winner; the null dashed
    ins, ks = ax.inset_axes([0.05, 0.11, 0.34, 0.36], zorder=7), list(fits)
    ys, gap = np.arange(len(ks))[::-1], [fits[k][SC] - top for k in ks]
    pad = max(-0.15 * min(gap), 0.004)
    ins.patch.set_facecolor("white")
    ins.axvline(0.0, color=HAIR, lw=0.4, zorder=0)
    if null is not None and null[SC] - top >= min(gap) - pad:  # off the left edge it is not drawn
        ins.axvline(null[SC] - top, color=GREY, lw=0.6, ls=CONTEXT, zorder=0)
    for y, k, x in zip(ys, ks, gap):
        ins.plot([x], [y], "o", ms=2.8, mfc=INK if k == winner else GREY, mec="none", zorder=6 if k == winner else 4)
    ins.set(xlim=(min(gap) - pad, max(0.0, max(gap)) + 0.4 * pad), ylim=(-0.6, len(ks) - 0.1), yticks=list(ys))
    ins.xaxis.set_major_locator(plt.MaxNLocator(2, prune="lower"))
    ins.yaxis.tick_right()
    for t, k in zip(ins.set_yticklabels([KERNELS[k][1] for k in ks], fontsize=3.9), ks):
        t.set_color(INK if k == winner else GREY)
    ins.tick_params(axis="y", length=0, pad=1.5)
    ins.tick_params(axis="x", labelsize=3.9, length=1.5, pad=1)
    ins.spines[["top", "right", "left"]].set_visible(False)
    ins.spines["bottom"].set_linewidth(0.4)

def si_full(name, datasets):
    """Every kernel's best fit with a latent mechanism, faded past r_99; the winner in ink unless it ties the null."""
    fig, axes = subplots(4, 4, (7.0, 4 * 1.75), PNAS | {"xtick.labelsize": 5.4, "ytick.labelsize": 5.4}, layout="constrained")
    for i, (ax, ds) in enumerate(zip(axes.flat, datasets)):
        (fits, null), sup = (picks(ds)[k] for k in ("si", "null")), SUP[ds]
        top, grid_r, inside, under = max(fits.values(), key=lambda r: r[SC]), np.linspace(0, max(sup["max"], 1e-6), 240), [], []
        winner = None if null is not None and top[SC] - null[SC] < 1e-3 else top.kernel
        ax.axhline(0, color=HAIR, lw=0.7, ls=ZERO, zorder=0)
        for k, row in fits.items():  # parametric: the named fold's lambda k(r) over the support; RFF: its exported curve
            r, y = A(C[row.fit], "r", "curve") if k == RFF else (grid_r, C[row.fit]["lmbda"] * GAINS[k](
                list(row.phi)).gain(torch.tensor(grid_r)).detach().double().numpy())
            color, lw, z = (INK, 2.3, 6) if k == winner else (LIGHT if k == RFF else GREY, 1.3, 4)
            m, out = r <= sup["p99"], max(np.argmax(r > sup["p99"]) - 1, 0) if (r > sup["p99"]).any() else r.size
            ax.plot(r[m], y[m], color=color, ls=KERNELS[k][2], lw=lw, zorder=z)
            ax.plot(r[out:], y[out:], color=color, ls=KERNELS[k][2], lw=lw, alpha=0.28, zorder=2)  # faded past r_99
            inside.append(y[m])
            under.append(y[r <= 0.51 * grid_r[-1]])  # the curve over the inset's width and its tick labels
        ax.axvline(sup["p99"], color=GREY, lw=0.4, ls=":", zorder=0)
        lo, hi = (f(np.percentile(np.concatenate(inside), q), 0.0) for f, q in ((min, 1), (max, 99)))  # 0 always shown
        lo, hi = lo - 0.08 * (hi - lo or 1.0), hi + 0.08 * (hi - lo or 1.0)
        need = (min((v.min() for v in under if v.size), default=hi) - 0.52 * hi) / 0.48  # floor lowered to clear the inset
        ax.set_ylim(max(min(lo, need), lo - 0.8 * (hi - lo)), hi)
        ax.set_title(LABEL[ds] + ("" if winner else " (null, tied)"), fontsize=6.3, pad=2)
        ax.set(ylabel="" if i % 4 else "$\\lambda\\,k(|d_{e}|)$", xlabel="$|d_{e}|$" if i + 4 >= len(datasets) else "")
        mll_inset(ax, fits, winner, null, top[SC])
    fig.legend([key(INK, 0, marker="o", ms=3.2, mec="none"), key(GREY, 0.7, CONTEXT)],
               ["inset: held-out MLL per transition, gap to the winner", "null"], loc="outside lower center", ncol=2, fontsize=6)
    save(fig, name)

FOUR = ["plos_counting", "plos_gauging", "adams_p05", "senate"]
ALL = DYADIC[5:] + ["becker_hubless", "takacs_control", "takacs_disliking", "takacs_study1", "plos_gauging", "plos_counting",
                    "senate", "house", "markets", "spinos", "kozitsin", "cmv"]
FIGURES = {"fig_kernel_arms_comparison_delta_latent": lambda n: arms(n, FOUR), "fig_si_kernel_arms_all": lambda n: arms(n, ALL),
           "fig_kernel_arms_comparison_kernel": lambda n: arms(n, FOUR, kview=True),
           "dyadic_kernel_comparison_full": lambda n: comparison(n, DYADIC, False),
           "pairwise_kernel_comparison_full": lambda n: comparison(n, NETS, True),
           "fig_si_kernel_comparison_full": lambda n: si_full(n, DYADIC[5:] + DYADIC[1:5] + DYADIC[:1] + NETS)}

if __name__ == "__main__" and sys.argv[1:2] == ["export"]:
    export(sys.argv[2:] or ORDER)
elif __name__ == "__main__":
    C, SUP = (json.loads((FD / n).read_text()) for n in ("kernel_curves.json", "canonical_r_support.json"))
    EV, pit = dict(np.load(FD / "kernel_events.npz")), np.load(FD / "canonical_pit.npz")  # canonical_pit: the PIT export
    PIT = {(k.split("__")[1], m["fit"]): pit[k] for k, m in json.loads(str(pit["meta"])).items()}
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name](name)
