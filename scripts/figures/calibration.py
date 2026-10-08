"""Calibration and curvature of the fitted models: held-out PIT values and the RFF gain on every ladder rung.

    python scripts/figures/calibration.py export [dataset ...]   (no datasets: all 18)
    python scripts/figures/calibration.py [name ...]             (no names: every figure)

The PIT of an update is the model's predictive CDF at it, so it needs the fitted model, not just the
stored held-out log-likelihoods. Each held-out update is scored by its own fold's fit, folds pooled.
"""
import json
import math
import sys
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from kernel_inference import results as R, selection as S
from kernel_inference.style import LABEL, LIGHT, ORDER, PAPER, PNAS, RUNG_COLORS, grid, save

FD, SC, DG, RFF = S.SCORES.parent, S.SCORE, "simplified_degroot", "random_fourier"
DASH = ["solid", "solid", (0, (6, 3)), "solid", "solid", "solid"]  # rungs 1 and 2 share ink
GAUSS, MAX_PAIRS, N_GRID = ["gaussian_nogate"], 2_000_000, 200
G, FITS = cache(lambda: S.load_scores(starved=True)), cache(R.fits)


@cache
def winners(fit):  # the fit of each fold
    return list(S.best_restarts(FITS()[FITS().fit == fit]).sort_values("cv_fold_index").iterrows())


@torch.no_grad()
def pit(em, ev):
    u = (ev.attention * ev.cand_valid * em.noise.cdf(ev.delta[:, None] - em.lmbda * em.interaction(ev), em.sigma)).sum(-1)
    u = u / (ev.attention * ev.cand_valid).sum(-1).clamp_min(1e-300)
    if em.c.infer_pi:
        u = (1 - em.pi) * em.noise.cdf(ev.delta - em.mu_null, em.b_null if em.c.decoupled_scale else em.sigma) + em.pi * u
    return u[ev.valid].numpy()


@cache
def heldout(fit):
    u = np.concatenate([pit(R.model(row), (f := R.fold(row)).events(f.test)) for _, row in winners(fit)])
    return u, {"fit": fit, "folds": len(winners(fit)), "n_events": int(u.size), "ks": float(stats.kstest(u, "uniform").statistic)}


def ad(u):  # Anderson-Darling A^2 against U(0, 1)
    u, n = np.sort(np.clip(u, 1e-12, 1 - 1e-12)), len(u)
    return float(-n - np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(u) + np.log1p(-u[::-1]))) / n)


def net(ds):
    return {S.MF, S.PW} <= set(G()[G().dataset == ds].representation)


def compared(ds):  # the models the kernel-comparison figures put side by side
    best = lambda arm, **kw: S.best_model(ds, arm, grain=G(), **kw)
    if net(ds):
        yield from [("1 NMF, Gaussian no-gate", best(S.MF, variants=GAUSS)), ("2 NMF, best noise", best(S.MF)),
                    ("3 Pairwise, best noise", best(S.PW))]
    yield from [("Gaussian no-gate RFF (base)", best(S.PW, variants=GAUSS, kernels=[RFF])), ("Selected", best(S.PW))]


@cache
def support(fit):  # every (transition, candidate) distance of the panel on the fit's arm
    ev = (f := R.fold(winners(fit)[0][1])).events(f.train | f.test)
    r = (ev.cand_x - ev.x_self[:, None]).abs()[ev.cand_valid & ev.valid[:, None]].numpy()
    return r[np.random.default_rng(0).choice(r.size, MAX_PAIRS, replace=False)] if r.size > MAX_PAIRS else r


def summarise(v):
    v, K = np.asarray(v), len(v)
    sd = float(np.std(v, ddof=1)) if K > 1 else None
    return {"K": K, "mean": float(v.mean()), "sd": sd, "nb_se": sd * math.sqrt(1 / K + 1 / (K - 1)) if K > 1 else None}


@torch.no_grad()
def rff_folds(fit, r, x):
    out = []
    for _, row in winners(fit):
        em = R.model(row)
        gain = lambda t: em.lmbda * em.k.gain(torch.as_tensor(t, dtype=torch.float64)).numpy()
        Gr, w = gain(r), r ** 2 / (r ** 2).sum()
        lam = float((w * Gr).sum())  # the constant gain closest to G, weight r^2
        out.append({"fold": int(row.cv_fold_index), "lmbda": em.lmbda, "lambda_star": lam,
                    "nl": float((w * (Gr - lam) ** 2).sum() / (w * Gr ** 2).sum()), "G_grid": gain(x).tolist()})
    return out


def nl_panel(ds):
    """From rung 2 up, the best RFF model in each rung's feasible set whether or not it wins there."""
    chain, grids, out, prev = S.ladder(G(), ds), {}, [], None
    for i, (label, kw) in enumerate(S.rung_spec(net(ds))):
        row = chain[i] if i < 2 else S.rung(G(), ds, **dict(kw, kernels=[RFF]))
        row = prev = prev if i > 2 and row[SC] < prev[SC] else row
        dg = S.rung(G(), ds, **dict(kw, kernels=[DG])) if DG in kw["kernels"] else None
        rec = {"label": label, "rung": i, "kernel": row.kernel, "noise_variant": row.noise_variant,
               "representation": row.representation, "fit": row.fit,
               "rff_gamma": None if np.isnan(row.rff_gamma) else float(row.rff_gamma), "mll": float(row[SC]),
               "same_as_prev": i > 0 and row.fit == out[-1]["fit"], "chain_kernel": chain[i].kernel,
               "degroot_mll": None if dg is None else float(dg[SC]),
               "beats_degroot": None if dg is None else bool(row[SC] > dg[SC]), "folds": []}
        if row.kernel == RFF:
            r = support(row.fit)
            x = grids.setdefault(row.representation, np.linspace(0, float(np.quantile(r, .99)), N_GRID))
            rec["folds"] = rff_folds(row.fit, r, x)
            rec["summary"] = {k: summarise([f[k] for f in rec["folds"]]) for k in ("nl", "lambda_star")}
        out.append(rec)
    return {"dataset": ds, "has_representation_axis": net(ds), "grid": {k: v.tolist() for k, v in grids.items()}, "rungs": out}


def export(datasets=S.PANELS):
    pit, rungs, ads = {}, {}, {}
    for ds in datasets:
        for label, row in compared(ds):
            u, m = heldout(row.fit)
            pit[f"PIT__{ds}__{label}"] = u, dict(m, kernel=row.kernel, noise_variant=row.noise_variant, representation=row.representation)
        ads[ds], first = {}, {}
        for i, row in enumerate(S.ladder(G(), ds)):
            key, (u, m) = first.setdefault(row.fit, f"RUNG__{ds}__{i}"), heldout(row.fit)
            rungs.setdefault(key, (u, dict(m, dataset=ds, rungs=[])))[1]["rungs"].append(i)
            ads[ds][str(i)] = {"label": S.RUNG_LABELS[i], "fit": row.fit, "pit_key": f"canonical_pit_rungs.npz::{key}",
                               "n": int(u.size), "ad": ad(u), "kernel": row.kernel, "noise": row.noise_variant}
    for name, new in (("canonical_pit", pit), ("canonical_pit_rungs", rungs)):
        np.savez_compressed(FD / f"{name}.npz", **{k: u for k, (u, _) in new.items()},
                            meta=json.dumps({k: m for k, (_, m) in new.items()}))
    (FD / "rung_pit_ad.json").write_text(json.dumps(ads, indent=1) + "\n")
    (FD / "nonlinearity_by_rung_rff.json").write_text(json.dumps({"panels": [nl_panel(d) for d in sorted(datasets)]}, indent=1) + "\n")


def qq_by_rung():  # one QQ curve per distinct fit of a panel's ladder, labelled with its rungs and A^2
    ads, q = json.loads((FD / "rung_pit_ad.json").read_text()), (np.arange(200) + .5) / 200
    fig, axes = grid(len(ORDER), 6, 1.55, rc=(PAPER, {k: .95 * v for k, v in PNAS.items()}))
    for k, (ax, ds) in enumerate(zip(axes, ORDER)):
        fits = {}  # older exports name the fit by its sweep id
        [fits.setdefault(e.get("fit", e.get("sweep_id")), []).append(int(i)) for i, e in ads[ds].items()]
        ax.plot([0, 1], [0, 1], color=LIGHT, lw=.6, zorder=1)
        for n, same in enumerate(sorted(fits.values(), key=min)):
            e, c = ads[ds][str(same[0])], RUNG_COLORS[max(same)]
            file, _, key = e["pit_key"].rpartition("::")
            ax.plot(q, np.quantile(np.clip(np.load(FD / file)[key], 0, 1), q), color=c, lw=1.1, ls=DASH[max(same)], zorder=3)
            ax.text(.97, .03 + .105 * (len(fits) - 1 - n), f"{'+'.join(map(str, same))}: {e['ad']:.1f}", color=c,
                    ha="right", va="bottom", fontsize=5.2, transform=ax.transAxes)
        ax.set_title(LABEL[ds], fontsize=5.6, pad=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], yticks=[0, 1], aspect="equal",
               xlabel="uniform quantile" if k >= 12 else "", ylabel="PIT quantile" if k % 6 == 0 else "")
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=1.4, ls=DASH[i], label=name) for i, (c, name) in enumerate(
        zip(RUNG_COLORS, ["0 null", "1 infl", "2 kern", "3 tail", "4 ret", "5 sel"]))], loc="outside upper center", ncol=6, fontsize=6)
    return fig


FIGS = {"fig_si_qq_by_rung": qq_by_rung}
if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2:] or S.PANELS)
    for name in [] if sys.argv[1:2] == ["export"] else sys.argv[1:] or FIGS:
        save(FIGS[name](), name)
