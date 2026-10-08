"""Which fitted model the paper reports.

Per fold, the restart with the best training likelihood is the fit (a Laplace fit whose influence
collapsed counts only if all did). A configuration scores its held-out log-likelihood summed over
folds per held-out transition; a dataset's model is the best allowed configuration, except that on
the random-feature bandwidth grid the smallest bandwidth within one standard error of the best wins.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .results import configs

SCORE = "test_mll_per_transition"
SCORES = Path(__file__).resolve().parents[1] / "figure_data" / "scores.parquet"
PANELS = ["adams", "adams_p02", "adams_p05", "adams_p08", "becker_hubless", "cmv", "house",
          "kozitsin", "markets", "plos_counting", "plos_gauging", "senate", "spinos", "takacs",
          "takacs_control", "takacs_disliking", "takacs_disliking_interior", "takacs_study1"]
PARAMS = ["lmbda", "sigma", "pi", "b_null", "mu_null"]
MF, PW = "neighbour_mean_field", "pairwise"
GAUSS, NOGATE = ["gaussian_nogate"], ["gaussian_nogate", "laplace_nogate"]
LADDER_KERNELS = ["simplified_degroot", "random_fourier"]
RUNG_LABELS = ["0 null", "1 + influence", "2 + best kernel", "3 + heavy tail",
               "4 + latent retention", "5 + latent selection"]
STEPS = [("influence", 1, 0), ("shape", 2, 1), ("tail", 3, 2), ("retention", 4, 3), ("selection", 5, 4)]
TOP_RUNG, GATE_RUNG = 5, 4


def best_restarts(fits):
    """One fit per (configuration, fold)."""
    f = fits.assign(train=[np.nanmax(np.asarray(h, dtype=float)) for h in fits.train_mll_history],
                    collapsed=(fits.fit.map(lambda f: configs()[f].noise_model) == "laplace") & ~(fits.lmbda > 0.005))
    all_collapsed = f.groupby(["fit", "cv_fold_index"]).collapsed.transform("all")
    f = f[~f.collapsed | all_collapsed].sort_values(["fit", "cv_fold_index", "init"])
    return f.loc[f.groupby(["fit", "cv_fold_index"], sort=False).train.idxmax()]


def scores(fits):
    """One row per configuration: held-out score summed over folds, parameters averaged."""
    w, rows = best_restarts(fits), []
    for fit, idx in w.groupby("fit", sort=False).indices.items():
        g, c = {col: w[col].to_numpy()[idx] for col in w.columns}, configs()[fit]
        mll, n = sum(g["test_mll"].tolist()), sum(g["test_n"].tolist())
        row = {"fit": fit, "dataset": fit.split("/")[1], "noise_variant": fit.split("/")[0].rsplit("_", 1)[0],
               "representation": c.representation, "kernel": c.kernel_func,
               "rff_gamma": c.rff_gamma if c.kernel_func == "random_fourier" else np.nan,
               "test_mll": mll, "n_transitions": n, SCORE: mll / n, "num_folds": c.cv_num_folds,
               "phi": list(g["phi"][np.argmax(g["train"])]),
               "per_fold": [{"fold": k, SCORE: s, "n_transitions": t}
                            for k, s, t in zip(g["cv_fold_index"], g[SCORE], g["test_n"])],
               "folds_train_mll": g["train"].tolist()}
        for p in PARAMS:
            values = [v for v in g[p].tolist() if v is not None and v == v]
            row[p], row[f"folds_{p}"] = sum(values) / len(values) if values else np.nan, values
        rows.append(row)
    return pd.DataFrame(rows)


def load_scores(path=SCORES, starved=False):
    """The scores table; `starved` drops configurations scored on 2% fewer held-out transitions than
    the best-covered one of their (dataset, representation)."""
    s = pd.read_parquet(path)
    return s[s.n_transitions >= s.groupby(["dataset", "representation"]).n_transitions.transform("max") * 0.98] \
        if starved else s


def best_model(dataset, representation, *, grain, variants=None, kernels=None, drop=("null", "legendre")):
    """The configuration the paper reports for (dataset, representation), or None."""
    s = grain[[pf is not None and len(pf) >= nf for pf, nf in zip(grain.per_fold, grain.num_folds)]]
    s = s[(s.dataset == dataset) & (s.representation == representation) & ~s.kernel.isin(drop)
          & s.noise_variant.isin(variants or s.noise_variant) & s.kernel.isin(kernels or s.kernel)]
    if s.empty:
        return None
    grids = []  # the one-standard-error bandwidth of each random-feature grid
    for _, cell in s[s.rff_gamma.notna()].groupby(["kernel", "noise_variant"]):
        best = cell.loc[cell[SCORE].idxmax()]
        folds = [f[SCORE] for f in best["per_fold"]]
        if len(cell) > 1 and len(folds) > 1:
            se = float(np.std(folds, ddof=1) / np.sqrt(len(folds)))
            best = cell[cell[SCORE] >= best[SCORE] - se].sort_values("rff_gamma").iloc[0]
        grids.append(best)
    s = pd.concat([s[s.rff_gamma.isna()], pd.DataFrame(grids)]) if grids else s
    return s.loc[s[SCORE].idxmax()]


def rung_spec(net, linear_kernel="simplified_degroot"):
    """[(label, filters)] of the six rungs, each adding one capability. With a network the mean-field
    arm comes first and the pairwise arm enters at the last rung."""
    low, shaped, gated = [MF] if net else [PW], ["null", *LADDER_KERNELS], NOGATE + ["gaussian_gate", "laplace_gate"]
    spec = [(low, GAUSS, ["null"]), (low, GAUSS, ["null", linear_kernel]), (low, GAUSS, shaped),
            (low, NOGATE, shaped), (low, gated, shaped), ([MF, PW] if net else [PW], gated, shaped)]
    return [(label, dict(arms=a, variants=v, kernels=k)) for label, (a, v, k) in zip(RUNG_LABELS, spec)]


def rung(grain, dataset, arms, variants, kernels):
    """The best model of one rung over its arms."""
    rows = [best_model(dataset, a, grain=grain, variants=variants, kernels=kernels, drop=("legendre",)) for a in arms]
    return max((r for r in rows if r is not None), key=lambda r: r[SCORE], default=None)


def ladder(grain, dataset):
    """The six rungs' models; a rung scoring below the one beneath (a 1-SE bandwidth can) repeats that one."""
    rows = []
    for _, kw in rung_spec({MF, PW} <= set(grain[grain.dataset == dataset].representation)):
        r = rung(grain, dataset, **kw)
        rows.append(r if not rows or r[SCORE] >= rows[-1][SCORE] else rows[-1])
    return rows
