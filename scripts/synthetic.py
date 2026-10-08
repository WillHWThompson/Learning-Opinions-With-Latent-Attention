"""The synthetic experiments: python scripts/synthetic.py EXPERIMENT [ARGS] [--out-dir DIR], writing figure_data/
network (synthetic_noise_misspec*), dyadic [N ...] (synthetic_dyadic_noise_misspec*), rzb_capability
(synthetic_rzb_capability.json), mechanism (synthetic_mechanism_direction_*) and hitrates
(synthetic_discrimination_hitrates.csv). discrimination TASK [--device cuda] runs one of the sweep's 960 tasks
(the Snakefile submits all 960) into results/synthetic_discrimination/, which hitrates reads."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kernel_inference import synthetic as S

TRUTHS = ["sbc", "rzb", "degroot"]

def network(out):   # 24 train / 40 test markets of 10 steps, Laplace noise, pi 0.5 or 1, null-arm drift 0.5 sigma
    def grid(truths, pi, rep):
        res = [S.fit_row(d, k, arm, rep) for d in (S.network(t, S.les_miserables(), pi=pi) for t in truths)
               for arm in S.ARMS for k in S.FIT_KERNELS]
        return pd.DataFrame([r for r, _ in res]), pd.concat([c for _, c in res], ignore_index=True)
    pick = lambda f: f[(f.truth_kernel == "rzb") & (f.fit_kernel == "rff") & (f.fit_noise == "laplace_gate_drift")]
    (rows, curves), nmf = grid(TRUTHS, 0.5, "pairwise"), grid(["rzb"], 0.5, "neighbour_mean_field")
    for f, name in [(rows, ""), (pick(rows), "_rzb_pairwise_rff"), (pick(curves), "_shapes_rzb_pairwise_rff"),
                    (grid(TRUTHS, 1.0, "pairwise")[0], "_pi1"), (nmf[0], "_rzb_nmf"), (nmf[1], "_shapes_rzb_nmf")]:
        f.to_csv(out / f"synthetic_noise_misspec{name}.csv", index=False)

def dyadic(out, *sizes):   # two events per subject, half the subjects held out; shapes at 4000 events, with the RFF
    for pi, name in [(0.5, ""), (1.0, "_pi1")]:   # bandwidth chosen on held-out MLL, 256 features
        rows, gsel = [], []
        for d in (S.dyadic(t, pi=pi, n_events=int(n)) for n in sizes or [1000, 4000, 100000] for t in TRUTHS):
            res = [(k, *S.fit_row(d, k, arm)) for arm in S.CANONICAL for k in S.FIT_KERNELS]
            rows += [r for _, r, _ in res]
            if pi == 0.5 and d.meta["n_events"] == 4000:
                gsel += [c for k, _, c in res if k != "rff"] + [S.fit_row(d, "rff", arm, fitted=S.fit_data(
                    d, "rff", arm, scales=S.BANDWIDTHS, features=256))[1] for arm in S.CANONICAL]
        pd.DataFrame(rows).to_csv(out / f"synthetic_dyadic_noise_misspec{name}.csv", index=False)
        if gsel:
            pd.concat(gsel).to_csv(out / "synthetic_dyadic_noise_misspec_shapes_gsel.csv", index=False)

def rzb_capability(out):   # retention: dyadic, 4000 events; selection: les_miserables; both on the rzb truth, pi 0.5
    ci = lambda m, se, n, g: dict(mean=m, se=se, lo=m - 1.96 * se, hi=m + 1.96 * se, n_events=n, n_clusters=g,
                                  resolved=bool(abs(m) > 1.96 * se))
    def panel(d, ll):   # ll(kernel, with the mechanism): held-out ll per event; zero is the null kernel without it
        null, (lb, rb, la, ra) = ll("null", 0).mean(), (ll(k, w) for w in (0, 1) for k in ("degroot", "rff"))
        return dict(null_mll=float(null), base=[float(lb.mean() - null), float(rb.mean() - null)],
                    totals=[float(la.mean() - null), float(ra.mean() - null)],
                    before=ci(*S.paired_contrast(rb, lb, d.cluster)), after=ci(*S.paired_contrast(ra, la, d.cluster)))
    dy, net = S.dyadic("rzb", pi=0.5, n_events=4000), S.network("rzb", S.les_miserables(), pi=0.5)
    result = dict(description="RZB synthetic data used in panels (a) and (b), respectively",
                  interval="node-clustered paired normal 95% interval",
                  retention=panel(dy, lambda k, w: S.fit_data(dy, k, ["laplace_nogate", "laplace_gate"][w],
                                                              scales=S.BANDWIDTHS if k == "rff" else (1,))[3]),
                  selection=panel(net, lambda k, w: S.fit_data(net, k, "laplace_gate_drift",
                                                               ["neighbour_mean_field", "pairwise"][w])[3]))
    (out / "synthetic_rzb_capability.json").write_text(json.dumps(result, indent=2) + "\n")

# Retention events: dyadic, mover share 0.7 or 0.9 exp(-|d| / 1.2). Selection: one of K=6 uniform sources, picked
# uniformly or with p(j) ~ exp(-|d_j| / 0.25). Controls: DeGroot truth, distance-independent mechanism. Fits without
# the mechanism are laplace_nogate or the neighbour mean field; with it, laplace_gate or the pairwise arm.
CELLS = dict(ret_indep=("retention", 0, "rzb"), ret_dep=("retention", 1, "degroot"), sel_indep=("selection", 0, "rzb"),
             sel_dep=("selection", 1, "degroot"), retention_dd_control=("retention", 0, "degroot"),
             selection_hom_control=("selection", 0, "degroot"))
size = lambda G: np.sqrt(np.mean(G ** 2))
tilt = lambda G: (G[-(len(G) // 4):].mean() - G[:len(G) // 4].mean()) / size(G)
shape = lambda G, p="": {p + "S": size(G), p + "C": (C := np.abs(np.diff(G)).sum()), p + "C_over_S": C / size(G),
                         p + "tilt": tilt(G)}   # size, curvature (total variation), tilt

def mechanism(out, n=4000, seeds=range(20261006, 20261016)):   # ten seeds; the figure draws the first
    g = np.random.default_rng(0)
    control_share = np.mean(0.9 * np.exp(-np.abs(g.random(2_000_000) - g.random(2_000_000)) / 1.2))
    lmbda = {"rzb": S.LMBDA,   # DeGroot's S equals SAR's over the 99th percentile of |d|
             "degroot": float(size(S.gain_curve(S.TRUTH["rzb"](), S.LMBDA, np.linspace(0.0, 0.88, 200)))) / 0.15}
    truth_curve = lambda truth, grid: S.gain_curve(S.TRUTH[truth](), lmbda[truth], grid)

    @torch.no_grad()
    def generate(cell, n, seed, scale):   # x, candidates, delta, mu, mover
        (mech, dep, truth), g = CELLS[cell], torch.Generator().manual_seed(seed)
        x = torch.rand((n,), generator=g, dtype=S.F64)
        cand = torch.rand((n, 1 if mech == "retention" else 6), generator=g, dtype=S.F64)
        d, mu = cand - x[:, None], lmbda[truth] * S.TRUTH[truth]()(x[:, None].expand_as(cand), cand).to(S.F64)
        if mech == "retention":
            share = 0.9 * np.exp(-d[:, 0].abs().numpy() / 1.2) if dep else np.full(n, control_share if "dd" in cell else 0.7)
            mover = torch.rand((n,), generator=g, dtype=S.F64) < torch.as_tensor(share)
            eps, eps0 = S.draw((n,), "laplace", scale, g), S.draw((n,), "laplace", S.B_NULL * scale, g)
            mu = torch.where(mover, mu[:, 0], torch.zeros_like(eps))
            return x, cand, mu + torch.where(mover, eps, eps0), mu, mover
        w = torch.exp(-d.abs() / 0.25) if dep else torch.ones_like(d)
        mu = mu.gather(1, torch.multinomial(w / w.sum(1, keepdim=True), 1, generator=g))[:, 0]
        return x, cand, mu + S.draw((n,), "laplace", scale, g), mu, torch.ones(n, dtype=torch.bool)

    def events(x, cand, delta, mean_field):
        att, one = torch.full(cand.shape, 1.0 / cand.shape[1], dtype=S.F64), torch.ones(len(x), dtype=torch.bool)
        cand = (att * cand).sum(-1, keepdim=True) if mean_field else cand
        return S.Events(delta, x, cand, torch.ones_like(cand) if mean_field else att, one, cand == cand)
    summary, shapes = [], []
    for seed in seeds:
        for cell, (mech, dep, truth) in CELLS.items():
            big = generate(cell, 200_000, seed + 999, 1.0)
            scale = float(big[3][big[4]].std()) / np.sqrt(2.0)   # Laplace(b) has sd b sqrt 2
            ev, perm = generate(cell, n, seed, scale), torch.as_tensor(np.random.default_rng(seed).permutation(n))
            k_true = truth_curve(truth, np.linspace(0.0, np.quantile((ev[1] - ev[0][:, None]).abs().numpy(), 0.99), 200))
            for fit in ("without", "with"):
                arm = "laplace_gate" if mech == "retention" and fit == "with" else "laplace_nogate"
                tr, te = (events(*(v[rows].contiguous() for v in ev[:3]), mech == "selection" and fit == "without")
                          for rows in (perm[n // 2:], perm[:n // 2]))
                r, ll = (tr.cand_x - tr.x_self[:, None]).abs().reshape(-1).numpy(), {}
                grid = np.linspace(0.0, float(np.quantile(r, 0.99)), 200)
                for kernel in ("rff", "degroot"):
                    em, _, _, ll[kernel] = S.fit(kernel, tr, te, arm, gamma=1 / float(np.median(r)), rmax=float(r.max()),
                                                 seed=seed)
                    G = S.gain_curve(em.k, em.lmbda, grid)
                    summary.append(dict(cell=cell, seed=seed, fit=fit, kernel=kernel, **shape(G), **shape(k_true, "truth_"),
                                        heldout_mll=ll[kernel].mean(), noise_scale=scale))
                    shapes.append(pd.DataFrame(dict(cell=cell, seed=seed, fit=fit, kernel=kernel, r=grid,
                                                    k_true=truth_curve(truth, grid), k_fit=G)))
                gap = ll["rff"] - ll["degroot"]
                summary[-2].update(gap_rff_minus_degroot=gap.mean(), gap_se=gap.std(ddof=1) / np.sqrt(gap.size))
    pd.DataFrame(summary).to_csv(out / "synthetic_mechanism_direction_summary.csv", index=False)
    pd.concat(shapes).to_csv(out / "synthetic_mechanism_direction_shapes.csv", index=False)

# Discrimination: truth kernel x noise x retention (pi, drift in sigmas) x selection DGP, 16/64/256 train markets,
# each dataset fitted with six kernels, six noise arms and both representations. On each axis a level scores its
# best fit's held-out MLL; argmax picks the best level, one-SE the simplest within one paired node-clustered SE.
RETENTIONS = {"none": (1.0, 0.0), "pi0.5": (0.5, 0.0), "pi0.25": (0.25, 0.0), "pi0.5_drift": (0.5, 0.5)}
CELLS_D = list(itertools.product(["degroot", "sbc", "rzb", "bc"], ["gaussian", "laplace"], RETENTIONS.items(),
                                 ["pairwise", "mean_field"]))
SIZES, TEST_MARKETS = [16, 64, 256], {16: 8, 64: 32, 256: 64}
KCLASS = dict(null="null", degroot="linear", bc="nonlinear", rzb="nonlinear", sbc="nonlinear", rff="nonlinear")
ORDER = dict(kernel=dict(null=0, degroot=1, bc=2, rzb=2, sbc=3, rff=4), kernel_class=dict(null=0, linear=1, nonlinear=2),
             degroot_vs_rff=dict(degroot=0, rff=1), noise=None, retention=dict(nogate=0, gate=1),
             drift=dict(nodrift=0, drift=1), selection=dict(mean_field=0, pairwise=1))   # simplicity order of the levels

def level(arm, k, rep, axis):   # a fit's level on an axis, None to leave it out
    noise, gate = arm.split("_", 1)
    return dict(kernel=k, kernel_class=KCLASS[k], degroot_vs_rff=k if k in ("degroot", "rff") else None, noise=noise,
                retention=None if gate == "gate_drift" else gate, drift={"gate": "nodrift", "gate_drift": "drift"}.get(gate),
                selection="pairwise" if rep == "pairwise" else "mean_field")[axis]

def axis_answers(ll, cluster, axis):
    order, best = ORDER[axis], {}
    for key, v in ll.items():   # the first fit wins a tie
        if (lv := level(*key, axis)) is not None and (lv not in best or v.mean() > best[lv][0]):
            best[lv] = (v.mean(), key)
    win, *rest = sorted(best, key=lambda lv: -best[lv][0])
    gaps = {lv: S.paired_contrast(ll[best[win][1]], ll[best[lv][1]], cluster)[:2] for lv in rest}
    pool = [win] + [lv for lv, (d, se) in gaps.items() if not d > se]
    one_se = win if len(pool) == 1 else "undecided" if order is None else max(
        (lv for lv in pool if order[lv] == min(order[p] for p in pool)), key=lambda lv: best[lv][0])
    return dict(argmax_level=win, one_se_level=one_se, runner_up=rest[0], gap_to_runner=gaps[rest[0]][0],
                se_to_runner=gaps[rest[0]][1], best_mll=float(best[win][0]), best_fit=json.dumps(list(best[win][1]))), best

def discrimination(out, task, device="cpu"):
    size, cell_id, s0 = [(n, c, b) for n in SIZES for c in range(len(CELLS_D)) for b in range(0, 10, 2)][int(task)]
    (kernel, noise, (retention, (pi, drift)), dgp), rows = CELLS_D[cell_id], []
    truth = dict(kernel=kernel, kernel_class=KCLASS[kernel], degroot_vs_rff="degroot" if kernel == "degroot" else "rff",
                 noise=noise, retention="gate" if pi < 1 else "nogate", drift="drift" if drift else "nodrift", selection=dgp)
    for s in (s0, s0 + 1):
        seed = 20260930 + 100003 * cell_id + 1009 * SIZES.index(size) + s
        d = S.network(kernel, S.les_miserables(), pi=pi, family=noise, drift=drift, n_train=size,
                      n_test=TEST_MARKETS[size], seed=seed, dgp=dgp, device=device)
        ll = {(arm, k, rep): S.fit_data(d, k, arm, rep, device=device)[3] for arm in S.ARMS
              for k in ["null", "degroot", "rff", "sbc", "rzb", "bc"] for rep in ("pairwise", "neighbour_mean_field")}
        meta = dict(dataset=f"c{cell_id:02d}_n{size}_s{s}", cell_id=cell_id, seed_index=s, seed=seed, truth_kernel=kernel,
                    truth_noise=noise, truth_retention=retention, truth_dgp=dgp, truth_pi=pi, truth_sigma=d.scale,
                    n_train_markets=size, n_test_markets=TEST_MARKETS[size])
        for axis, t in truth.items():
            ans, best = axis_answers(ll, d.cluster, axis)
            rows.append(dict(meta, axis=axis, truth_level=t, **ans, argmax_hit=ans["argmax_level"] == t,
                             one_se_hit=ans["one_se_level"] == t, one_se_undecided=ans["one_se_level"] == "undecided",
                             truth_mll_best=float(best[t][0]) if t in best else np.nan))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"summary_task{int(task):04d}.parquet", index=False)

def hitrates(out, src="results/synthetic_discrimination"):
    keys, rows = ["axis", "truth_level", "n_train_markets"], []
    for key, g in pd.concat(map(pd.read_parquet, sorted(Path(src).glob("summary_task*.parquet")))).groupby(keys):
        hit, und = g.argmax_hit.mean(), g.one_se_undecided.mean()
        rows += [dict(zip(keys, key), rule="argmax", n=len(g), hit=hit, undecided=0.0, miss=1 - hit),
                 dict(zip(keys, key), rule="one_se", n=len(g), hit=g.one_se_hit.mean(), undecided=und,
                      miss=1 - g.one_se_hit.mean() - und)]
    pd.DataFrame(rows).sort_values(keys + ["rule"]).to_csv(out / "synthetic_discrimination_hitrates.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment"), ap.add_argument("args", nargs="*"), ap.add_argument("--device", default="cpu")
    a = (ap.add_argument("--out-dir", type=Path), ap.parse_args())[1]
    out = a.out_dir or Path("results/synthetic_discrimination" if a.experiment == "discrimination" else "figure_data")
    globals()[a.experiment](out, *a.args, **({"device": a.device} if a.experiment == "discrimination" else {}))
