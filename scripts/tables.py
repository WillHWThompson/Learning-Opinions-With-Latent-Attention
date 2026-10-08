"""The appendix tables: python scripts/tables.py [name ...] writes output/paper/tables/<name>.tex (no name: all)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.optimize import brentq
from kernel_inference.panels import read_panel
from kernel_inference.selection import LADDER_KERNELS, MF, NOGATE, PANELS, PARAMS, PW, SCORE, STEPS, best_model, load_scores, rung_spec
from kernel_inference.style import GROUPS, KERNELS, NETWORK, NOISE, ORDER

FD, OUT = (ROOT := Path(__file__).resolve().parents[1]) / "figure_data", ROOT / "output/paper/tables"
MACRO = dict(zip(PANELS, "adams adamsLo adamsMid adamsHi becker cmv house kozitsin markets vkc vkg senate spinos takacs takacsCtl "
                 "takacsDis takacsInt takacsOne".split()))
KERNS = ["null", "simplified_degroot", "bounded_confidence", "sigmoidal_bounded_confidence", "rzb", "random_fourier"]
NOISES, DG, RFF, ARM = list(NOISE), KERNS[1], KERNS[5], {PW: "pairwise", MF: "mean field"}
GATED, SUBSET = NOISES[2:], "takacs_disliking_interior"   # a subset of takacs_disliking, left out of the ladder tables
S, Z = load_scores(), np.load(FD / "kernel_events.npz")
LADDER, NL = ({p["dataset"]: p for p in json.loads((FD / f).read_text())["panels"]} for f in ("capability_ladder.json",
                                                                                            "nonlinearity_by_rung_rff.json"))
rungs, m = lambda d: {r["rung"]: r for r in NL[d]["rungs"]}, lambda d: rf"\{MACRO[d]}{{}}"
line, ok = lambda *cells: " & ".join(map(str, cells)) + r" \\", lambda b: r"\checkmark" if b else r"$\times$"
short, nz = lambda k: KERNELS[k][1], lambda v: NOISE[v][0].replace(" +", "")
f3, sig2 = lambda v: "---" if v != v else f"{v:.3f}", lambda v: "$<$0.001" if v < 0.001 else f"{v:#.2g}"
def fmt(v, digits=3):   # blank for NaN, scientific outside [0.01, 1000)
    mant, _, e = f"{v:.{digits - 1}e}".partition("e")
    return "" if v != v else rf"${mant}\times10^{{{int(e)}}}$" if v and not 1e-2 <= abs(v) < 1e3 else f"{v:.{digits}g}"
def tex(cols, head, body, env="tabular", small=None, note=""):   # small: (font size, tabcolsep); note: a footnote-size line below
    lines = [rf"\begin{{{env}}}{{{cols}}}", r"\toprule", *head, r"\midrule", *[r"\endhead"][:env == "longtable"], *body, r"\bottomrule",
             rf"\end{{{env}}}" + "}" * bool(small), note and r"\par\smallskip{\footnotesize %s}" % note]
    return "\n".join(([r"{%s\setlength{\tabcolsep}{%gpt}" % small] if small else []) + lines).rstrip() + "\n"
def sections(items, ncol, row=str, sep=r"\midrule", label=m):   # a heading line per non-empty section, then its rows
    return [x for i, (h, k) in enumerate([(h, k) for h, k in items if k])
            for x in [sep][:i > 0] + [rf"\multicolumn{{{ncol}}}{{l}}{{{label(h)}}} \\"] + list(map(row, k))]
def grouped(keep, ncol, row):   # the panels in keep, SUBSET aside, under their italic group headings
    return sections([(h, [d for d, _ in rows if d in keep and d != SUBSET]) for h, rows in GROUPS], ncol, row, r"\addlinespace",
                    lambda h: rf"\textit{{{h}}}")
best = lambda d, **kw: max([r for a in (PW, MF) if (r := best_model(d, a, grain=S, **kw)) is not None],
                           key=lambda r: r[SCORE], default=None)
shape = lambda r, phi=None, hat="": ", ".join(f"{q:.3g}" for q in ([] if r.kernel == DG else r.phi if phi is None else phi)) or (
    rf"${hat}\gamma={r.rff_gamma:.3g}$" if r.kernel == RFF else "")
def named(r):   # the fold with the best training likelihood, the one whose phi the scores carry
    i = int(np.argmax(r.folds_train_mll))
    return {p: r[f"folds_{p}"][i] if len(r[f"folds_{p}"]) else np.nan for p in PARAMS} | {"phi": list(r.phi)}
def size_curvature(r, keep=slice(None)):   # L2 norm and total variation of the gain on its grid, means over folds
    G = [np.asarray(f["G_grid"])[keep] for f in r["folds"] if f.get("G_grid")]
    return np.mean([np.sqrt(np.mean(g ** 2)) for g in G]), np.mean([np.abs(np.diff(g)).sum() for g in G])

def si_panel_summary():
    pit, body = np.load(FD / "canonical_pit.npz"), []
    for d in ORDER:
        units, obs = read_panel(d)
        t = obs.groupby(["sim_id", "batch_idx"]).timestep.max()   # transitions per trajectory
        n = S[(S.dataset == d) & (S.kernel == "null") & (S.noise_variant == NOISES[0]) & (S.representation == PW)].n_transitions.iloc[0]
        body.append(line(m(d), f"{round(n):,}", f"{len(pit[next(k for k in pit.files if k.startswith(f'PIT__{d}__'))]):,}",
                         f"{units.node_num.max():,}", t.min() if t.min() == t.max() else f"{t.min()}--{t.max()}",
                         "async" if (units.update_mode == "asynchronous").all() else "sync"))
    return tex("lrrrrl", [r"panel & $|\cE|$ & scored & $N_{\max}$ & $T$ & update \\"], body)
def params(arm):   # dagger: lambda moves over 10x across folds; double dagger: b_0 on its floor in every fold
    def row(r, top):
        lam, b0 = sorted(r.folds_lmbda) or [0], bool(sum(b <= 1.0001e-5 for b in r.folds_b_null) == 5)
        lmbda = "" if r.kernel == "null" else fmt(r.lmbda) + r"$^{\dagger}$" * bool(lam[0] > 0 and lam[-1] / lam[0] > 10)
        gate = [fmt(r.pi), fmt(r.b_null) + r"$^{\ddagger}$" * b0, fmt(r.mu_null)] if r.noise_variant in GATED else [""] * 3
        c = [short(r.kernel), nz(r.noise_variant), shape(r), lmbda, fmt(r.sigma), *gate, f"{r[SCORE]:.4f}", f"{r[SCORE] - top[SCORE]:+.4f}"]
        return line(*[rf"\textbf{{{x}}}" if x and r.fit == top.fit else x for x in c])
    groups = [(d, [row(r, top) for k in KERNS for v in NOISES
                   if (r := best_model(d, arm, grain=S, kernels=[k], variants=[v], drop=())) is not None])
              for d in ORDER if (top := best_model(d, arm, grain=S)) is not None]
    head = r"kernel & noise & $\phi$ & $\lambda$ & $\sigma$ / $b$ & $\pi$ & $b_0$ & $\mu_0$ & MLL & $\Delta$MLL \\"
    return tex("llllllllrr", [head], sections(groups, 10), env="longtable")
si_params_pairwise, si_params_meanfield = lambda: params(PW), lambda: params(MF)
def ladder_rows():   # {dataset: [(rung, step gain, gain over the null, Anderson-Darling A^2 of the held-out PIT)]}
    z = np.load(FD / "canonical_pit_rungs.npz", allow_pickle=True)
    a2 = lambda x: -len(x) - np.sum((2 * np.arange(1, len(x) + 1) - 1) * np.log(x * (1 - x[::-1]))) / len(x)
    ad = {(q["dataset"], i): a2(np.sort(np.clip(z[k], 1e-12, 1 - 1e-12)))
          for k, q in json.loads(str(z["meta"])).items() for i in q["rungs"]}
    step = lambda p, i: p["steps"][STEPS[i - 1][0]]["value"] / 2 if i and p["steps"][STEPS[i - 1][0]] else np.nan
    return {d: [(r, step(p, i), r["mll"] - p["rungs"][0]["mll"], ad.get((d, i), np.nan)) for i, r in enumerate(p["rungs"])]
            for d, p in LADDER.items()}
def si_rung_table():
    row = lambda i, r, s, vs, a: line(r["label"], short(r["kernel"]), nz(r["noise_variant"]), ARM[r["representation"]], f"{r['mll']:.3f}",
        ("n/a" if i == 5 else "") if s != s else f"{s:+.3f}", f"{vs:+.3f}" if i else "", f"{a:.1f}" if a == a else "")
    return tex("llllrrrr", [r"rung & kernel & noise & arm & MLL & $\Delta$MLL step & vs null & AD $A^2$ \\"],
               sections([(d, [row(i, *x) for i, x in enumerate(rows)]) for d, rows in ladder_rows().items()], 8), env="longtable")
def ladder_summary(compact):   # step gains, total, rung-2 kernel, then the last AD A^2 or the final kernel
    def final(d):   # does RFF beat DeGroot at the top rung, under the one-SE and argmax bandwidths
        f = rung_spec(LADDER[d]["has_representation_axis"])[-1][1]
        top = lambda g, k: max(r[SCORE] for a in f["arms"]
                               if (r := best_model(d, a, grain=g, variants=f["variants"], kernels=[k])) is not None)
        nonlinear = {bool(top(g, RFF) > top(g, DG)) for g in (S, S.assign(rff_gamma=np.nan))}
        return r"depends on $\gamma$" if len(nonlinear) > 1 else ["linear", "nonlinear"][nonlinear.pop()]
    lad = ladder_rows()
    last = lambda d: next(("%.1f (%d)" % (a, i) for i, (*_, a) in reversed(list(enumerate(lad[d]))) if a == a), "---")
    body = [line(m(d), *[f3(s) for _, s, _, _ in lad[d][1:]], f3(lad[d][-1][2]), short(lad[d][2][0]["kernel"]),
                 last(d) if compact else final(d)) for d in ([d for d in lad if d not in ("adams", "takacs")] if compact else ORDER)]
    head = [r"panel & influence & shape & tail & retention & selection & total & rung-2 kernel & final kernel \\",
            r"panel & infl. & shape & tail & ret. & sel. & total & rung 2 & AD final (rung) \\"][compact]
    return tex("lrrrrrrl" + "lr"[compact], [head], body, small=(r"\small", 2.5) if compact else None)
si_rung_gains, si_ladder_compact = lambda: ladder_summary(False), lambda: ladder_summary(True)
def si_winners():
    body = [line(m(d), short(r.kernel), nz(r.noise_variant), ARM[r.representation], shape(r), fmt(r.lmbda), fmt(r.sigma),
                 *([fmt(r.pi), fmt(r.mu_null)] if r.noise_variant in GATED else ["", ""]), f"{r[SCORE]:.3f}")
            for d in ORDER if (r := best(d)) is not None]
    return tex("lllllrrrrr", [r"panel & kernel & noise & arm & $\phi$ & $\lambda$ & $\sigma$ / $b$ & $\pi$ & $\mu_0$ & MLL \\"], body)
def si_winners_compact():   # a tie (1e-3 nats) with the null, or with no mixture, reports the simpler model
    body = []
    for d in ORDER:
        r, null, plain = best(d), best(d, kernels=["null"], drop=()), best(d, variants=NOGATE)
        r = null if r[SCORE] - null[SCORE] < 1e-3 else plain if r.noise_variant in GATED and r[SCORE] - plain[SCORE] < 1e-3 else r
        arm = [{PW: "latent selection", MF: "neighbor aggregation"}[r.representation]] * (d in NETWORK and r.kernel != "null")
        model = ", ".join(["null (tied)" if r.kernel == "null" else short(r.kernel), nz(r.noise_variant)] + arm)
        phi = ((f := named(r))["phi"] or [])[::-1 if r.kernel == "sigmoidal_bounded_confidence" else 1]   # theta = (width, steepness)
        body.append(line(m(d), model, shape(r, phi, r"\hat"), "" if r.kernel == "null" else fmt(f["lmbda"], 2), fmt(f["sigma"], 2),
                         fmt(f["pi"], 2) if r.noise_variant in GATED else "", f"{r[SCORE]:.3f}"))
    return tex("llrrrrr", [r"dataset & model (kernel, noise, arm) & $\hat\phi$ / $\hat\gamma$ & $\hat\lambda$ & $\hat\eta_1$ & $\hat\pi$ & "
                           r"MLL/transition \\"], body, small=(r"\footnotesize", 2))
def dmll(names, show, select):   # gap to the row's best cell; a dagger marks the selected model where it is not that cell
    body, gap = [], lambda v: f"{v:.3f}" if abs(v) >= 1e-3 else f"{v:.4f}"
    for d in ORDER:
        cell, pick = {n: select(d, n) for n in names}, best(d)
        ref = max(r[SCORE] for r in cell.values() if r is not None)
        body.append(line(m(d), *["---" if r is None else r"\textbf{0}" if r[SCORE] == ref else gap(r[SCORE] - ref)
                                 + r"$^\dagger$" * (n in (pick.kernel, pick.noise_variant)) for n, r in cell.items()]))
    return tex("l" + "r" * len(names), [line("panel", *map(show, names))], body, small=(r"\small", 3.5), note=(
        r"Gap in held-out MLL per transition to the best cell in the row (bold, 0). "
        r"$^\dagger$: the model the 1-SE rule selects when it is not the best cell."))
si_dmll_kernel_compact = lambda: dmll(KERNS, short, lambda d, k: best(d, kernels=[k], drop=()))
si_dmll_noise_compact = lambda: dmll(NOISES, nz, lambda d, v: best(d, variants=[v]))
def si_synthetic_2e_compact():
    s = S[S.dataset == "synthetic_2e_network"].assign(mll=lambda f: f[SCORE].round(3))
    s = s.assign(gap=s.mll - s.mll.max(), row=s.representation.map(ARM) + " / " + s.noise_variant.map(nz))
    ks = [k for k in KERNS[1:] if k in set(s.kernel)]
    body = [line(row, *[r"\textbf{0}" if abs(v) < 5e-4 else f"{v:.3f}" for v in g.set_index("kernel").gap[ks]])
            for row, g in s.groupby("row")]
    return tex("l" + "r" * len(ks), [line("arm / noise", *map(short, ks))], body, small=(r"\small", 3), note=(
        r"Synthetic data, truth: SBC kernel, Laplace mixture noise, $\pi=0.5$, $\lambda=0.5$, $\eta_1=0.01$. "
        r"Gap in held-out MLL per transition to the best cell (bold, 0). "))
SOURCES = [f"synthetic_{s}" for s in ("noise_misspec noise_misspec_pi1 noise_misspec_rzb_nmf noise_misspec_rzb_pairwise_rff "
                                      "dyadic_noise_misspec dyadic_noise_misspec_pi1").split()]   # the rzb runs carry their own arm
SYN = {"bc": "BC", "sbc": "SBC", "degroot": "DeGroot", "rzb": "RZB", "rff": "RFF"}
syn_noise = lambda n: n.replace("_nogate", "").replace("_gate", " gate").replace("_drift", " + drift").capitalize()
def discrimination():   # per synthetic dataset: best restart per cell by training MLL, then the best held-out cell
    for i, name in enumerate(SOURCES):
        setting = "dyadic" if "dyadic" in name else "network, pi " + ("1" if name.endswith("pi1") else "0.5")
        f = pd.read_csv(FD / f"{name}.csv").pipe(lambda f: f[f.fit_kernel.isin(list(SYN))])
        f = f.assign(representation=f.get("representation", None if "rzb" in name else PW))
        f = f.loc[f.groupby(["dataset", "fit_kernel", "fit_noise", "representation"]).train_mll_per_event.idxmax()]
        for _, s in f.loc[f.groupby("dataset").test_mll_per_event.idxmax()].iterrows():
            gate, n = s.truth_pi < 1, int((re.search(r"_n(\d+)$", s.dataset) or [0, 0])[1])
            truth = f"{s.truth_noise}_{'gate' if gate else 'nogate'}" + "_drift" * bool(gate and s.truth_mu_null != 0)
            yield i, n, setting + f", n={n:,}" * bool(n), s, truth
def si_synthetic_discrimination():
    body = [line(setting, SYN[s.truth_kernel], syn_noise(t), fmt(s.truth_pi), f"{SYN[s.fit_kernel]} {ok(s.fit_kernel == s.truth_kernel)}",
                 f"{syn_noise(s.fit_noise)} {ok(s.fit_noise == t)}", ARM[s.representation],
                 *map(fmt, s[["pi_hat", "truth_mu_null", "mu_null_hat", "truth_lmbda", "lmbda_hat", "shape_nrmse_weighted"]]))
            for _, _, setting, s, t in discrimination()]
    return tex("lllcllclccccc", [r"Setting & True kernel & True noise & $\pi$ & Selected kernel & Selected noise & Rep. & $\hat\pi$ & "
                                 r"$\mu_0$ & $\hat\mu_0$ & $\lambda$ & $\hat\lambda$ & nRMSE \\"], body, env="longtable")
def si_synthetic_discrimination_compact():
    rows, size, body = list(discrimination()), {1000: "10^3", 4000: r"4{\times}10^3", 100000: "10^5"}, []
    for i, n in sorted({(i, n) for i, n, *_ in rows}):
        pi = r"$\pi{=}$" + ("1" if SOURCES[i].endswith("pi1") else "0.5")
        label = [f"network, {pi}"] * 2 + [f"network, {pi}, SAR side study: {a}" for a in ("mean-field arm", "pairwise RFF")]
        cells = {s.truth_kernel: f"{SYN[s.fit_kernel]} {ok(s.fit_kernel == s.truth_kernel)} / {ok(s.fit_noise == t)}"
                 for j, k, _, s, t in reversed(rows) if (j, k) == (i, n)}
        body.append(line((label + [f"dyadic, $n{{=}}{size.get(n)}$, {pi}"] * 2)[i],
                         *[cells.get(k, "---").replace("RZB", "SAR") for k in ("degroot", "sbc", "rzb")]))
    head = r"setting & true DeGroot & true SBC & true SAR \\ & \multicolumn{3}{c}{selected kernel / noise recovered} \\"
    return tex("lccc", [head], body, small=(r"\small", 5))
def si_synthetic_discrimination_replicated():   # hit rate, and in brackets the one-SE rule's undecided share
    h = pd.read_csv(FD / "synthetic_discrimination_hitrates.csv").set_index(["axis", "truth_level", "rule", "n_train_markets"])
    sizes, body, k = sorted(h.index.unique("n_train_markets")), [], h.index.unique("n_train_markets").size
    name = lambda x: (SYN | {"rzb": "SAR", "gaussian": "Gaussian", "laplace": "Laplace", "nogate": "no gate", "nodrift": "no drift",
                             "degroot_vs_rff": "DeGroot vs RFF"}).get(x, x.replace("_", " "))
    axes = ("kernel degroot bc sbc rzb|kernel_class linear nonlinear|degroot_vs_rff degroot rff|noise gaussian laplace|"
            "retention nogate gate|drift nodrift drift|selection mean_field pairwise")
    for axis, *levels in map(str.split, axes.split("|")):
        for j, lv in enumerate(levels):
            hit = [h.loc[(axis, lv, r, n)] for r in ("one_se", "argmax") for n in sizes]
            body.append(line("" if j else name(axis), name(lv),
                             *[f"{x.hit:.2f}" + f" [{x.undecided:.2f}]" * bool(x.undecided > 0) for x in hit]))
    head = [r" & & \multicolumn{%d}{c}{one-SE rule} & \multicolumn{%d}{c}{argmax} \\" % (k, k),
            r"\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}" % (2 + k, 3 + k, 2 + 2 * k),
            line("axis", "true level", *[f"$n{{=}}{n}$" for n in sizes] * 2)]
    return tex("ll" + "c" * 2 * k, head, body, small=(r"\small", 4))
def before_after(selected):   # TV of the RFF gain at rung 2 and the top rung, or only where the ladder picks RFF
    tv = lambda v: "0" if v == 0 else f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
    def row(d):
        ks = [(rungs(d)[i], LADDER[d]["rungs"][i]["kernel"] if selected else RFF) for i in (2, max(rungs(d)))]
        v = [None if k == "null" else 0.0 if k == DG else size_curvature(r)[1] for r, k in ks]
        top = max([float(tv(x)) for x in v if x is not None], default=0)   # compared as printed: a tie bolds both
        change = 0 if selected else 100 * (v[1] / v[0] - 1)
        return line(m(d), *["null" if x is None else "0 (DG)" if x == 0 else rf"\textbf{{{tv(x)}}}" if float(tv(x)) == top
                            else tv(x) for x in v], *[rf"${change:+.0f}\%$" if round(change) else r"$0\%$"][:not selected])
    head = r"panel & before (kernel only) & after (full model)" + r" & change" * (not selected) + r" \\"
    return tex("lcc" if selected else "lccr", [head], grouped(NL, 4 - selected, row))
si_nl_before_after, si_tv_rff_before_after = lambda: before_after(True), lambda: before_after(False)
def si_size_curvature_by_rung():
    cells = lambda r: ["", ""] if r is None else ["=", "="] if r["same_as_prev"] else map(sig2, size_curvature(r))
    row = lambda d: line(m(d), *[c for n in (2, 3, 4, 5)
                                 for c in cells(rungs(d).get(n) if n < 5 or NL[d]["has_representation_axis"] else None)])
    head = [line("", *[rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in ("kernel only", "+ heavy tail", "+ retention", "+ selection")]),
            "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)), "panel" + r" & $S$ & $C$" * 4 + r" \\"]
    return tex("l" + "cc" * 4, head, grouped(NL, 9, row))
def si_selection_support():   # pairwise weight past the mean-field support, what the mean cancels, S and C on that range
    body, back = [], lambda rs, r: back(rs, rs[r["rung"] - 1]) if r["same_as_prev"] else r
    for d in [d for d in ORDER if {PW, MF} <= set(NL.get(d, {}).get("grid", {})) and f"pair_ev__{d}" in Z.files]:
        ev, g, a, u = (Z[f"{k}__{d}"] for k in ("pair_ev", "pair_gap", "pair_a", "gap"))
        end, deg = max(NL[d]["grid"][MF]), np.bincount(ev, minlength=u.size)
        k = (keep := (deg > 1) & (np.abs(u) > 1e-6))[ev]
        w = (a / np.bincount(ev, a, u.size)[ev] * Z.get(f"gamma__{d}", np.ones(u.size))[ev])[k]   # move weight split by the attention prior
        opp, tot = (np.bincount(ev, np.abs(g) * x, u.size) for x in (np.sign(g) == -np.sign(u[ev]), 1))
        sc = [size_curvature(r, np.asarray(NL[d]["grid"][r["representation"]]) <= end + 1e-12)
              for r in (back(rungs(d), rungs(d)[n]) for n in (4, 5))]
        body.append(line(m(d), f"{np.median(deg[keep]):.0f}", f"{w[np.abs(g[k]) > end].sum() / w.sum():.2f}",
                         f"{np.median(opp[keep & (tot > 0)] / tot[keep & (tot > 0)]):.2f}",
                         f"{np.quantile(np.abs(g[k]), .99) / np.quantile(np.abs(u[keep]), .99):.1f}",
                         *[rf"{sig2(sc[0][j])} $\to$ {sig2(sc[1][j])}" for j in (0, 1)]))
    return tex("lrccccc", [r"panel & degree & past support & opposite side & reach & $S$ & $C$ \\"], body)
def si_retention_median():   # Laplace median at constant pi, the fitted rung-3 gain, the data, and the pi(d) that fits
    D, edges, body = np.array([.1, .2, .3, .45, .6]), [.05, .15, .25, .375, .5, 1.01], []
    cdf = lambda x, mu, b: np.where(x < mu, .5 * np.exp((x - mu) / b), 1 - .5 * np.exp(-(x - mu) / b))
    root = lambda h, lo, hi: brentq(h, lo, hi) if h(lo) * h(hi) < 0 else np.nan
    for d in ["adams_p02", "adams_p05", "adams_p08"]:
        f = named(best_model(d, PW, grain=S, kernels=LADDER_KERNELS))
        mix = lambda x, q, r: q * cdf(x, f["lmbda"] * r, f["sigma"]) + (1 - q) * cdf(x, f["mu_null"], f["b_null"]) - .5
        G = np.mean([q["G_grid"] for q in rungs(d)[3]["folds"] if q.get("G_grid")], 0)
        fitted, x, y = np.interp(D, NL[d]["grid"][PW], G), np.abs(Z["gap__" + d]), Z["delta__" + d] * np.sign(Z["gap__" + d])
        rows = {"median at constant $\\pi$": [root(lambda v: mix(v, f["pi"], r), -2, 2) / r for r in D], "fitted, no retention": fitted,
                "data median $\\delta/d$": [np.median(y[k] / x[k]) for k in ((x > a) & (x <= b) for a, b in zip(edges, edges[1:]))],
                "influence share needed": [root(lambda q: mix(g * r, q, r), 1e-4, 1 - 1e-4) for g, r in zip(fitted, D)]}
        body.append((rf"{m(d)} ($\hat\pi={f['pi']:.2f}$, $\hat\lambda={f['lmbda']:.2f}$)",
                     [line("", name, *["" if v != v else f"{v:.2f}" for v in vals]) for name, vals in rows.items()]))
    return tex("llccccc", [line("", "", *[rf"$|d|={r:g}$" for r in D])], sections(body, 7, sep=r"\addlinespace", label=str))
def si_retention_stay_slope():   # near-stays (|delta| < 0.01) regressed on |d| above each floor: slope, HC1 s.e.
    def row(d):
        x, dl = np.abs(Z["gap__" + d]), Z["delta__" + d]
        cells = [m(d), f"{np.mean(np.abs(dl[x > .05]) < .01):.2f}"]
        for floor in (.05, .2):
            stay = (np.abs(dl[x > floor]) < .01).astype(float)
            fit = sm.OLS(stay, sm.add_constant(x[x > floor])).fit(cov_type="HC1") if len(stay) >= 50 and stay.std() else None
            cells += [f"{len(stay):,}", rf"${fit.params[1]:+.2f}\pm{fit.bse[1]:.2f}$" if fit else ""]
        return line(*cells)
    head = [r" & & \multicolumn{2}{c}{$|d|>0.05$} & \multicolumn{2}{c}{$|d|>0.2$} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"panel & near-stay share & events & slope $\pm$ s.e. & events & slope $\pm$ s.e. \\"]
    return tex("lcrcrc", head, grouped({k.split("__")[1] for k in Z.files if k.startswith("delta__")}, 6, row))

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or sorted(n for n in globals() if n.startswith("si_")):
        (OUT / f"{name}.tex").write_text(globals()[name]())
        print("wrote", (OUT / f"{name}.tex").relative_to(ROOT))
