"""Figures for the synthetic experiments: python scripts/figures/synthetic.py [name ...] (no names: every figure)."""
import json, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from kernel_inference import model as M, style as S
from kernel_inference.selection import SCORE, load_scores

FROZEN, DATA = S.ROOT / "data" / "figure_data_frozen", S.ROOT / "figure_data"
CLS = {"sigmoidal_bounded_confidence": M.SigmoidalBoundedConfidence, "bounded_confidence": M.BoundedConfidence,
       "rzb": M.RZB, "simplified_degroot": M.DeGroot}
PAR, R, K, PHI0, PHI1, DOT = list(CLS), r"$|d_e|$", r"$k(|d_e|)$", r"$\phi_0$", r"$\phi_1$", (0, (1, 2))
DIST, ERR = f"opinion distance  {R}", lambda s: rf"$\mathcal{{E}}_{{\mathrm{{{s}}}}}$"
TRUTH_2E = dict(kernel="sigmoidal_bounded_confidence", phi=[8, .3], lmbda=.5, noise="laplace_gate", rep="pairwise")  # sim 118711

gain = lambda kernel, phi, r: CLS[kernel](list(phi)).gain(torch.as_tensor(r, dtype=torch.float64)).detach().numpy() * np.ones_like(r)
band = lambda a, x, m, sd, c, alpha, **kw: (a.plot(x, m, color=c, **kw), a.fill_between(x, m - sd, m + sd, color=c, alpha=alpha))
line, NAME = lambda label, **kw: plt.Line2D([], [], label=label, **kw), lambda noise: S.NOISE[noise][0].replace(" +", "")
styled = lambda *rc: lambda f: lambda: mpl.rc_context()(lambda: (S.setup(*rc), f()))()  # the figure's rcParams, restored after
DARK, BODY, FOCUS, FRAMED = "#222018", "#3A3830", "#D97706", dict(fontsize=7, frameon=True, edgecolor="none", framealpha=1)

def simulate(kernel, phi, T=500, lmbda=0.1, sigma=0.01, seed=20260902):  # synchronous: each node moves towards a random neighbour
    nbrs = [list(n) for _, n in sorted(nx.convert_node_labels_to_integers(nx.les_miserables_graph()).adj.items())]
    x = np.tile((rng := np.random.default_rng(seed)).random(len(nbrs)), (T, 1))
    for t in range(T - 1):
        d = x[t, [rng.choice(n) for n in nbrs]] - x[t]
        x[t + 1] = x[t] + lmbda * gain(kernel, phi, np.abs(d)) * d + sigma * rng.standard_normal(len(nbrs))
    return x

@styled(S.LEGACY)
def fig_kernel_atlas():  # kernel, its equation, swept phi, the one drawn bold and simulated, inset y range
    E, D = r"k_\phi(|d_e|) = ", r"|d_e|^2"
    atlas = [("simplified_degroot", "k(|d_e|) = 1", [(1.,)], 0, (-.08, 1.12)), ("sigmoidal_bounded_confidence",
              E + rf"\frac{{1}}{{1 + \exp\!\left(\phi_1({D} - \phi_0)\right)}}", [(20., .1), (50., .1), (100., .1)], 2, (-.38, 1.3)),
             ("bounded_confidence", E + r"\mathbf{1}_{\{|d_e| < \phi\}}", [(.1,), (.2,), (.4,)], 1, (-.38, 1.4)),
             ("rzb", E + rf"\left(1 - \frac{{{D}}}{{\phi^2}}\right)\exp\!\left(-\frac{{{D}}}{{2\phi^2}}\right)", [(.25,), (.4,), (.6,)], 1,
              (-.85, 3.25))]
    (fig, ax), r = plt.subplots(2, 2, figsize=(9.6, 8.2), layout="constrained"), np.linspace(0, 2, 600)
    for a, (k, eq, sweep, hi, ylim) in zip(ax.flat, atlas):
        for i, phi in enumerate(sweep):
            a.plot(r, gain(k, phi, r), color=S.INFLUENCE, lw=1 + (i == hi), alpha=1 if i == hi else .28, zorder=2 + (i == hi))
        a.set(xlim=(0, 2), ylim=(-.55 if k == "rzb" else -.45, 1.15), xlabel="opinion distance " + R)
        a.set_ylabel(r"interaction gain $k_\phi(|d_e|)$"), a.axhline(0, color=S.HAIR, lw=.6, zorder=1), a.tick_params(labelsize=9)
        a.set_title(f"{S.KERNELS[k][0]}\n${eq}$", color=DARK, fontsize=17, pad=15)
        ins = a.inset_axes([.55, .56, .39, .3], xticks=[], yticks=[], xlim=(0, 499), ylim=ylim)
        ins.plot(simulate(k, sweep[hi]), color=S.INFLUENCE, alpha=.14, lw=.55), ins.spines[:].set_linewidth(.45)
    S.save(fig, "fig_kernel_atlas")

@styled(S.LEGACY)
def fig_sbc_diagnostic():
    d, r = json.loads((FROZEN / "sbc_summary_v2.json").read_text()), np.linspace(0, 1, 300)
    (p0, p1), ko, dyn = (np.array(d["axes"][k]) for k in ("phi0", "phi1")), d["kernel_overlay"], d["representative_dynamics"]
    fig, cmap = plt.figure(figsize=(14, 6.5)), mpl.colors.LinearSegmentedColormap.from_list("", ["#FAF8F2", S.INK])
    gs = fig.add_gridspec(2, 4, hspace=.55, wspace=.4, height_ratios=[1, 1.05])
    ax = [fig.add_subplot(g) for g in [gs[0, k] for k in range(4)] + [gs[1, :].subgridspec(1, 4, wspace=.45)[0, k] for k in range(4)]]
    for j, (a, vals, lab) in enumerate([(ax[0], [1, 3, 8, 20, 60], PHI1), (ax[1], [.04, .09, .16, .25, .36], PHI0)]):
        for i, v in enumerate(vals):  # the model orders SBC's phi (phi_1, phi_0)
            a.plot(r, gain(PAR[0], [(v, .09), (20, v)][j], r), color=cmap(.25 + .75 * i / 4), lw=1.8)
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vals[0], vals[-1]), cmap), ax=a, fraction=.046, pad=.04, shrink=.85)
        cb.ax.set_title(lab, fontsize=8, color=BODY, pad=4), cb.ax.tick_params(labelsize=7, colors=BODY), cb.outline.set_edgecolor("#E0E0E0")
        a.set(xlim=(0, 1), ylim=(-.05, 1.15), xlabel=R, ylabel=K)
    ax[0].axvline(.3, color=BODY, lw=.8, ls=":", alpha=.5)
    ax[0].text(.32, .06, r"$|d_e|^* = \sqrt{\phi_0}$", fontsize=7, color=BODY, va="bottom")
    tw, hist, bins = ax[2].twinx(), ko["pair_distance_hist_true"], np.array(ko["pair_distance_bins"])  # pair distances behind the curves
    tw.bar(c := (bins[1:] + bins[:-1]) / 2, hist, width=.85 * (c[1] - c[0]), color="#D1D5DB", alpha=.65, zorder=1)
    tw.set(ylim=(0, (top := 1.5 * tw.get_ylim()[1])), yticks=np.linspace(0, top, 4)), tw.spines["right"].set(visible=True, color="#CCCCCC")
    tw.set_ylabel("Pair density", fontsize=7, color="#999999"), tw.tick_params(axis="y", labelsize=6, colors="#999999")
    ax[2].set_zorder(tw.get_zorder() + 1), ax[2].patch.set_visible(False)
    for k, p, ls, c, L, fm in (("gt", ko["sim_phi"], "--", DARK, "GT", ".1f"), ("inferred", ko["inferred_phi"], "-", S.INK, "Inf", ".2f")):
        ax[2].plot(ko["radius"], ko[f"{k}_gain"], ls, color=c, lw=1.8, zorder=4, label=rf"{L}: {PHI0}={p[0]:.2f}, {PHI1}={p[1]:{fm}}")
    ax[2].fill_between(ko["radius"], ko["gt_gain"], ko["inferred_gain"], alpha=.12, color=S.INK, zorder=3), ax[2].legend(loc=1, **FRAMED)
    for k, c, lab in (("true", "#8B7355", "GT"), ("inferred", "#6E8899", "Inferred")):
        m, sd = (np.array(dyn[f"node_trace_{k}_{s}"]) for s in ("mean", "std"))
        band(ax[3], np.arange(len(m)), m, sd, c, .25, lw=1.8, zorder=3, label=lab)
    ax[2].set(ylim=(.3, .72), xlabel=R, ylabel=K), ax[2].set_title("Kernel Shape", pad=4), ax[3].set_title("Node trace", pad=4)
    ax[3].set(xlabel="Time step", ylabel=r"$x_i(t)$"), ax[3].margins(0, .08), ax[3].legend(**FRAMED)  # y: the data plus 8%
    err, cell = np.zeros((len(p0), len(p1))), lambda phi: (np.abs(p0 - phi[0]).argmin(), np.abs(p1 - phi[1]).argmin())
    for t, f in ((np.array(c["sim_phi"]), np.array(c["mean_phi_inferred"])) for c in d["cell_summaries"] if any(c["sim_phi"])):
        err[cell(t)] = np.linalg.norm(f - t) / np.linalg.norm(t)  # relative error of the recovered phi
    (i0, i1), h, dx, dy = cell(ko["sim_phi"]), d["heatmaps"], p0[1] - p0[0], p1[1] - p1[0]  # outline the cell shown in (c)
    for a, v, lab in zip(ax[4:], [h["tv_distance"], h["tw_tv_distance"], h["kl_divergence"], err], [*map(ERR, ("TV", "TW", "KL")),
                                                                                                 r"$\mathcal{E}_{\phi}$"]):
        im = a.imshow(np.array(v).T, aspect="auto", origin="lower", cmap="Greys", extent=[p0[0], p0[-1], p1[0], p1[-1]])
        fig.colorbar(im, ax=a, shrink=.8, pad=.02).set_label(lab, fontsize=8), a.set_title(lab, fontsize=8, pad=4)
        a.add_patch(plt.Rectangle((p0[i0] - dx / 2, p1[i1] - dy / 2), dx, dy, fill=False, ec="#111111", lw=2))
        a.set(xlabel=PHI0, ylabel=PHI1, box_aspect=1)
    [a.text(-.12 if i < 2 else -.18, 1.08, f"({chr(97 + i)})", transform=a.transAxes, fontsize=9, fontweight="bold", va="top")
     for i, a in enumerate(ax)]
    S.save(fig, "fig_sbc_diagnostic")

@styled(S.LEGACY)
def fig_tv_sweep():
    fig, ax = plt.subplots(3, 3, figsize=(11, 4.4), gridspec_kw=dict(hspace=.2, wspace=.25, top=.84, bottom=.1))
    cols, ink = [("simplified_degroot", 518, 532), ("bounded_confidence", 516, 537), ("rzb", 520, 744)], np.array(mpl.colors.to_rgb(S.INK))
    for j, (k, uid, fid) in enumerate(cols):
        u, f = (pd.read_csv(FROZEN / f"sweep_metrics_{k}_{i}.csv") for i in (uid, fid))
        for i, (a, (m, lab)) in enumerate(zip(ax[:, j], [("tv", "TV"), ("tw_tv", "TW"), ("kld", "KL")])):  # rows: TV, TW, KL
            band(a, u.phi_true, u[f"{m}_mean"], u[f"{m}_std"], S.INK, .1, lw=1.6), a.tick_params(labelsize=9, length=3)
            a.plot(f.phi_true, f[f"{m}_mean"], color=S.INK, lw=1.6, ls="--")
            a.set_ylabel(ERR(lab), fontsize=11) if j == 0 else a.set_yticklabels([]), i < 2 and a.set_xticklabels([])
        ax[2, j].set_xlabel(r"$\phi$", fontsize=11), ax[0, j].set_title(S.KERNELS[k][0], fontsize=12, fontweight="bold", color=S.INK, pad=6)
        ins = ax[0, j].inset_axes([.55 if k == "rzb" else .03, .42, .42, .52])
        r = np.linspace(0, 3 * u.phi_true.median() if k == "rzb" else 1.05, 300)
        for t, phi in zip(np.linspace(0, 1, len(u)), u.phi_true):  # the true gain at every swept phi, light to dark
            ins.plot(r, gain(k, [phi], r), color=(1 - t) * (.3 * ink + .7) + t * ink, lw=.7)
        k != "bounded_confidence" and ins.axhline(0, color="#CCCCCC", lw=.4, zorder=0), ins.set_ylabel(K, fontsize=6, labelpad=2)
        ins.spines[:].set_linewidth(.5), ins.tick_params(labelsize=5, length=1.5, width=.4, pad=1), ins.set_xlabel(R, fontsize=6, labelpad=1)
    inits = [line("Uniform init", color="#555555", lw=1.8), line("Fiedler init", color="#555555", lw=1.8, ls="--"), line(" ", ls="none")]
    fig.legend(handles=[h for k, _, _ in cols for h in (line(S.KERNELS[k][0], color=S.INK, lw=2, marker=S.KERNELS[k][3], ms=4),
               inits.pop(0))], ncol=3, loc="lower center", bbox_to_anchor=(.5, .97), fontsize=8, handlelength=1.6, columnspacing=1.4)
    S.save(fig, "fig_tv_sweep")

@styled(S.LEGACY)
def synthetic_2x2():
    rows, (fig, ax) = json.loads((FROZEN / "synthetic_2x2_results.json").read_text())["rows"], plt.subplots(2, 2, figsize=(11.5, 9.5))
    (ax := ax.ravel())[0].plot([0, 1.05], [0, 1.05], "--", color="#555555", lw=1, label="$y=x$")
    for a, (m, lab, title) in zip(ax, [("pi_hat", r"recovered $\hat\pi$", "(a) π recovery"), ("tv", "TV distance",
                                       "(b) total variation"), ("tw_tv", "TW-TV distance", "(c) trajectory-weighted TV"),
                                       ("kld", "KL divergence", "(d) time-series KLD")]):
        for n in ["laplace_gate", "gaussian_gate"] if m == "pi_hat" else [*S.NOISE][:2] + ["laplace_gate", "gaussian_gate"]:
            v = pd.DataFrame([(r["gate_pi"], r["corners"].get(n, {}).get(m)) for r in rows]).astype(float).dropna().groupby(0)[1]
            a.errorbar(v.mean().index, v.mean(), v.sem().fillna(0), color=S.NOISE[n][1], marker=S.NOISE[n][2], ms=4, lw=1.5, capsize=3,
                       label=S.NOISE[n][0])  # mean +/- 1 SE over seeds
        a.set(xlabel=r"generating $\pi$ (mover fraction)", ylabel=lab), a.set_title(title, pad=4)
    ax[0].set(xlim=(0, 1.05), ylim=(0, 1.05)), ax[0].legend(fontsize=8), ax[1].legend(fontsize=8), fig.tight_layout()
    S.save(fig, "synthetic_2x2")

@styled(S.PAPER, {"savefig.pad_inches": .08})
def fig_rff_recovery():
    d, (fig, ax) = json.loads((FROZEN / "rff_vs_mlp.json").read_text()), plt.subplots(2, 2, figsize=(S.WIDTH, 4.6),
                                                                                       gridspec_kw={"wspace": .22, "hspace": .42})
    for a, p, r in zip(ax.flat, d["panels"], [np.array(d["r_rff"])] * 4):
        g, lk0, k = np.array(p["rff_gain"]), p["rff_lambda_gain0"], p["kernel"]  # lambda * k(0): only the product is identified
        a.spines[["left", "bottom"]].set_color(S.GREY), a.tick_params(labelsize=8, colors=S.INK), a.set_axisbelow(True)
        a.grid(axis="y", color=S.HAIR, alpha=.7, lw=.8), a.axhline(0, color=S.GREY, lw=.8, ls="--", zorder=0)
        a.plot(r, gain(k, p["phi"], r), color=S.GREY, lw=1.6, ls="--", zorder=6, label="Ground truth")
        a.plot(r, lk0 / g[0] * g, color=S.INK, lw=2, zorder=5, label=r"RFF ($R$=64, $\gamma$=10)")
        a.text(.97, .6 if k == "simplified_degroot" else .97, rf"$\widehat{{\lambda k}}(0)={lk0:.2f}$", transform=a.transAxes,
               ha="right", va="top", fontsize=7.5, color=S.INK, linespacing=1.4)
        a.set_title(S.KERNELS[k][0] + " (Linear)" * (k == "simplified_degroot"), fontsize=9.5, fontweight="bold", color=S.INK, pad=4)
    [a.set_ylabel(r"$\lambda k(r)$", color=S.INK) for a in ax[:, 0]], [a.set_xlabel(r"$r = |x_j - x_i|$", color=S.INK) for a in ax[1]]
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=8.5)
    S.save(fig, "fig_rff_recovery")

@styled(S.PAPER, S.PNAS)
def fig_synthetic_2e_recovery():
    s, t, kernels = load_scores(), TRUTH_2E, PAR + ["legendre", "random_fourier"]
    reps, r = [("pairwise", "pairwise"), ("neighbour_mean_field", "mean-field")], np.linspace(0, 1, 400)
    s = s[s.dataset == "synthetic_2e_network"].assign(full=lambda f: f.per_fold.map(len) >= 5)  # partial fits are never drawn or selected
    fits = {rep: s[(s.representation == rep) & (s.noise_variant == t["noise"]) & s.full].set_index("kernel").reindex(PAR) for rep, _ in reps}
    fig, ax = plt.subplot_mosaic("abc;ddd", figsize=(S.WIDTH, 4.6), layout="constrained", height_ratios=(1, 1.15))
    for key, (rep, lab) in zip("ab", reps):
        ax[key].plot(r, t["lmbda"] * gain(t["kernel"], t["phi"], r), color=S.INK, lw=1.6, zorder=6)
        for k, f in fits[rep].dropna(subset=["lmbda"]).iterrows():
            ax[key].plot(r, f.lmbda * gain(k, f.phi, r), color=S.INK, ls=S.KERNELS[k][2], lw=2.3 if k == t["kernel"] else 1.3, zorder=4)
        ax[key].set(xlim=(0, 1), xlabel=DIST, ylabel=r"influence  $\lambda\,k$", title=f"({key}) fitted {lab}")
    for j, (p, mk) in enumerate([("lmbda", "o"), ("pi", "s")]):
        for i, (rep, lab) in enumerate(reps):  # pairwise outlined, mean field faded
            ax["c"].scatter(np.arange(4) + (j - .5) * .36 + (i - .5) * .14, fits[rep][p], marker=mk, s=18, c=S.INK, ec=S.INK, zorder=6,
                            lw=.5 - .5 * i, alpha=1 - .55 * i)
    ax["b"].sharey(ax["a"]), ax["c"].axhline(t["lmbda"], color=S.INK, lw=1.6, zorder=2), ax["c"].grid(axis="y", color=S.HAIR, lw=.6)
    ax["c"].set(xticks=range(4), xticklabels=[S.KERNELS[k][1] for k in PAR], ylim=(0, 1.05), title=rf"(c) truth $\lambda=\pi={t['lmbda']}$, "
                f"noise {NAME(t['noise'])}", ylabel=r"$\hat\lambda$ (circles), $\hat\pi$ (squares)")  # the truth sets lambda = pi
    cols, g, a = [(rep, n) for rep, _ in reps for n in S.NOISE], s.set_index(["kernel", "representation", "noise_variant"]), ax["d"]
    mll, full = (np.array([[g[col].get((k, *c), na) for c in cols] for k in kernels]) for col, na in ((SCORE, np.nan), ("full", False)))
    gap = np.where(full, mll, np.nan) - np.nanmax(np.where(full, mll, np.nan))
    a.imshow(gap, cmap="Greys_r", vmin=-.6, vmax=0, aspect="auto")
    for (i, j), v in np.ndenumerate(mll):
        a.text(j, i, f"{v:.3f}" + ("" if full[i, j] else "\n(partial)"), ha="center", va="center", fontsize=5.4 if full[i, j] else 4.6,
               color="white" if gap[i, j] < -.3 else S.INK, fontweight="bold" if gap[i, j] == 0 else "normal")
        full[i, j] or a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////", ec=S.GREY, lw=0))
    ti, tj, (bi, bj) = kernels.index(t["kernel"]), cols.index((t["rep"], t["noise"])), np.argwhere(gap == 0)[0]
    a.add_patch(plt.Rectangle((tj - .5, ti - .5), 1, 1, fill=False, ec=S.INK, lw=1.4, ls="--"))
    a.add_patch(plt.Rectangle((bj - .45, bi - .45), .9, .9, fill=False, ec=FOCUS, lw=1.2))
    a.set(yticks=range(6), yticklabels=[S.KERNELS[k][1] for k in kernels], xticks=range(8))
    a.set_xticklabels([NAME(n) for _, n in cols], fontsize=5.2), a.axvline(3.5, color="white", lw=3), a.tick_params(length=0)
    [a.text(4 * k + 1.5, -.62, f"fitted {lab}", ha="center", va="bottom", fontweight="bold") for k, (_, lab) in enumerate(reps)]
    a.set_title("(d) held-out MLL per transition (5-fold node CV)", pad=16), a.spines[:].set_visible(False)
    fig.legend(handles=[line(f"truth ({S.KERNELS[t['kernel']][1]})", color=S.INK, lw=1.6)] + [line(S.KERNELS[k][1], color=S.INK, lw=2.3,
               ls=S.KERNELS[k][2]) for k in PAR] + [line(f"{lab} fit (c)", marker="o", ls="none", mfc=S.GREY, mec=[S.INK, "none"][i], mew=.5,
               alpha=1 - .55 * i) for i, (_, lab) in enumerate(reps)] + [plt.Rectangle((0, 0), 1, 1, fill=False, ec=ec, ls=ls, label=lab)
               for ec, ls, lab in ((S.INK, "--", "true model (d)"), (FOCUS, "-", "selected (d)"))], loc="outside lower center", ncol=5)
    S.save(fig, "fig_synthetic_2e_recovery")

@styled(S.PAPER, {"font.size": 8.28, "axes.titlesize": 8.625, "axes.labelsize": 9.775, "xtick.labelsize": 8.05, "ytick.labelsize": 8.05,
                  "legend.fontsize": 8.51, "legend.handlelength": 2.4, "legend.columnspacing": 2.2, "legend.labelspacing": .55,
                  "legend.handletextpad": .7})  # PNAS type sizes times 1.15: the figure spans the page
def fig_synthetic_rzb_2x2():
    curve = lambda f, noise, k="rff": f[(f.fit_kernel == k) & (f.fit_noise == noise)].sort_values("r")
    dy = pd.read_csv(DATA / "synthetic_dyadic_noise_misspec_shapes_gsel.csv").query("truth_kernel == 'rzb' and n_events == 4000")
    G, (pw, mf) = "laplace_gate_drift", (pd.read_csv(DATA / f"synthetic_noise_misspec_shapes_rzb_{n}.csv") for n in ("pairwise_rff", "nmf"))
    fig, ax = plt.subplots(1, 3, figsize=(S.WIDTH, 2.1), layout="constrained", width_ratios=(1, 1, 1.12))
    for a, L, (on, off, lin), c, size, stitle in (  # inset: size (RMS) for retention, curvature (total variation) for selection
            (ax[0], "a", (curve(dy, "laplace_gate"), curve(dy, "laplace_nogate"), curve(dy, "laplace_nogate", "degroot")), S.RETENTION,
             lambda k: np.sqrt(np.mean(np.square(k))), "size $S$"),
            (ax[1], "b", (curve(pw, G), curve(mf, G), curve(mf, G, "degroot")), S.SELECTION,
             lambda k: np.abs(np.diff(k)).sum(), "curvature $C$")):
        a.plot(on.r, on.k_true, color=S.INK, lw=1.6, zorder=6), a.plot(off.r, off.k_fit, color=S.GREY, lw=1.3, zorder=4)
        a.plot(lin.r, lin.k_fit, color=S.GREY, ls=DOT, lw=1.3, zorder=4), a.plot(on.r, on.k_fit, color=c, lw=2.3, zorder=6)
        ins, v = a.inset_axes((.59, .56, .37, .38)), [size(k.to_numpy()) for k in (on.k_true, off.k_fit, on.k_fit)]
        ins.bar(range(3), v, color=[S.INK, S.GREY, c], width=.68, ec="white", lw=.5), ins.set_ylim(0, 1.18 * max(v))
        ins.set_xticks(range(3), ["true", "off", "on"], fontsize=6.555), ins.tick_params(axis="y", labelsize=6.67, length=2)
        ins.set_title(stitle, fontsize=8.28, loc="right", pad=2), ins.set_in_layout(False), ins.spines[:].set_linewidth(.6)
        a.set(xlabel=DIST, ylabel="influence gain"), a.set_title(f"({L})", fontweight="bold")
    cap, a = json.loads((DATA / "synthetic_rzb_capability.json").read_text()), ax[2]
    for name, x0, head, c in (("retention", 0, "retention", S.RETENTION), ("selection", 2.3, "neighbor\nselection", S.SELECTION)):
        sc = np.array([cap[name]["base"], cap[name]["totals"]])  # rows without / with the mechanism, columns linear / RFF
        for k, ls in ((1, "-"), (0, DOT)):  # filled: the higher-scoring kernel
            a.plot([x0, x0 + 1], sc[:, k], color=S.GREY, ls=ls, lw=1.3, zorder=4)
            [a.plot(x, row[k], "o", ms=6.2, mew=1.6, mec=cc, mfc=cc if row[k] >= row[1 - k] else "white", zorder=6)
             for x, row, cc in zip((x0, x0 + 1), sc, (S.GREY, c))]
        a.text(x0 + .5, .99, head, transform=a.get_xaxis_transform(), ha="center", va="top", color=c, fontweight="bold", linespacing=.95)
    a.axvline(1.65, color=S.HAIR, lw=.7), a.set_xticks([0, 1, 2.3, 3.3], ["without", "with"] * 2, fontsize=7.82)
    a.set(xlim=(-.6, 3.75), ylim=(0, 1.32 * max(max(v["base"] + v["totals"]) for v in (cap["retention"], cap["selection"]))))
    a.set_ylabel("held-out MLL / trans.\n(nats above null)", fontsize=8.28), a.tick_params(axis="y", labelsize=7.59)
    a.set_title("(c)", fontweight="bold", fontsize=9.775), a.grid(axis="y", color=S.HAIR, lw=.6)
    fig.legend(handles=[line("SAR truth", color=S.INK, lw=1.6), line("linear baseline", color=S.GREY, ls=DOT, lw=1.3), line("RFF baseline",
               color=S.GREY, lw=1.3), line("retention", color=S.RETENTION, lw=2.3), line("neighbor selection", color=S.SELECTION, lw=2.3),
               line("filled = selected (c)", color=S.INK, marker="o", ms=5.5, ls="none")], loc="outside lower center", ncol=3)
    S.save(fig, "fig_synthetic_rzb_2x2")

def fig_synthetic_mechanism_direction():  # cell, mechanism colour, title, truth, its linear-truth control
    cells = [("ret_indep", S.RETENTION, "retention, constant stay share", "SAR", None),
             ("ret_dep", S.RETENTION, "retention, stay share rises with distance", "DeGroot", "retention_dd_control"),
             ("sel_indep", S.SELECTION, "selection, uniform pick", "SAR", None),
             ("sel_dep", S.SELECTION, "selection, homophilous pick", "DeGroot", "selection_hom_control")]
    sh, su = (pd.read_csv(DATA / f"synthetic_mechanism_direction_{n}.csv").query("kernel == 'rff'") for n in ("shapes", "summary"))
    means = lambda c: su[su.cell == c].groupby("fit")[["S", "C_over_S", "tilt", "gap_rff_minus_degroot"]].mean()  # over seeds
    fig, ax, row = *S.grid(4, ncols=2, height=2.5), lambda *v: v
    for a, L, (cell, col, title, truth, control) in zip(ax, "abcd", cells):  # curves of the first seed
        off, on = (sh[(sh.cell == cell) & (sh.fit == f) & (sh.seed == sh.seed.min())].sort_values("r") for f in ("without", "with"))
        a.plot(on.r, on.k_true, color=S.INK, lw=1.6), a.plot(off.r, off.k_fit, color=S.GREY, lw=1.3)
        a.plot(on.r, on.k_fit, color=col, lw=2.3), a.axhline(0, color=S.HAIR, lw=.7)
        t = su[su.cell == cell].iloc[0]
        rows = [(row("", "S", "C/S", "tilt", "gap"), S.INK),
                (row("truth", f"{t.truth_S:.2f}", f"{t.truth_C_over_S:.1f}", f"{t.truth_tilt:+.2f}", ""), S.INK)]
        rows += [(row(pre + nm, *(f"{v:{fm}}" for v, fm in zip(means(cc).loc[f], (".2f", ".1f", "+.2f", "+.3f")))), cl)
                 for cc, pre in [(cell, ""), (control, "floor ")][:2 if control else 1]
                 for nm, f, cl in (("off", "without", S.GREY), ("on", "with", col))]
        a.set_ylim(a.get_ylim()[0], a.get_ylim()[1] + (.5 + .09 * len(rows)) * np.ptp(a.get_ylim()))
        for i, (cells_, cl) in enumerate(rows):   # columns at fixed x, header in bold, the floor rows faded
            for x, txt in zip((.30, .55, .69, .84, .99), cells_):
                a.text(x, .975 - .075 * i, txt, transform=a.transAxes, ha="left" if x == .30 else "right", va="top",
                       fontsize=5.6, color=cl, fontweight="bold" if i == 0 else "normal",
                       alpha=.8 if cells_[0].startswith("floor") else 1)
        if off.r.max() < on.r.max() - 1e-6:
            a.annotate("mean-field fit stops at\nits own 99th pct\nof |mean gap|", (off.r.max(), off.k_fit.iloc[-1]), xytext=(5, 3),
                       textcoords="offset points", fontsize=5.4, color=S.GREY)
        a.text(.02, .03, f"{truth} truth", transform=a.transAxes, fontsize=6)
        a.set(xlabel=DIST, ylabel="influence gain"), a.set_title(f"({L}) {title}", fontweight="bold")
    fig.suptitle(f"RFF fit without (grey) and with (colour) the mechanism;\ntable: mean over {su.seed.nunique()} seeds, floor = linear "
                 "truth with a distance-independent mechanism; gap = held-out RFF minus linear, nats/transition", fontsize=6.2)
    S.save(fig, "fig_synthetic_mechanism_direction")

if __name__ == "__main__":
    for name in sys.argv[1:] or [n for n in list(globals()) if n.startswith(("fig_", "synthetic_"))]:
        globals()[name]()
