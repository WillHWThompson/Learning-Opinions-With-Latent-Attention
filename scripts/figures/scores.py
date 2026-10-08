"""Figures from the held-out scores: kernel ladders, the capability ladder, the kernel by rung.
python scripts/figures/scores.py [name ...]   (no names: every figure; "export" writes capability_ladder.json)"""
import json, sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from kernel_inference import selection as S
from kernel_inference.style import (GREY, GROUPS, HAIR, INK, KERNELS, LABEL, LADDER as L, LEGACY, MONO, NAVY, NETWORK, NOISE,
                                    ORDER, PAGE, PAPER, PNAS, RETENTION, SELECTION, TAIL, grid, pack, pictogram, save, setup)

FD, SC, DG, RFF, G = S.SCORES.parent, S.SCORE, "simplified_degroot", "random_fourier", S.load_scores(starved=True)
LAD = [d for d in ORDER if d != "takacs_disliking_interior"]  # a subset of takacs_disliking
RUNG, RT = {2: GREY, 3: TAIL, 4: RETENTION, 5: SELECTION}, {2: "baseline", 3: "+ tail", 4: "+ ret.", 5: "+ sel."}
Line, DOT = lambda **kw: mpl.lines.Line2D([], [], **kw), (0, (1, 2))

def step(hi, lo):  # 2 x the held-out gain per transition, fold-size weighted
    lo, n = {f["fold"]: f[SC] for f in lo.per_fold}, sum(f["n_transitions"] for f in hi.per_fold)
    return {"value": sum(f["n_transitions"] * 2 * (f[SC] - lo[f["fold"]]) for f in hi.per_fold) / n}

def ladder_panel(ds):
    net, rows = {S.MF, S.PW} <= set(G[G.dataset == ds].representation), S.ladder(G, ds)
    b = list(rows[S.GATE_RUNG].folds_b_null)
    return {"dataset": ds, "has_representation_axis": net, "rungs": [
        {"label": lab, "kernel": r.kernel, "noise_variant": r.noise_variant, "representation": r.representation,
         "mll": float(r[SC])} for lab, r in zip(S.RUNG_LABELS, rows)],
        "steps": {k: None if k == "selection" and not net else step(rows[h], rows[lo]) for k, h, lo in S.STEPS},
        "gate": {"degenerate": bool(b) and all(x <= 1.001e-5 for x in b)}, "total": step(rows[S.TOP_RUNG], rows[0])}

def best(ds, kw, kernels):  # best score over the arms a rung allows, null and Legendre included
    rows = [S.best_model(ds, a, grain=G, variants=kw["variants"], kernels=kernels, drop=()) for a in kw["arms"]]
    return max((r[SC] for r in rows if r is not None), default=None)

def climb(ds, kernels):  # null, DeGroot at rung 1, {kernel: scores at rungs 2..top}, top
    spec, top = [kw for _, kw in S.rung_spec(ds in NETWORK)], 5 if ds in NETWORK else 4
    ys = {k: [best(ds, spec[r], [k]) for r in range(2, top + 1)] for k in kernels}
    lines = {k: np.array(y) for k, y in ys.items() if None not in y}
    return best(ds, spec[0], ["null"]), best(ds, spec[1], [DG]), lines, top

def curve(ds, i):  # gain of the kernel the ladder selects at rung i over its mean |k|, on [0, r99]; None for the null
    (kern, nv, rep), nl = (LADDER[ds]["rungs"][i][k] for k in ("kernel", "noise_variant", "representation")), NL[ds]
    r = np.array(nl["grid"][rep])
    if kern != RFF:
        return (r, np.ones_like(r)) if kern == DG else None
    gam = S.best_model(ds, rep, grain=G, variants=[nv], kernels=[RFF]).rff_gamma
    rec = [q for q in nl["rungs"] if (q["kernel"], q["noise_variant"], q["representation"]) == (RFF, nv, rep)
           and abs(q["rff_gamma"] - gam) < 1e-6]
    grids = [f["G_grid"] for f in rec[0]["folds"] if "G_grid" in f] if rec else []
    return (r, (k := np.mean(grids, 0)) / np.mean(np.abs(k))) if grids else None

def rows(groups, head, off, shift=0):  # top-down [(y, heading, None) or (y, None, dataset)], and the bottom
    y, out = 0, []
    for h, members in groups:
        y, ds = y - head, [d for d, _ in members if d in LAD]
        out, y = out + [(y + off, h, None)] + [(y - i + shift, None, d) for i, d in enumerate(ds, 1)], y - len(ds)
    return out, y

def noise(arm, title):
    dss, P = [d for d in ORDER if d in set(G[G.representation == arm].dataset)], [*list(KERNELS)[:2], DG, "rzb"]
    n, (fig, axes) = -(-len(dss) // 4), grid(len(dss), 4, 2.15, width=10.2, rc=(LEGACY, PNAS | {"legend.fontsize": 6}), layout=None)
    for ax, ds in zip(axes, dss):
        sc = {v: {k: S.best_model(ds, arm, grain=G, variants=[v], kernels=[k])[SC] for k in P + [RFF]} for v in NOISE}
        order = sorted(P, key=lambda k: (np.mean([sorted(s, key=s.get).index(k) for s in sc.values()]),  # mean rank (RFF ranked too)
                                         np.mean([s[k] for s in sc.values()]), k))
        for (lab, c, m, _), s in ((NOISE[v], s) for v, s in sc.items()):
            ax.plot(range(4), [s[k] for k in order], "-" + m, color=c, ms=3, lw=1.3, alpha=.95, zorder=4, label=lab)
            ax.plot([3, 4.6], [s[order[-1]], s[RFF]], ":", color=c, lw=1, alpha=.65, zorder=3)
            ax.plot(4.6, s[RFF], m, color=c, ms=3.6, zorder=5)
        ax.axvline(3.8, color="0.7", lw=.7, ls="--", zorder=1), ax.grid(axis="y", alpha=.25, lw=.5)
        ax.set_xticks([0, 1, 2, 3, 4.6], [KERNELS[k][1] for k in order] + ["RFF"], rotation=30, ha="right")
        ax.set_title(LABEL[ds], loc="left", family="monospace"), ax.set_ylabel("held-out MLL / transition" * (ax in axes[::4]))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(.5, -.06 / n))
    fig.suptitle(f"Kernel ordering within each noise family ({title})", y=.999, fontsize=9, fontweight="bold")
    return fig.tight_layout(rect=(0, .14 / n, 1, 1 - .075 / n)) or fig

def ladder(ncols):  # the manuscript's layout: margins measured from the text they hold, pictograms sized in points
    W, colors, segs, (_, fig) = 4 * ncols + 1.0, [GREY, TAIL, RETENTION, SELECTION], {}, (setup(PAPER), plt.figure())
    for d in LAD:
        v = [(LADDER[d]["steps"][k] or {"value": 0})["value"] / LADDER[d]["total"]["value"] for k, *_ in S.STEPS]
        segs[d] = [(c, w, curve(d, i)) for c, w, i in zip(colors, [v[0] + v[1], *v[2:]], range(2, 6)) if w > 0]
    smax = max(.25, *(np.ptp(c[1]) for s in segs.values() for *_, c in s if c is not None))  # one scale for all glyphs
    wd = lambda s, fs, **kw: ((t := fig.text(0, 0, s, fontsize=fs, linespacing=1, **kw)).get_window_extent(
        fig.canvas.get_renderer()).width / fig.dpi, t.remove())[0]   # inches
    lay = [rows(c, L["head"], L["head"] / 2, .5) for c in ([GROUPS] if ncols == 1 else [GROUPS[:2], GROUPS[2:]])]
    tick = lambda d: LABEL[d] + " †" * LADDER[d]["gate"]["degenerate"]   # a dagger: the retention gate degenerated
    tot = {d: f"{np.exp(LADDER[d]['total']['value'] / 2):.2f}×" for d in LAD}
    xl = -(3.5 / 72 + max(wd(tick(d), L["label"], family=MONO) for d in LAD))   # left edge of the dataset names
    num, hd = max(wd(t, L["total"]) for t in tot.values()), max(wd(s, L["total"] - .6) for s in ("likelihood", "× null"))
    cw, xlab, gh = W / ncols, "", L["bar"] * L["row"] * 72
    plot, gw = cw - .06 + xl - 4 / 72 - max(num, (num + hd) / 2) - .04, 1.45 * gh
    for w in "share of the log likelihood ratio over the random-walk null".split():
        xlab = w if not xlab else xlab + (" " if wd(xlab.split("\n")[-1] + " " + w, L["axis"]) <= plot else "\n") + w
    ylo, xin, xpt = min(b for _, b in lay) - .2, .32 + L["axis"] * 1.25 / 72 * (xlab.count("\n") + 1), 1 / (plot * 72)
    fig.set_size_inches(W, H := L["row"] * (.1 - ylo) + L["legend"] + xin + .38)
    for i, (rws, bot) in enumerate(lay):
        ax = fig.add_axes([(i * cw + .06 - xl) / W, (b := (L["legend"] + xin) / H), plot / W, 1 - .38 / H - b])
        tr, hs = ax.get_yaxis_transform(), [(y, h) for y, h, _ in rws if h]
        for k, (y, h) in enumerate(hs):  # heading, a rule above all but the first, the left spine one piece per group
            ax.text(xl / plot, y, h, transform=tr, va="center", fontsize=L["heading"], style="italic", color=GREY, clip_on=False)
            ax.plot([xl / plot, 1], [y + .4] * 2, transform=tr, color=HAIR, lw=.6, zorder=2, clip_on=False, visible=k > 0)
            ax.plot([0, 0], [y - .4, ([z + .4 for z, _ in hs[1:]] + [bot - .2])[k]], transform=tr, color=INK, lw=.8,
                    solid_capstyle="butt", zorder=2.5, clip_on=False)
        for y, d in [(y, d) for y, h, d in rws if d]:
            want = [(sum(s[1] for s in segs[d][:j]) + w / 2, w, cu, c) for j, (c, w, cu) in enumerate(segs[d])]
            [ax.barh(y, w, left=cx - w / 2, height=L["bar"], color=c, edgecolor=PAGE, lw=.9, zorder=4) for cx, w, _, c in want]
            want = [g for g in want if g[2] is not None]  # a glyph wider than its segment sits on a tile of its colour
            for (cx, w, (r, k), c), x in zip(want, pack([g[0] for g in want], (gw + 1.6) * xpt, (gw / 2 + 1) * xpt, 1 - gw / 2 * xpt)):
                pictogram(ax, x, y, r, k, np.sqrt(max(np.ptp(k), 1e-9) * smax), gh - .9, gw, .8 * gh / (gh - .9), lw=.95, color=PAGE,
                          fill=c if abs(x - cx) + gw / 2 * xpt > w / 2 else None, zorder=5, ls=DOT if np.ptp(k) == 0 else "solid")
            ax.text(1 + 4 * xpt, y, tot[d], va="center", fontsize=L["total"], color=GREY, zorder=5, clip_on=False)
        ax.set_yticks(*zip(*[(y, tick(d)) for y, h, d in rws if d]), fontsize=L["label"], family=MONO), ax.spines.left.set_visible(False)
        ax.tick_params(axis="y", length=0), ax.set(xlim=(0, 1), ylim=(ylo, .1)), ax.set_xlabel(xlab, fontsize=L["axis"], linespacing=1.25)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"], fontsize=L["tick"])
        ax.text(1 + (4 + num * 36) * xpt, .1, "likelihood\n× null", ha="center", va="bottom", multialignment="center",
                fontsize=L["total"] - .6, color=GREY, linespacing=1, clip_on=False)
    lax, lw, b = fig.add_axes([.012, .01, .976, (L["legend"] - .06) / H], xlim=(0, 1), ylim=(0, 1)), .976 * W, np.linspace(0, 1, 40)
    ww, sh, g = lambda s: wd(s, L["key"]) / lw, .62 * L["key"] / 72 / (L["legend"] - .06), .8 * gw / 72 / lw
    key = [(.018, c, t, INK) for c, t in zip(colors, ["Influence + kernel shape", "Heavy tail", "Latent retention",
           "Latent neighbor selection"])] + [(g, (b, 1 - .8 * b ** 2, 1.6), r"fitted gain $k(r)$ at the step's model, $r\in[0,r_{99}]$",
                                              GREY), (g, (np.r_[0., 1], np.r_[1., 1], 1), "linear", GREY)]
    for items, y in ((key[:1], .45), (key[1:4], 1.5), (key[4:], 2.75)):  # legend rows, centred
        x, y = max(0, (1.03 - sum(u + .052 + ww(t) for u, _, t, _ in items)) / 2), 1 - (y + .5) / 4
        for u, c, t, tc in items:
            pictogram(lax, x + g / 2, y, *c, .8 * gh, .8 * gw, lw=.95, color=INK, ls=DOT if c[2] == 1 else "solid") if u == g else \
                lax.add_patch(mpl.patches.Rectangle((x, y - sh / 2), u, sh, facecolor=c, edgecolor=PAGE, lw=.8))
            lax.text(x + u + (.022 if u == g else .006), y, t, va="center", fontsize=L["key"], color=tc)
            x += u + .052 + ww(t)
    return lax.axis("off") and fig

def by_rung(mode):  # "all": every kernel minus DeGroot; "glyph"/"shapes": DeGroot and RFF minus the null
    dss, every = (ORDER, True) if mode == "all" else (LAD, False)
    fig, axes = grid(len(dss), 4, 1.45 if every else 1.95, rc=(PAPER, PNAS))
    for ax, ds in zip(axes, dss):
        null, base, lines, top = climb(ds, [DG, *list(KERNELS)[:2], "rzb", RFF, "legendre"] if every else [DG, RFF])
        xs, _ = np.arange(2, top + 1), ax.set_title(LABEL[ds], pad=15.3 if mode == "glyph" else 2)
        if every:  # main axes against DeGroot; the inset climbs from rung 1 against the null
            ins, _ = ax.inset_axes([.1, .62, .36, .34], facecolor=PAGE), ax.axhline(0, color=HAIR, lw=.7, zorder=0)
            ins.plot([1, 2], [base - null] * 2, color=GREY, lw=.5, zorder=0)
            for k, ys in lines.items():
                st = dict(color=INK if k in (DG, RFF) else GREY, ls=KERNELS[k][2], marker=KERNELS[k][3], mfc="none", mew=.6, zorder=4,
                          label=KERNELS[k][1])
                ax.plot(xs, ys - lines[DG], ms=2.6, lw=1.1, **st), ins.plot(range(1, top + 1), np.r_[base, ys] - null, ms=1.4, lw=.5, **st)
            ax.set_xticks(range(2, 6), ["+kernel", "+tail", "+ret.", "+sel."], rotation=45, ha="right")
            ax.set(xlim=(1.7, 5.3), ylim=(lambda lo, hi: (lo, hi + 1.15 * (hi - lo)))(*ax.get_ylim()))
            ins.set(xlim=(.7, 5.3), xticks=[]), ins.set_ylim(bottom=0), ins.yaxis.set_major_locator(plt.MaxNLocator(2))
            ins.tick_params(labelsize=3.9, length=1.5, pad=1), ins.set_ylabel("vs null", fontsize=3.9, labelpad=1)
            ins.spines[["left", "bottom"]].set_linewidth(.4)
            continue
        for k, ls, lw in ((DG, (0, (6, 4)), 1.3), (RFF, "solid", 1.6)):
            ax.plot(xs, lines[k] - null, color=INK, ls=ls, lw=lw, zorder=4, solid_capstyle="butt")
            for x, y in zip(xs, lines[k] - null):  # filled: the kernel the ladder selects at that rung, drawn larger and last
                win = LADDER[ds]["rungs"][x]["kernel"] == k
                ax.plot(x, y, "o", ms=3.6 * (1 + .35 * win), mew=1, mec=RUNG[x], mfc=RUNG[x] if win else PAGE, zorder=6 + win)
        ax.set_xticks(list(RT), list(RT.values()), rotation=45, ha="right")
        [t.set(color=RUNG[x], fontweight="bold" if x == top else "normal", alpha=float(x <= top)) for x, t in zip(RT, ax.get_xticklabels())]
        ax.set_xlim(1.6, 5.4), ax.yaxis.set_major_locator(plt.MaxNLocator(3)), ax.set_box_aspect(1)
        cs = {x: c for x in xs if (c := curve(ds, x)) is not None}
        if mode == "glyph":
            span = max(.25, *(np.ptp(k) for _, k in cs.values()))
            [pictogram(ax, x, 1.1, *c, span, 9, tr=ax.get_xaxis_transform(), box=RUNG[x], color=INK) for x, c in cs.items()]
            continue
        ins = ax.inset_axes([.56, .05, .42, .4])
        ins.axhline(1, color=HAIR, lw=1.1, zorder=2), ins.set(xticks=[], yticks=[]), ins.patch.set_alpha(.9)
        [ins.plot(*c, color=RUNG[x], lw=1.3, zorder=4 + x) for x, c in cs.items()], [s.set(color=GREY, lw=.5) for s in ins.spines.values()]
    fig.supylabel("held-out MLL / transition, minus " + ("DeGroot's at the same rung" if every else "the null's"), fontsize=7,
                  color=INK if every else GREY)
    hs = list({h.get_label(): h for a in axes for h in a.get_legend_handles_labels()[0]}.values()) if every else [
        Line(color=INK, lw=1.6, label="nonlinear (RFF)"), Line(color=INK, ls=(0, (6, 4)), lw=1.3, label="linear (DeGroot)"),
        Line(ls="none", marker="o", ms=3.6, mew=1, mec=INK, mfc=INK, label="filled = selected at that rung")] + [
        Line(color=c, lw=1.3, label=rf"$\hat{{k}}$ at {RT[x]}") for x, c in RUNG.items() if mode == "shapes"]
    return fig.legend(handles=hs, loc="outside upper center", ncol=6 if every else 3) and fig

def gaps():  # RFF minus DeGroot at the baseline rung and at the top rung
    return {d: (g[0], g[-1], t) for d in LAD for _, _, ln, t in [climb(d, [DG, RFF])] for g in [ln[RFF] - ln[DG]]}

AT = {"takacs_study1": (-4, 1, "right"), "house": (4, -5, "left"), "plos_counting": (-4, 0, "right"), "spinos": (4, 2, "left"),
      "kozitsin": (4, -6, "left"), "markets": (0, -8, "center"), "takacs_disliking": (3, -7, "left"), "senate": (-4, 3, "right"),
      "becker_hubless": (-4, -6, "right"), "cmv": (-4, -1, "right"), "takacs_control": (-5, -1, "right")}  # labels clear of the crowd

def scatter():  # the y axis stops above the second-highest point; one beyond it is an arrow at the top edge
    gp, (fig, (ax,)) = gaps(), grid(1, 1, 3.1, width=3.1, rc=(PAPER, PNAS))
    xs, ys = np.array([v[:2] for v in gp.values()]).T
    lo, hi, right = 1.25 * min(xs.min(), ys.min()), 1.35 * np.sort(ys)[-2], 1.4 * xs.max()
    pad, tail = .03 * (right - lo), hi - .07 * (hi - lo)
    ax.axhline(0, color=HAIR, lw=1.1, zorder=2), ax.axvline(0, color=HAIR, lw=1.1, zorder=2)
    ax.plot([lo, right], [lo, right], ":", color=GREY, lw=1.1, zorder=2)
    for x, y, ha, va, t in [(lo + pad, hi - pad, "left", "top", "RFF wins only\nwith mechanisms"), (lo + pad, -pad, "left", "top",
                            "DeGroot wins\nat both"), (right - pad, pad, "right", "bottom", "RFF wins\nat both"),
                            (right - pad, lo + pad, "right", "bottom", "DeGroot wins only\nwith mechanisms")]:
        ax.text(x, y, t, ha=ha, va=va, color=GREY, fontsize=6.5, linespacing=1.1)
    for d, (b, f, top) in gp.items():  # "null selected": neither kernel beats the null at the top rung
        (*off, ha), up = AT.get(d, (4, 1, "left")), f > hi
        name = LABEL[d] + " (null selected)" * (LADDER[d]["rungs"][top]["kernel"] == "null")
        ax.annotate("", (b, hi), (b, tail), annotation_clip=False, visible=up, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1,
                                                                                            mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.plot(b, f, "o", ms=4.2, mew=1, mec=INK, mfc=PAGE, zorder=6, visible=not up)
        ax.annotate(f"{name}, full {f:+.3f}" if up else name, (b, (hi + tail) / 2 if up else f), xytext=(-4, 0) if up else off,
                    textcoords="offset points", fontsize=5.5, color=INK, ha="right" if up else ha, va="center" if up else "baseline")
    ax.set(xlim=(lo, right), ylim=(lo, hi), aspect="equal", xlabel="RFF − DeGroot, baseline (nats / trans.)",
           ylabel="RFF − DeGroot, full model (nats / trans.)")
    ax.annotate("no change", (.82 * lo, .82 * lo), xytext=(3, -3), textcoords="offset points", rotation=45, fontsize=6,
                rotation_mode="anchor", ha="left", va="top", color=GREY, transform_rotates_text=True)
    return fig

def glyph_grid():
    gp, (rws, depth) = gaps(), rows(GROUPS, .9, .25)
    fig, (ag, ad) = grid(2, 2, .9 - .26 * depth, width=5, rc=(PAPER,), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    ad.axvline(0, color=HAIR, lw=1.1, zorder=2)
    [ag.text(1.45, y, h, va="center", color=GREY, fontsize=7.5, style="italic", clip_on=False) for y, h, _ in rws if h]
    for y, d in [(y, d) for y, h, d in rws if d]:
        (b, f, top), sty = gp[d], dict(ms=4, mew=1, zorder=6)
        cs = {x: c for x in range(2, top + 1) if (c := curve(d, x)) is not None}
        [pictogram(ag, x, y, *c, max(.25, *(np.ptp(k) for _, k in cs.values())), 14.6, lw=.85, box=RUNG[x], color=INK)
         for x, c in cs.items()]
        ad.annotate("", (f, y), (b, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1, shrinkA=2.5, shrinkB=3.5, mutation_scale=6))
        ad.plot(b, y, "o", mec=GREY, mfc=PAGE, **sty), ad.plot(f, y, "o", color=RUNG[top], **sty)
        if min(b, f) > 1e-3:  # a ratio of gaps means something only where both are positive
            ad.text(1, y, f"×{f / b:.1f}", transform=ad.get_yaxis_transform(), va="center", fontsize=7, color=INK, clip_on=False)
    ag.set(xlim=(1.45, 5.55), ylim=(depth - .6, -.2)), ag.set_xticks(list(RT), list(RT.values())), ag.xaxis.tick_top()
    ag.set_yticks(*zip(*[(y, LABEL[d]) for y, h, d in rws if d]), family=MONO, fontsize=7)
    ag.spines[:].set_visible(False), ag.tick_params(length=0), ad.tick_params(axis="y", length=0)
    ad.spines[["left", "top", "right"]].set_visible(False), ad.locator_params(axis="x", nbins=4)
    ad.set_xlabel("RFF minus DeGroot,\nheld-out MLL / transition (nats)")
    ad.legend(handles=[Line(ls="none", marker="o", mec=GREY, mfc=PAGE, label="baseline", **sty), Line(ls="none", marker="o",
              color=SELECTION, label="full model (top rung)", **sty)], loc="lower right", fontsize=6.5, handletextpad=.3)
    return fig

def before_after():  # total variation of the gain the ladder selects at rung 2 and at the top: 0 for DeGroot, None for null
    tv = {d: [None if (k := LADDER[d]["rungs"][i]["kernel"]) == "null" else 0.0 if k == DG else np.mean([np.abs(np.diff(
        f["G_grid"])).sum() for f in NL[d]["rungs"][i]["folds"] if "G_grid" in f]) for i in (2, 5)] for d in LAD}
    norm = mpl.colors.PowerNorm(.5, 0, max(x or 0 for v in tv.values() for x in v))
    cmap, (rws, depth) = mpl.colors.LinearSegmentedColormap.from_list("nl", [PAGE, NAVY]), rows(GROUPS, .22 / .19, .15)
    _, (fig, ax) = setup(PAPER), plt.subplots(figsize=(3, .19 * len(LAD) + .22 * len(GROUPS) + .45))
    [ax.text(-.5, y, h, va="center", fontsize=6.5, style="italic", color=GREY) for y, h, _ in rws if h]
    for y, d in [(y, d) for y, h, d in rws if d]:
        ax.text(-.58, y, LABEL[d], ha="right", va="center", fontsize=6.5, family=MONO)
        for j, x in enumerate(tv[d]):  # text in ink or white, whichever contrasts more with the cell
            ax.add_patch(mpl.patches.Rectangle((j - .5, y - .5), 1, 1, fc=(c := cmap(norm(x or 0))), ec=PAGE, lw=1))
            lum = np.dot([.2126, .7152, .0722], [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in c[:3]])
            ax.text(j, y, "null" if x is None else f"{x:.{3 if x < .1 else 2}f}" if x else "0  DG", ha="center", va="center",
                    fontsize=6.8, color=(INK if lum > .36 else PAGE) if x else GREY,
                    fontweight="bold" if x and x == max(v or 0 for v in tv[d]) else "normal")
    ax.set(xlim=(-.5, 1.5), ylim=(depth - .5, 0), yticks=[]), ax.xaxis.tick_top(), ax.tick_params(length=0)
    ax.set_xticks([0, 1], ["before\n(kernel only)", "after\n(full model)"], fontsize=6.8), ax.spines[:].set_visible(False)
    return fig

FIGS = {"fig_kernel_ladder_by_noise_pairwise": lambda: noise(S.PW, "pairwise"),
        "fig_kernel_ladder_by_noise_meanfield": lambda: noise(S.MF, "mean field"),
        "fig_capability_ladder": lambda: ladder(1), "fig_capability_ladder_wide": lambda: ladder(2),
        "fig_si_kernel_by_rung": lambda: by_rung("all"), "fig_si_kernel_by_rung_dg_rff": lambda: by_rung("glyph"),
        "fig_si_kernel_by_rung_dg_rff_shapes": lambda: by_rung("shapes"), "fig_si_kernel_glyph_grid": glyph_grid,
        "fig_si_nonlinearity_gap_scatter": scatter, "fig_nl_before_after": before_after}
if __name__ == "__main__":
    if "export" in sys.argv:
        (FD / "capability_ladder.json").write_text(json.dumps({"panels": [ladder_panel(d) for d in S.PANELS]}, indent=2) + "\n")
    LADDER, NL = ({p["dataset"]: p for p in json.loads((FD / f).read_text())["panels"]}
                  for f in ("capability_ladder.json", "nonlinearity_by_rung_rff.json"))
    for name in [n for n in sys.argv[1:] if n != "export"] or ([] if "export" in sys.argv else FIGS):
        save(FIGS[name](), name)
