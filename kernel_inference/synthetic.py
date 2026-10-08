"""Synthetic data from known kernels, and the fits the experiments in scripts/synthetic.py share. On a network a
node moves with probability pi (delta = lmbda k(x_i, x_J) + eps, J a uniform neighbour, or k(x_i, xbar_i) under
the mean-field DGP), else delta = mu_null + eps_null. Dyadic events have uniform partners. Mover-arm SNR is 1."""
import math
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .config import Config
from .model import (EM, RZB, BoundedConfidence, DeGroot, Events, SigmoidalBoundedConfidence, event_index,
                    make_events, make_kernel)

F64 = torch.float64
SEED, LMBDA, B_NULL, SIGMA0, BANDWIDTHS = 20260825, 0.5, 0.35, 0.05, (0.5, 1, 2, 4)   # RFF gamma: s / median(r)
TRUTH = {"sbc": lambda: SigmoidalBoundedConfidence([30.0, 0.09]), "rzb": lambda: RZB([0.30]),
         "degroot": lambda: DeGroot([0.15]), "bc": lambda: BoundedConfidence([0.30])}
ARMS = {f"{n}_{g}": (n, g != "nogate", g == "gate_drift")       # (family, gate, null-arm drift)
        for g in ("nogate", "gate", "gate_drift") for n in ("gaussian", "laplace")}
CANONICAL, FIT_KERNELS = list(ARMS)[:4], ["bc", "sbc", "degroot", "rzb", "rff", "legendre"]
KIND = {"bc": "bounded_confidence", "sbc": "sigmoidal_bounded_confidence", "degroot": "simplified_degroot",
        "rzb": "rzb", "rff": "random_fourier", "legendre": "legendre", "null": "null"}
# bc's gain is an indicator, fitted by golden-section search; rff and legendre solve their weights in closed
# form, where lambda is not identified and is pinned at 1. The restarts bracket the truth.
METHOD = {"bc": "bisection", "rff": "closed_form", "legendre": "closed_form"}
RESTARTS = {"sbc": [(8.0, 0.05), (40.0, 0.20)], "rzb": [(0.15,), (0.50,)], "degroot": [(0.05,), (0.40,)],
            "bc": [(0.15,), (0.50,)]}

def les_miserables(G=nx.les_miserables_graph()):
    return torch.as_tensor(nx.to_numpy_array(G, nodelist=sorted(G), weight=None), dtype=F64)

def draw(shape, family, scale, g):
    if family == "gaussian":
        return torch.randn(shape, generator=g, dtype=F64) * scale
    u = torch.rand(shape, generator=g, dtype=F64) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())

@torch.no_grad()
def trajectories(truth, A, dgp, family, scale, n_markets, n_steps, seed, pi, mu_null, b_null):
    """x (M, N, n_steps + 1), and mu, eps, mover gate per transition. No gate draw at pi = 1 (the original stream)."""
    k, g, (M, N), deg = TRUTH[truth](), torch.Generator().manual_seed(seed), (n_markets, A.shape[0]), A.sum(1, keepdim=True)
    P, x, out = A * torch.where(deg > 0, 1.0 / deg, 0.0), [torch.rand((M, N), generator=g, dtype=F64)], []
    for _ in range(n_steps):
        if dgp == "pairwise":
            j = torch.multinomial(P.expand(M, N, N).reshape(M * N, N), 1, True, generator=g).reshape(M, N)
            mu = LMBDA * k(x[-1], torch.gather(x[-1], 1, j)).to(F64)
        else:
            mu = LMBDA * k(x[-1], x[-1] @ P.T).to(F64)
        eps, z = draw((M, N), family, scale, g), torch.ones((M, N), dtype=torch.bool)
        if pi < 1:
            z = torch.rand((M, N), generator=g, dtype=F64) < pi
            mu, eps = torch.where(z, mu, torch.full_like(mu, mu_null)), torch.where(z, eps, draw((M, N), family, b_null, g))
        out.append((mu, eps, z))
        x.append(x[-1] + mu + eps)
    return (torch.stack(x, -1), *(torch.stack(v, -1) for v in zip(*out)))

@torch.no_grad()
def dyadic_events(truth, family, scale, n_subjects, n_exposures, seed, pi):   # each (subjects, exposures)
    k, g, S, out = TRUTH[truth](), torch.Generator().manual_seed(seed), n_subjects, []
    x_i = torch.rand((S,), generator=g, dtype=F64)
    for _ in range(n_exposures):
        x_j = torch.rand((S,), generator=g, dtype=F64)
        mu, eps = LMBDA * k(x_i, x_j).to(F64), draw((S,), family, scale, g)
        z = torch.rand((S,), generator=g, dtype=F64) < pi
        eps = torch.where(z, eps, draw((S,), family, B_NULL * scale, g))
        out.append((x_i, x_j, torch.where(z, mu, torch.zeros_like(mu)), eps, z))
    return [torch.stack(v, 1) for v in zip(*out)]

def solve_scale(run, scale):
    for _ in range(3):   # the null arm's scale is a multiple of the noise scale, hence the iteration
        mu, eps, gate = (v.reshape(-1) for v in run(scale)[-3:])
        scale = float(mu[gate].std()) / (float(eps[gate].std()) / scale)
    return scale

def gain_curve(kernel, lmbda, grid):
    with torch.no_grad():
        return float(lmbda) * kernel.gain(torch.as_tensor(grid, dtype=F64)).cpu().reshape(-1).to(F64).numpy()

def dataset(truth, r, **kw):
    """RFF bandwidth 1 / median(r), Legendre range max(r), and a grid to the 99th percentile of r, with r's density."""
    d = SimpleNamespace(**kw, q50=float(np.quantile(r, 0.5)), rmax=float(r.max()),
                        grid=np.linspace(0.0, float(np.quantile(r, 0.99)), 201))
    mid, top = 0.5 * (d.grid[:-1] + d.grid[1:]), d.grid[-1] + 0.5 * (d.grid[-1] - d.grid[-2])
    counts = np.histogram(r, bins=np.concatenate([[0.0], mid, [top]]))[0]
    d.gamma, d.weights, d.k_true = 1.0 / d.q50, counts / counts.sum(), gain_curve(TRUTH[truth](), LMBDA, d.grid)
    return d

def network(truth, A, *, pi, family="laplace", drift=0.5, n_train=24, n_test=40, n_steps=10, seed=SEED,
            dgp="pairwise", device="cpu"):   # trajectories split by market into train and test
    run = lambda s: trajectories(truth, A, dgp, family, s, n_train + n_test, n_steps, seed, pi, drift * s, B_NULL * s)
    scale = solve_scale(run, 0.02)
    x, Ad, mf = run(scale)[0], A.to(device), {"pairwise": False, "neighbour_mean_field": True}
    xs, train, test = x[..., :-1], x[:n_train].contiguous().to(device), x[n_train:].contiguous().to(device)
    ones = torch.ones(A.shape, dtype=torch.bool, device=device)
    events = lambda x, rep: make_events(x, Ad, x == x, ones, x == x, False, mf[rep])
    ev_test = {rep: events(test, rep) for rep in mf}
    m, i, _ = event_index(ev_test["pairwise"], test, test == test, False).cpu().numpy().T
    return dataset(truth, (xs[:, None] - xs[:, :, None]).abs()[:, A != 0].reshape(-1).numpy(), seed=seed, scale=scale,
                   train={rep: events(train, rep) for rep in mf}, test=ev_test, cluster=m * A.shape[0] + i,
                   meta=dict(dataset=f"{truth}_{family}_pi{pi:g}", truth_kernel=truth, truth_lmbda=LMBDA,
                             truth_noise=family, truth_pi=pi, truth_sigma=scale, truth_mu_null=drift * scale))

def dyadic(truth, *, pi, n_events, family="laplace", n_exposures=2, seed=SEED):   # split by subject into halves
    S = n_events // n_exposures
    scale = solve_scale(lambda s: dyadic_events(truth, family, s, S, n_exposures, seed, pi), 0.05)
    x_i, x_j, mu, eps, _ = dyadic_events(truth, family, scale, S, n_exposures, seed, pi)
    one, test = torch.ones(n_events, dtype=torch.bool), np.zeros(S, dtype=bool)
    ev = Events((mu + eps).reshape(-1), x_i.reshape(-1), x_j.reshape(-1, 1), one[:, None].to(F64), one, one[:, None])
    test[np.random.default_rng(seed).permutation(S)[:round(0.5 * S)]] = True
    rows = np.repeat(test, n_exposures)
    return dataset(truth, (x_j - x_i).abs().reshape(-1).numpy(), seed=seed, scale=scale,
                   train={"pairwise": ev[torch.as_tensor(~rows)]}, test={"pairwise": ev[torch.as_tensor(rows)]},
                   cluster=np.repeat(np.arange(S), n_exposures)[rows],
                   meta=dict(dataset=f"{truth}_{family}_pi{pi:g}_n{n_events}", truth_kernel=truth, truth_lmbda=LMBDA,
                             truth_noise=family, truth_pi=pi, truth_sigma=scale, truth_mu_null=0.0, n_events=n_events))

def fit(kernel, train, test, arm, *, gamma, rmax, seed, features=64, device="cpu"):
    """Fit every restart, keep the best by train log-likelihood: (em, train ll, restart, held-out ll per event)."""
    (family, gate, drift), best = ARMS[arm], None
    free, positive = kernel not in ("rff", "legendre", "null"), kernel in ("sbc", "rzb")
    for i, phi in enumerate(RESTARTS.get(kernel, [()])):
        c = Config(fit="synthetic", panel="none", kernel_func=KIND[kernel], phi_init=[list(phi)], noise_model=family,
                   infer_pi=gate, init_pi=0.5 if gate else 1.0, decoupled_scale=gate, init_b_null=SIGMA0 if gate else None,
                   infer_mu_null=drift, lmbda_init=0.5 if free else 1.0, infer_lmbda=free, lmbda_min=-math.inf,
                   kernel_positive_phi_transform=positive, legendre_max_distance=rmax, rff_gamma=gamma,
                   rff_num_features=features, optimization_method=METHOD.get(kernel, "gradient_descent"),
                   optimizer_class="adam", gd_rate=0.05, max_gd_iter=60, max_em_iter=40, ll_stopping_criterion=1e-7,
                   bisection_max_iter=200, bisection_max_distance=rmax)
        phi = [v + math.log(-math.expm1(-v)) for v in phi] if positive else list(phi)   # inverse softplus
        em = EM(make_kernel(c, phi, seed + i).to(device), c, SIGMA0)
        em.fit(train)
        with torch.no_grad():
            ll = float(em.log_likelihood(train).mean())
            if best is None or ll > best[1]:
                best = (em, ll, i, em.log_likelihood(test).cpu().numpy())
    return best

def fit_data(d, kernel, arm, rep="pairwise", scales=(1,), features=64, device="cpu"):   # best held-out s
    return max((fit(kernel, d.train[rep], d.test[rep], arm, gamma=s * d.gamma, rmax=d.rmax, seed=d.seed, features=features,
                    device=device) for s in scales), key=lambda f: f[3].mean())

def fit_row(d, kernel, arm, rep="pairwise", fitted=None):
    """A results row, and the curve lambda g(r) of the fit and of the truth on the shape grid."""
    em, train, restart, test = fitted or fit_data(d, kernel, arm, rep)
    k_fit, gate = gain_curve(em.k, em.lmbda, d.grid), em.c.infer_pi
    print(d.meta["dataset"], rep, arm, kernel, f"test_mll={test.mean():.5f}", flush=True)
    return dict(d.meta, fit_noise=arm, fit_kernel=kernel, representation=rep, restart=restart, train_mll_per_event=train,
                test_mll_per_event=float(test.mean()), n_test_scored=test.size, lmbda_hat=em.lmbda, sigma_hat=em.sigma,
                pi_hat=em.pi if gate else np.nan, b_null_hat=em.b_null if gate else np.nan, mu_null_hat=em.mu_null,
                shape_nrmse_weighted=float(np.sqrt((d.weights * (k_fit - d.k_true) ** 2).sum()))
                / float(np.sqrt((d.weights * d.k_true ** 2).sum())), r_q50=d.q50), pd.DataFrame(dict(
        {k: v for k, v in d.meta.items() if k in ("dataset", "truth_kernel", "n_events")}, fit_kernel=kernel,
        fit_noise=arm, representation=rep, r=d.grid, r_weight=d.weights, k_true=d.k_true, k_fit=k_fit))

def paired_contrast(a, b, cluster):   # mean of a - b over the same events, its cluster-robust SE, events, clusters
    diff, c = np.asarray(a) - np.asarray(b), np.unique(cluster, return_inverse=True)[1].ravel()
    g, totals = c.max() + 1, np.bincount(c, weights=diff - diff.mean())
    return float(diff.mean()), float(np.sqrt(g / (g - 1) * (totals ** 2).sum() / diff.size ** 2)), diff.size, int(g)
