"""The opinion-dynamics model and its EM fit.

Node i's opinion change delta is a mixture: with probability pi it was influenced by one
neighbour j, drawn with attention weight a_ij, and then delta = lmbda * k(x_i, x_j) + noise;
otherwise it is a "null" move, noise around mu_null with scale b_null (the mover gate).
k(x_i, x_j) = g(|x_j - x_i|) (x_j - x_i), with the gain g given by a kernel. EM alternates the
responsibilities of each (event, neighbour) pair and of the gate with updates of the kernel,
lmbda, the noise scale and the gate.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TINY = torch.finfo(torch.float64).tiny


# ---------------------------------------------------------------- kernels

class Kernel(nn.Module):
    def forward(self, x_i, x_j):
        d = x_j - x_i
        return self.gain(d.abs()) * d

    def penalty(self):
        return None


class Parametric(Kernel):
    """A gain with parameters phi; with `positive`, phi = softplus(raw) + floor."""

    def __init__(self, phi, positive=False, floor=0.0):
        super().__init__()
        self.phi = nn.Parameter(torch.Tensor(phi))
        self.positive = positive
        self.register_buffer("floor", torch.as_tensor(floor, dtype=torch.float32).reshape(-1))

    def effective_phi(self):
        if not self.positive:
            return self.phi
        floor = self.floor.to(self.phi)
        return F.softplus(self.phi) + (floor.expand_as(self.phi) if floor.numel() == 1 else floor)


class BoundedConfidence(Parametric):
    def gain(self, r):
        return (r < self.effective_phi()).to(r.dtype)


class SigmoidalBoundedConfidence(Parametric):
    def gain(self, r):
        phi = self.effective_phi()
        return torch.sigmoid(-(phi[0] * (r.square() - phi[1])))


class DeGroot(Parametric):
    def __init__(self, phi, positive=False, floor=0.0, unit_gain=False):
        super().__init__(phi, positive, floor)
        self.unit_gain = unit_gain
        self.phi.requires_grad_(not unit_gain)

    def gain(self, r):
        return torch.ones_like(r) if self.unit_gain else self.effective_phi()[0].expand_as(r)


class RZB(Parametric):
    def gain(self, r):
        phi = self.effective_phi()
        return (1 - (r / phi).square()) * torch.exp(-r.square() / (2 * phi.square()))


class Null(Kernel):
    def gain(self, r):
        return torch.zeros_like(r)


class Basis(Kernel):
    """g(r) = c0 + sum_d w_d psi_d(r), fitted in closed form by ridge regression.
    With unit gain at zero, psi_d(r) = basis_d(r) - basis_d(0) and c0 = 1, so g(0) = 1."""

    def basis(self, r):
        out = self.raw_basis(r)
        return out - self.basis0.to(out) if self.unit_gain else out

    def gain(self, r):
        w = self.weights.to(r)
        step = max(1, int(2 ** 28 // max(r.numel() // max(r.shape[0], 1) * w.numel() * r.element_size(), 1)))
        parts = [float(self.unit_gain) + (self.basis(r[i:i + step]) * w).sum(-1)
                 for i in range(0, max(r.shape[0], 1), step)]
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def closed_form(self, ev, B, lmbda, jitter=1e-8):
        #TODO: Make sure we include IRWLS for Laplace fits
        lam, D = torch.as_tensor(lmbda, dtype=torch.float64), self.weights.numel()
        step = max(1, 20_000_000 // max(ev.cand_x.shape[1] * D, 1))
        gram = torch.zeros(D, D, dtype=torch.float64, device=ev.delta.device)
        rhs = torch.zeros(D, dtype=torch.float64, device=ev.delta.device)
        for s in range(0, ev.E, step):
            e = slice(s, s + step)
            real = ev.cand_valid[e] & ev.valid[e][:, None]
            diff = (ev.cand_x[e].double() - ev.x_self[e].double()[:, None])[real]
            design = (lam * diff[:, None]) * self.basis(diff.abs()).double()
            w = B[e].double()[real]
            y = ev.delta[e].double()[:, None].expand_as(real)[real]
            if self.unit_gain:
                y = y - lam * diff
            gram += design.T @ (w[:, None] * design)
            rhs += design.T @ (w * y)
        new = torch.linalg.solve(self.regularize(gram, jitter), rhs)
        with torch.no_grad():
            self.weights.copy_(new.to(self.weights))


class Legendre(Basis):
    def __init__(self, num_basis=9, max_distance=1.0, degree_penalty=0.02, unit_gain=False):
        super().__init__()
        self.n, self.max_distance, self.degree_penalty = num_basis, max_distance, degree_penalty
        rng = torch.Generator().manual_seed(0)
        self.weights = nn.Parameter(torch.randn(num_basis, generator=rng) * 0.01)
        self.unit_gain = False
        self.register_buffer("basis0", self.raw_basis(torch.zeros(1, dtype=torch.float64))[0].float())
        self.unit_gain = unit_gain

    def raw_basis(self, r):
        t = (2.0 * r.clamp(0.0, self.max_distance) / self.max_distance - 1.0).reshape(-1)
        p = torch.zeros((t.shape[0], self.n), dtype=t.dtype, device=t.device)
        p[:, 0] = 1.0
        p[:, 1] = t
        for n in range(2, self.n):
            p[:, n] = ((2 * n - 1) * t * p[:, n - 1] - (n - 1) * p[:, n - 2]) / n
        return p.reshape(*r.shape, self.n)

    def regularize(self, gram, jitter):
        degrees = torch.arange(self.n, dtype=gram.dtype, device=gram.device)
        return (gram + self.degree_penalty * torch.diag(degrees.square())
                + jitter * torch.eye(self.n, dtype=gram.dtype, device=gram.device))

    def penalty(self):
        degrees = torch.arange(self.n, dtype=self.weights.dtype, device=self.weights.device)
        return self.weights.sum() * 0.0 + self.degree_penalty * torch.mean(degrees.square() * self.weights.square())


class RandomFourier(Basis):
    """Random features sqrt(2/D) cos(w r + b), w ~ N(0, gamma^2), b ~ U(0, 2 pi), drawn from `seed`."""

    def __init__(self, num_features=256, gamma=2.0, seed=42, unit_gain=False, ridge=1e-3):
        super().__init__()
        rng = torch.Generator().manual_seed(int(seed))
        self.register_buffer("W", torch.randn(num_features, generator=rng) * float(gamma))
        self.register_buffer("b", torch.rand(num_features, generator=rng) * 2.0 * math.pi)
        self.register_buffer("scale", torch.tensor(math.sqrt(2.0 / num_features)))
        self.register_buffer("basis0", math.sqrt(2.0 / num_features) * torch.cos(self.b))
        self.weights = nn.Parameter(torch.zeros(num_features))
        self.unit_gain, self.ridge = unit_gain, ridge

    def raw_basis(self, r):
        return self.scale.to(r) * torch.cos(r[..., None] * self.W.to(r) + self.b.to(r))

    def regularize(self, gram, jitter):
        return gram + torch.diag(torch.full((self.weights.numel(),), self.ridge + jitter, dtype=gram.dtype,
                                            device=gram.device))

    def penalty(self):
        return self.ridge * torch.mean(self.weights.square())


def make_kernel(c, phi_init, rff_seed=None):
    """The kernel of Config `c`, starting from phi_init."""
    kind = c.kernel_func
    if kind == "legendre":
        return Legendre(c.legendre_num_basis, c.legendre_max_distance, c.legendre_degree_penalty,
                        c.legendre_unit_gain_at_zero)
    if kind == "random_fourier":
        return RandomFourier(c.rff_num_features, c.rff_gamma, rff_seed, c.rff_unit_gain_at_zero)
    if kind == "null":
        return Null()
    args = dict(phi=phi_init, positive=c.kernel_positive_phi_transform, floor=c.kernel_phi_floor)
    if kind == "simplified_degroot":
        return DeGroot(**args, unit_gain=c.degroot_unit_gain)
    return {"bounded_confidence": BoundedConfidence, "rzb": RZB,
            "sigmoidal_bounded_confidence": SigmoidalBoundedConfidence}[kind](**args)


# ---------------------------------------------------------------- noise

def weighted_median(v, w):
    keep = (w > 0) & torch.isfinite(v)
    v, w = v[keep], w[keep]
    if v.numel() == 0:
        return torch.tensor(float("nan"), dtype=v.dtype)
    order = torch.argsort(v)
    cw = torch.cumsum(w[order], 0)
    return v[order][torch.searchsorted(cw, 0.5 * cw[-1]).clamp(max=v.numel() - 1)]


class Gaussian:
    def pdf(self, r, s):
        s = torch.as_tensor(s, dtype=r.dtype, device=r.device)
        return 1.0 / (s * math.sqrt(2 * math.pi)) * torch.exp(-0.5 * torch.clamp((r / s) ** 2, max=700.0))

    def cdf(self, r, s):
        return 0.5 * torch.special.erfc(-r / (torch.as_tensor(s, dtype=r.dtype) * math.sqrt(2.0)))

    dev = staticmethod(torch.square)          # the deviation a scale averages
    root = staticmethod(torch.sqrt)           # scale from mean deviation

    def null_arm(self, w, delta):
        return w * delta * delta

    def lmbda(self, delta, k, Bw):
        num = (Bw * delta[:, None] * k).sum(dtype=torch.float64)
        den = (Bw * k.square()).sum(dtype=torch.float64)
        return torch.tensor(float("nan")) if not torch.isfinite(den) or den <= 1e-12 else (num / den).to(delta.dtype)

    def location(self, delta, w):
        num, den = (w * delta).sum(dtype=torch.float64), w.sum(dtype=torch.float64)
        return torch.tensor(float("nan")) if den <= 1e-12 else (num / den).to(delta.dtype)


class Laplace(Gaussian):
    def pdf(self, r, s):
        s = torch.as_tensor(s, dtype=r.dtype, device=r.device).clamp_min(torch.finfo(r.dtype).tiny)
        return (1.0 / (2.0 * s)) * torch.exp(-torch.clamp(r.abs() / s, max=700.0))

    def cdf(self, r, s):
        z = r / torch.as_tensor(s, dtype=r.dtype)
        return torch.where(r < 0, 0.5 * torch.exp(z), 1.0 - 0.5 * torch.exp(-z))

    dev = staticmethod(torch.abs)
    root = staticmethod(lambda v: v)

    def null_arm(self, w, delta):
        return w * delta.abs()

    def lmbda(self, delta, k, Bw):
        nz = k.abs() > 0
        return torch.tensor(float("nan")) if not nz.any() else weighted_median(
            delta[:, None].expand_as(k)[nz] / k[nz], (Bw * k.abs())[nz])

    def location(self, delta, w):
        return torch.tensor(float("nan")) if w.sum() <= 1e-12 else weighted_median(delta, w).to(delta.dtype)


NOISE = {"gaussian": Gaussian(), "laplace": Laplace()}


# ---------------------------------------------------------------- events

@dataclass
class Events:
    """One row per scored update: who moved (x_self), how far (delta), and its candidate
    influencers with their opinions (cand_x) and attention weights."""
    delta: torch.Tensor
    x_self: torch.Tensor
    cand_x: torch.Tensor
    attention: torch.Tensor
    valid: torch.Tensor
    cand_valid: torch.Tensor

    @property
    def E(self):
        return self.delta.shape[0]

    def __getitem__(self, s):
        return Events(*(getattr(self, f)[s] for f in self.__dataclass_fields__))


def attention_weights(adj, x, node_mask, edge_mask, transition_mask):
    """a[m, i, j, t]: uniform over i's neighbours j with an opinion at t."""
    m, n, tau = x.shape
    A = adj.float().to(x.device)
    pair = (A.unsqueeze(-1).expand(m, n, n, tau) * transition_mask[:, :, None, :]
            * node_mask[:, None, :, :] * edge_mask.unsqueeze(-1))
    den = pair.sum(-2, keepdim=True)
    return torch.where(den > 0, pair / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(pair))


def active_actor(transition_mask):
    """(M, T): the one node whose transition is scored at each step of an async panel, -1 if none."""
    any_active = transition_mask.any(1)
    return torch.where(any_active, transition_mask.long().argmax(1), torch.full_like(any_active, -1, dtype=torch.long))


def make_events(x, adj, node_mask, edge_mask, mask, is_async, mean_field, dtype=torch.float64):
    """Events of the transitions in `mask` (M, N, T) of float64 opinions x."""
    m, n, tau = x.shape
    xf, delta = x.to(dtype), F.pad(x.diff(dim=-1), (0, 1)).to(dtype)
    if is_async:
        actor = active_actor(mask)
        idx = actor.clamp(min=0)
        valid = mask.gather(1, idx.unsqueeze(1)).squeeze(1) & (actor >= 0)
        att = torch.empty(m, n, tau, dtype=dtype, device=x.device)
        step = max(1, 20_000_000 // max(m * n * n, 1))
        for t in range(0, tau, step):
            s = slice(t, t + step)
            block = attention_weights(adj, xf[..., s], node_mask[..., s], edge_mask, mask[..., s])
            att[..., s] = block.gather(1, idx[:, None, None, s].expand(m, 1, n, block.shape[-1])).squeeze(1).to(dtype)
        att = torch.where(valid[:, None, :], att, torch.zeros_like(att))
        rows = (adj != 0)[torch.arange(m, device=x.device)[:, None], idx]
        ev = Events(delta.gather(1, idx.unsqueeze(1)).reshape(-1), xf.gather(1, idx.unsqueeze(1)).reshape(-1),
                    xf.permute(0, 2, 1).reshape(m * tau, n), att.permute(0, 2, 1).reshape(m * tau, n),
                    valid.reshape(-1), (node_mask.permute(0, 2, 1) & rows).reshape(m * tau, n))
    else:
        T = tau - 1
        att = attention_weights(adj, xf, node_mask, edge_mask, mask)[..., :T].to(dtype)
        valid = node_mask[..., :T] & node_mask[..., 1:] & mask[..., :T]
        cand = node_mask[..., :T].unsqueeze(1).expand(m, n, n, T) & (adj != 0).unsqueeze(-1)
        ev = Events(delta[..., :T].permute(0, 2, 1).reshape(-1), xf[..., :T].permute(0, 2, 1).reshape(-1),
                    xf[..., :T].unsqueeze(1).expand(m, n, n, T).permute(0, 3, 1, 2).reshape(-1, n),
                    att.permute(0, 3, 1, 2).reshape(-1, n), valid.permute(0, 2, 1).reshape(-1),
                    cand.permute(0, 3, 1, 2).reshape(-1, n))
    ev.valid = ev.valid & ((ev.attention * ev.cand_valid.to(dtype)).sum(-1) > 0)
    if mean_field:
        xbar = (ev.attention * ev.cand_valid.to(dtype) * ev.cand_x).sum(-1, keepdim=True)
        ones = torch.ones_like(xbar)
        ev = Events(ev.delta, ev.x_self, xbar, ones, ev.valid, ones.bool())
    return ev


def event_index(ev, x, mask, is_async):
    """(market, node, time) of every valid event."""
    m, n, tau = x.shape
    e = ev.valid.nonzero().flatten()
    if is_async:
        return torch.stack([e // tau, active_actor(mask).clamp(min=0).reshape(-1)[e], e % tau], -1)
    T = tau - 1
    return torch.stack([e // (T * n), e % (T * n) % n, e % (T * n) // n], -1)


# ---------------------------------------------------------------- EM

class EM:
    def __init__(self, kernel, c, sigma):
        self.k, self.noise = kernel, NOISE[c.noise_model]
        self.c = c
        self.lmbda = 1.0 if c.lmbda_init is None else float(c.lmbda_init)
        self.sigma, self.pi, self.mu_null = float(sigma), float(c.init_pi), 0.0
        self.b_null = (float(sigma if c.init_b_null is None else c.init_b_null)
                       if c.decoupled_scale else None)
        self.gamma = None

    def interaction(self, ev):
        x_i = ev.x_self[:, None].expand_as(ev.cand_x)
        if not isinstance(self.k, Basis):
            return self.k(x_i, ev.cand_x)
        out = torch.zeros_like(ev.cand_x)      # basis kernels only at real neighbour slots
        out[ev.cand_valid] = self.k(x_i[ev.cand_valid], ev.cand_x[ev.cand_valid])
        return out

    def null_pdf(self, ev):
        return self.noise.pdf(ev.delta - self.mu_null, self.b_null if self.c.decoupled_scale else self.sigma)

    def log_likelihood(self, ev, k=None):
        """Per valid event log-likelihood."""
        k = self.interaction(ev) if k is None else k
        q = self.noise.pdf(ev.delta[:, None] - self.lmbda * k, self.sigma)
        mass = (ev.attention * q * ev.cand_valid.to(q.dtype)).sum(-1)
        if self.c.infer_pi:
            mass = (1.0 - self.pi) * self.null_pdf(ev) + self.pi * mass
        mass = mass[ev.valid]
        return torch.log(mass.clamp_min(torch.finfo(mass.dtype).tiny))

    @torch.no_grad()
    def e_step(self, ev):
        """B[e, j]: probability that neighbour j moved event e (times the gate's pi share)."""
        chunk = self.c.e_step_chunk_size or ev.E
        parts = [self._e_step(ev[s:s + chunk]) for s in range(0, ev.E, chunk)]
        self.gamma = torch.cat([g for _, g in parts]) if self.c.infer_pi else None
        return torch.cat([b for b, _ in parts])

    def _e_step(self, ev):
        q = self.noise.pdf(ev.delta[:, None] - self.lmbda * self.interaction(ev), self.sigma)
        gamma = None
        if self.c.infer_pi:
            q_bar = (ev.attention * q * ev.cand_valid.to(q.dtype)).sum(-1)
            num = self.pi * q_bar
            gamma = num / ((1.0 - self.pi) * self.null_pdf(ev) + num).clamp_min(torch.finfo(q.dtype).tiny)
        log_pq = ev.attention.log().add_(q.clamp_min_(torch.finfo(q.dtype).tiny).log_())
        log_pq.masked_fill_(~ev.cand_valid, float("-inf"))
        log_den = torch.logsumexp(log_pq, -1, keepdim=True)
        B = log_pq.sub_(log_den).exp_()
        B.masked_fill_(~(ev.valid[:, None] & torch.isfinite(log_den)), 0.0)
        return (B * gamma.unsqueeze(-1) if gamma is not None else B), gamma

    def kernel_loss(self, ev, B):
        k = self.interaction(ev)
        w = (ev.valid[:, None] & ev.cand_valid).to(k.dtype)

        def loss(k):
            residual = ev.delta[:, None] - torch.tensor(self.lmbda, dtype=k.dtype) * k
            return (B * w * self.noise.dev(residual)).sum(dtype=torch.float64) / ev.valid.sum().clamp_min(1).double()
        out = torch.utils.checkpoint.checkpoint(loss, k, use_reentrant=False) if self.c.kernel_loss_checkpoint else loss(k)
        penalty = self.k.penalty()
        return out if penalty is None else out + penalty.to(out)

    def kernel_step(self, ev, B):
        c = self.c
        if c.optimization_method == "closed_form":
            return self.k.closed_form(ev, B, self.lmbda)
        if c.optimization_method == "bisection":
            return self.bisection(ev, B)
        params = [p for p in self.k.parameters() if p.requires_grad]
        if not params:
            return
        opt = (torch.optim.Adam if c.optimizer_class == "adam" else torch.optim.Adagrad)(self.k.parameters(), lr=c.gd_rate, eps=1e-10)
        clip, previous, calm = 1.0 * max(1, sum(p.numel() for p in params)) ** 0.5, None, 0
        for step in range(c.max_gd_iter):
            opt.zero_grad()
            loss = self.kernel_loss(ev, B)
            if not torch.isfinite(loss):
                break
            loss.backward()
            norm = sum(p.grad.norm(2).item() ** 2 for p in self.k.parameters() if p.grad is not None) ** 0.5
            if norm > clip:
                for p in self.k.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip / norm)
            opt.step()
            with torch.no_grad():
                new = self.kernel_loss(ev, B).item()
            change = abs(previous - new) / (abs(previous) + 1e-12) if previous is not None else np.inf
            previous = new
            if step + 1 < 5:
                continue
            calm = calm + 1 if change <= 1e-5 and norm <= 0.1 else 0
            if calm >= 3:
                break

    @torch.no_grad()
    def bisection(self, ev, B):
        """Golden-section search for a scalar phi on [0.01, the opinion range]."""
        x = torch.cat([ev.x_self, ev.cand_x[ev.cand_valid]])
        span = float(x.max() - x.min()) if self.c.bisection_max_distance is None else self.c.bisection_max_distance
        lo, hi = 0.01, max(span, 0.05)

        def loss(phi):
            self.k.phi = nn.Parameter(torch.tensor([float(phi)], dtype=self.k.phi.dtype, device=self.k.phi.device))
            value = self.kernel_loss(ev, B).item()
            return value if np.isfinite(value) else float("inf")
        gr = (np.sqrt(5.0) + 1.0) / 2.0
        x1, x2 = hi - (hi - lo) / gr, lo + (hi - lo) / gr
        f1, f2 = loss(x1), loss(x2)
        for _ in range(self.c.bisection_max_iter):
            if hi - lo < 1e-6:
                break
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - (hi - lo) / gr
                f1 = loss(x1)
            elif f2 < f1:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + (hi - lo) / gr
                f2 = loss(x2)
            else:
                lo, hi = x1, x2
                x1, x2 = hi - (hi - lo) / gr, lo + (hi - lo) / gr
                f1, f2 = loss(x1), loss(x2)
        loss((lo + hi) / 2.0)

    @torch.no_grad()
    def scale_step(self, ev, B, k):
        """Closed-form updates of lmbda, the noise scale and the gate."""
        c, noise = self.c, self.noise
        w = (ev.valid[:, None] & ev.cand_valid).to(k.dtype)
        valid = ev.valid.to(k.dtype)
        if c.infer_lmbda:
            new = float(noise.lmbda(ev.delta, k, B * w))
            self.lmbda = max(new, c.lmbda_min) if np.isfinite(new) else self.lmbda
        if c.infer_sigma:
            kernel_arm = (B * w * noise.dev(ev.delta[:, None] - torch.tensor(self.lmbda, dtype=k.dtype) * k)).sum(dtype=torch.float64)
            if c.decoupled_scale:
                den = (self.gamma * valid).sum(dtype=torch.float64)
                new = float(noise.root(kernel_arm / den).to(k.dtype)) if den > 1e-12 else float("nan")
            elif c.infer_pi:
                null_arm = (noise.null_arm(1.0 - self.gamma, ev.delta) * valid).sum(dtype=torch.float64)
                new = float(noise.root((kernel_arm + null_arm) / valid.sum(dtype=torch.float64).clamp_min(1.0)).to(k.dtype))
            else:
                den = (B * w).sum(dtype=torch.float64)
                new = float(noise.root(kernel_arm / den).to(k.dtype)) if den > 1e-12 else float("nan")
            self.sigma = max(new, 1e-8) if np.isfinite(new) else self.sigma
        if c.infer_mu_null:
            new = float(noise.location(ev.delta, (1.0 - self.gamma) * valid))
            self.mu_null = new if np.isfinite(new) else self.mu_null
        if c.decoupled_scale:
            weight = (1.0 - self.gamma) * valid
            den = weight.sum(dtype=torch.float64)
            new = (float(noise.root((weight * noise.dev(ev.delta - self.mu_null)).sum(dtype=torch.float64) / den).to(k.dtype))
                   if den > 1e-12 else float("nan"))
            if np.isfinite(new):
                self.b_null = max(new, 1e-12, c.b_null_min or 0.0)
        if c.infer_pi:
            new = float((self.gamma * valid).sum(dtype=torch.float64) / valid.sum(dtype=torch.float64).clamp_min(1.0))
            self.pi = min(max(new, c.pi_min), c.pi_max) if np.isfinite(new) else self.pi

    def state(self):
        return (copy.deepcopy(self.k.state_dict()), self.lmbda, self.sigma, self.pi, self.b_null, self.mu_null)

    def fit(self, ev):
        """EM until the training log-likelihood stops improving; ends at the best state seen."""
        c, history, best, best_ll, calm = self.c, [], self.state(), -np.inf, 0
        for _ in range(c.max_em_iter):
            B = self.e_step(ev)
            self.kernel_step(ev, B)
            with torch.no_grad():
                k = self.interaction(ev)
            self.scale_step(ev, B, k)
            with torch.no_grad():
                ll = float(self.log_likelihood(ev, k).sum())
            if ll > best_ll:
                best_ll, best = ll, self.state()
            if history:
                calm = calm + 1 if abs(ll - history[-1]) / (abs(history[-1]) + 1e-12) <= c.ll_stopping_criterion else 0
            history.append(ll)
            if calm >= c.ll_stop_patience:
                break
        kernel, self.lmbda, self.sigma, self.pi, self.b_null, self.mu_null = best
        self.k.load_state_dict(kernel)
        return history

    def predict(self, ev):
        """Expected update of every event under the influence component."""
        k = self.interaction(ev)
        return self.lmbda * (ev.attention * ev.cand_valid.to(ev.attention.dtype) * k).sum(-1)
