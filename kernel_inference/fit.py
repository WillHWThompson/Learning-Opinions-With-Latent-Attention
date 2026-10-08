"""Fit one configuration on one cross-validation fold and score it on the held-out transitions."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import cv
from .config import Config
from .model import EM, Parametric, event_index, make_events, make_kernel
from .panels import load_panel


@dataclass
class Fold:
    c: Config
    x: torch.Tensor  # (M, N, T) opinions
    adj: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    train: torch.Tensor  # transitions fitted
    test: torch.Tensor  # transitions scored
    sigma: float  # the panel's noise scale

    def events(self, mask):
        return make_events(self.x, self.adj, self.node_mask, self.edge_mask, mask, self.c.use_async_em,
                           self.c.representation == "neighbour_mean_field", getattr(torch, self.c.events_dtype))


def fold(c, k, device="cpu"):
    """Fold k of configuration c; only transitions that move the recorded opinion are fitted or scored."""
    p = load_panel(c.panel, c.no_opinion_value, c.grid_h if c.dequantize else None)
    moved = F.pad(p.x_raw.diff(dim=-1), (0, 1)) != 0
    train, test = cv.split(p.transition_mask & moved, c.cv_num_folds, k)
    return Fold(c, *(t.to(device) for t in (p.x.float().double(), p.adj, p.node_mask, p.edge_mask, train, test)),
                p.sigma)


def fit(c, k, phi_init, rff_seed=None, device="cpu"):
    """Fit configuration c on fold k from one restart. Returns (record, fitted EM)."""
    f = fold(c, k, device)
    em = EM(make_kernel(c, phi_init, rff_seed).to(device), c, c.sigma_init if c.sigma_init is not None else f.sigma)
    history = em.fit(f.events(f.train))
    ev = f.events(f.test)
    with torch.no_grad():
        ll = em.log_likelihood(ev)
    n, state = int(f.test.sum()), em.k.state_dict()
    learned = next((state[p] for p in ("weights", "phi") if p in state), torch.tensor([]))
    record = {"fit": c.fit, "cv_fold_index": k, "phi_init": list(phi_init), "rff_seed": rff_seed,
              "phi": em.k.effective_phi().detach().tolist() if isinstance(em.k, Parametric) else [],
              "kernel_params": learned.double().flatten().tolist(),
              "lmbda": em.lmbda, "sigma": em.sigma, "pi": em.pi, "b_null": em.b_null, "mu_null": em.mu_null,
              "train_mll_history": history, "test_n": n, "test_mll": float(ll.sum()),
              "test_mll_per_transition": float(ll.sum()) / n,
              "test_ll": ll.tolist(), "test_index": event_index(ev, f.x, f.test, c.use_async_em).tolist()}
    return record, em
