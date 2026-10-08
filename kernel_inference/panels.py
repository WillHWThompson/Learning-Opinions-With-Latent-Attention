"""Opinion panels from data/panels/<name>/: units.parquet (one row per unit with its graph; a unit
packs trajectories told apart by batch_idx) and observations.parquet (one row per sim_id, batch_idx,
node_id, timestep with the opinion, is_observable and is_active, the latter scoring the transition)."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PANEL_DIR = Path(__file__).resolve().parents[1] / "data" / "panels"


@dataclass
class Panel:
    """Every trajectory of a panel padded to (M, N, T); x_raw is x before dequantization."""
    x: torch.Tensor
    x_raw: torch.Tensor
    adj: torch.Tensor
    node_mask: torch.Tensor
    transition_mask: torch.Tensor
    edge_mask: torch.Tensor
    sim_ids: list
    sigma: float


def jitter(n, t, h, sim_id, b):
    """(n, t) of U(-h/2, h/2), one generator per (sim_id, batch_idx, node) seeded by a hash."""
    seed = lambda i: int.from_bytes(hashlib.blake2b(np.array([sim_id, b, i, 0], "<i8").tobytes(), digest_size=8)
                                    .digest(), "little") & (2 ** 63 - 1)
    gens = [torch.Generator().manual_seed(seed(i)) for i in range(n)]
    return (torch.stack([torch.rand(t, generator=g, dtype=torch.float64) for g in gens]) - 0.5) * h


def read_panel(name, panel_dir=PANEL_DIR):
    d = Path(panel_dir) / name
    return pd.read_parquet(d / "units.parquet"), pd.read_parquet(d / "observations.parquet")


def load_panel(name, no_opinion_value=None, grid_h=None, panel_dir=PANEL_DIR):
    """A panel as a batch of trajectories; grid_h adds U(-grid_h/2, grid_h/2) to every opinion (see jitter)."""
    units, obs = read_panel(name, panel_dir)
    rows = []  # (sim_id, batch, sigma, x, adj, node_mask, transition_mask), one per trajectory
    for u in units.itertuples():
        o = obs[obs.sim_id == u.sim_id]
        b, n, t = o.batch_idx.to_numpy(), o.node_id.to_numpy(), o.timestep.to_numpy()
        shape = (max(int(u.num_batches or 1), b.max() + 1), max(int(u.node_num or 0), n.max() + 1), t.max() + 1)
        x, seen, scored = np.zeros(shape, np.float32), np.ones(shape, bool), np.ones(shape, bool)
        x[b, n, t], seen[b, n, t], scored[b, n, t] = o.node_value, o.is_observable, o.is_active
        x, scored = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(scored[..., :-1])
        if no_opinion_value is not None:  # no transition into or out of the sentinel is scored
            sentinel = torch.isclose(x, torch.tensor(float(no_opinion_value), dtype=x.dtype), atol=1e-9, rtol=0.0)
            scored &= ~(sentinel[..., :-1] | sentinel[..., 1:])
        edges = np.asarray(json.loads(u.edge_list), dtype=np.int64).reshape(-1, 2)
        size = max(shape[1], int(edges.max()) + 1 if len(edges) else 0)
        adj = torch.zeros((size, size), dtype=torch.float32)
        adj[edges[:, 0], edges[:, 1]] = adj[edges[:, 1], edges[:, 0]] = 1.0
        rows += [(u.sim_id, i, u.sigma, x[i], adj, torch.as_tensor(seen[i]), scored[i]) for i in range(shape[0])]

    M, N, T = len(rows), max(r[3].shape[0] for r in rows), max(r[3].shape[1] for r in rows)
    x = torch.zeros(M, N, T, dtype=torch.float64)
    x_raw = x if grid_h is None else torch.zeros(M, N, T, dtype=torch.float64)
    adj, edge_mask = torch.zeros(M, N, N, dtype=torch.float32), torch.zeros(M, N, N, dtype=torch.bool)
    node_mask, transition_mask = torch.zeros(M, N, T, dtype=torch.bool), torch.zeros(M, N, T, dtype=torch.bool)
    for m, (sim_id, i, _, xm, am, nm, tm) in enumerate(rows):
        n, t = xm.shape
        x_raw[m, :n, :t], x_raw[m, :n, t:] = xm, xm[:, -1:]  # the last opinion repeats: padded deltas are zero
        if grid_h is not None:
            xm = xm + jitter(n, t, grid_h, sim_id, i)
            x[m, :n, :t], x[m, :n, t:] = xm, xm[:, -1:]
        adj[m, :len(am), :len(am)] = am
        node_mask[m, :n, :t], transition_mask[m, :n, :t - 1], edge_mask[m, :n, :n] = nm, tm, am[:n, :n].bool()
    return Panel(x, x_raw, adj, node_mask, transition_mask, edge_mask, [r[0] for r in rows],
                 float(np.mean([r[2] for r in rows])))
