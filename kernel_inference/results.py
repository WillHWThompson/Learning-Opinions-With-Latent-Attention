"""Fitted results: results/<sweep>/fits.parquet, one row per fit, and the models they define."""
from functools import cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

from . import fit as F, sweep
from .config import Config
from .model import EM, make_kernel

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ("paper", "synthetic_2e")


@cache
def configs():
    return {c["fit"]: Config(**c) for s in SWEEPS for c in sweep.load(ROOT / f"workflow/config/{s}.yml")}


def fits(sweep="paper", columns=None, events=False):
    """The fits of a sweep; per-event held-out scores only when `events`."""
    path = ROOT / "results" / sweep / "fits.parquet"
    columns = columns or [c for c in pq.read_schema(path).names if events or c not in ("test_ll", "test_index")]
    return pd.read_parquet(path, columns=columns)


def fold(row):
    """The panel and fold split a fit was trained and scored on."""
    return F.fold(configs()[row["fit"]], int(row["cv_fold_index"]))


def model(row):
    """The fitted EM of a row, rebuilt from its parameters (refitted if none were stored)."""
    c, k, phi = configs()[row["fit"]], int(row["cv_fold_index"]), list(row["phi_init"])
    seed = None if pd.isna(row.get("rff_seed")) else int(row["rff_seed"])
    if row["kernel_params"] is None:
        return F.fit(c, k, phi, seed)[1]
    em = EM(make_kernel(c, phi, seed), c, row["sigma"])
    state = em.k.state_dict()
    for name in state.keys() & {"weights", "phi"}:  # one of them; the null kernel learns nothing
        state[name] = torch.tensor(list(row["kernel_params"]), dtype=state[name].dtype)
        em.k.load_state_dict(state)
    for p in ("lmbda", "sigma", "pi", "b_null", "mu_null"):
        if not pd.isna(row[p]):
            setattr(em, p, float(row[p]))
    return em
