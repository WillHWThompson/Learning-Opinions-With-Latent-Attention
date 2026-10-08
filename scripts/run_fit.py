"""Fit one configuration on every fold from every restart.

    python scripts/run_fit.py gaussian_gate_pairwise/takacs_control/bc paper [cuda]

Writes results/<sweep>/parts/<configuration>.parquet, one row per fit.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

from kernel_inference.fit import fit
from kernel_inference.results import configs

key, sweep, device = sys.argv[1], sys.argv[2], (sys.argv[3:] or ["cpu"])[0]
c = configs()[key]
seeds = c.rff_seed if c.kernel_func == "random_fourier" else [None]
rows = [{**fit(c, k, phi, seed, device=device)[0], "init": i}
        for k in c.cv_fold_index
        for i, (phi, seed) in enumerate(itertools.product(c.phi_init, seeds))]
out = Path(__file__).resolve().parents[1] / "results" / sweep / "parts" / f"{key}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(out, index=False)
