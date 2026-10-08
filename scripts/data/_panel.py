"""Shared helpers for the panel builders. Each writes <out>/<name>/units.parquet and
observations.parquet, where <out> is its only argument (default data/panels)."""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def sim_ids(panel):
    return json.loads((ROOT / "config" / "canonical_sim_ids.json").read_text())[panel]


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
            path.write_bytes(response.read())
    return path


def mad_sigma(deltas):
    """1.4826 * MAD of the absolute nonzero moves."""
    nz = np.abs(deltas[np.abs(deltas) > 1e-12])
    return float(1.4826 * np.median(np.abs(nz - np.median(nz))))


def mysql_float(values):
    """float32 printed to 6 digits, as the original MySQL FLOAT column returned it."""
    return np.array([float(f"{v:.6g}") for v in np.asarray(values, dtype=np.float32)])


def unit(sim_id, graph, x, sigma, edges, active, observable=True, mode="synchronous"):
    """(unit row, observation rows) for x of shape (batch, node, time), or (node, time) for one
    unbatched unit; active and observable broadcast against x."""
    row = {"sim_id": sim_id, "graph": graph, "node_num": x.shape[-2],
           "num_batches": len(x) if x.ndim == 3 else None, "sigma": sigma, "lmbd": 1.0,
           "update_mode": mode, "edge_list": json.dumps([[int(a), int(b)] for a, b in edges])}
    x = x.reshape((-1,) + x.shape[-2:])
    b, n, t = np.indices(x.shape)
    cells = {"sim_id": sim_id, "batch_idx": b, "node_id": n, "timestep": t, "node_value": x,
             "is_active": active, "is_observable": observable}
    return row, pd.DataFrame({k: np.broadcast_to(v, x.shape).ravel() for k, v in cells.items()})


def write_panel(name, units):
    units, obs = zip(*units)
    units = pd.DataFrame(list(units))
    obs = pd.concat(obs, ignore_index=True).astype({"sim_id": np.int64, "is_active": bool,
                                                    "is_observable": bool})
    obs["node_value"] = mysql_float(obs["node_value"])
    units["sigma"] = mysql_float(units["sigma"])
    obs = obs.sort_values(["sim_id", "batch_idx", "node_id", "timestep"], ignore_index=True)
    d = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "panels") / name
    d.mkdir(parents=True, exist_ok=True)
    units.to_parquet(d / "units.parquet", index=False)
    obs.to_parquet(d / "observations.parquet", index=False, compression="zstd")
