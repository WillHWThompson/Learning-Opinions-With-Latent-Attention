"""kozitsin: Kozitsin, "Opinion dynamics of online social network users: a micro-level
analysis", Harvard Dataverse, doi 10.7910/DVN/H3ZBHR (X_opinions_cc, X_friends_cc.npz).

3000 egos drawn without replacement (seed 0), each a star unit: ego at node 0, friends (made
symmetric, no self loops, capped at 100 with seed 1000) at 1..k, all over three waves. Only the
ego's transitions are scored; sigma is 1.4826 * MAD of all ego deltas, zeros included.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from _panel import RAW, download, sim_ids, unit, write_panel

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/"

path = RAW / "kozitsin" / "X_opinions_cc_xi.npy"  # the opinion file's three x_i columns
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pd.read_csv(f"{DATAVERSE}4877227", usecols=[0, 1, 2], storage_options={
        "User-Agent": "Mozilla/5.0"}).to_numpy(dtype=np.float64))
xi = np.load(path)
d = np.load(download(f"{DATAVERSE}4877226", RAW / "kozitsin" / "X_friends_cc.npz"),
            allow_pickle=True)
adj = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
adj = (adj + adj.T).astype(bool).astype(np.float32)
adj.setdiag(0)
adj.eliminate_zeros()

egos = np.random.default_rng(0).choice(len(xi), 3000, replace=False)
deltas = np.diff(xi[egos], axis=1).ravel()
sigma = 1.4826 * np.median(np.abs(deltas - np.median(deltas)))
rng = np.random.default_rng(1000)
units = []
for block, (sim_id, ego) in enumerate(zip(sim_ids("kozitsin"), egos)):
    nb = adj.indices[adj.indptr[ego]:adj.indptr[ego + 1]]
    if len(nb) > 100:
        nb = rng.choice(nb, 100, replace=False)
    active = np.zeros((1 + len(nb), 3), dtype=bool)
    active[0, :2] = True
    units.append(unit(sim_id, f"kozitsin_fullgraph_n3000_seed0_blk{block}",
                      xi[np.concatenate([[ego], nb])], sigma,
                      [(0, k) for k in range(1, 1 + len(nb))], active))
write_panel("kozitsin", units)
