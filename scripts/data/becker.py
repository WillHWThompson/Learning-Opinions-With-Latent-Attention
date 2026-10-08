"""becker_hubless: Becker, Brackbill and Centola (2017) PNAS 114(26), doi
10.1073/pnas.1615978114, Dataset S1. PNAS can refuse scripted downloads; if so, save it from a
browser to data/raw/becker/pnas.1615978114.sd01.csv.

One unit per Centralized (group, task), members by subject_id, responses as
(response - truth) / sd_pool; members with three-round std above 3 (or undefined) are dropped
and a missing response repeats the previous round (0 in round 1). The hub stays as every leaf's
neighbour but its transitions are unscored. sigma is stored as 1.0.
"""
import numpy as np
import pandas as pd

from _panel import RAW, download, sim_ids, unit, write_panel

URL = ("https://www.pnas.org/doi/suppl/10.1073/pnas.1615978114/suppl_file/"
       "pnas.1615978114.sd01.csv")
ROUNDS = ["response_1", "response_2", "response_3"]

df = pd.read_csv(download(URL, RAW / "becker" / "pnas.1615978114.sd01.csv"))
df = df[df.network == "Centralized"].copy()
for col in ROUNDS:
    df[col] = (df[col] - df.truth) / df.sd_pool
groups = [g.sort_values("subject_id") for _, g in
          df[df[ROUNDS].std(axis=1) <= 3].groupby(["group_number", "task"]) if g.is_central.any()]
units = []
for m, (sim_id, group) in enumerate(zip(sim_ids("becker_hubless"), groups)):
    x = group[ROUNDS].to_numpy(dtype=np.float64)
    x[:, 0] = np.nan_to_num(x[:, 0], nan=0.0)
    for t in (1, 2):
        x[:, t] = np.where(np.isnan(x[:, t]), x[:, t - 1], x[:, t])
    hub = int(np.flatnonzero(group.is_central.to_numpy())[0])
    active = np.ones(x.shape, dtype=bool)
    active[:, -1] = active[hub] = False
    edges = sorted([min(i, hub), max(i, hub)] for i in range(len(x)) if i != hub)
    units.append(unit(sim_id, f"becker_2017_hubless_m{m}", x, 1.0, edges, active))
write_panel("becker_hubless", units)
