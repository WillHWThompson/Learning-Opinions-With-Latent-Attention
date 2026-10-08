"""spinos: Sakketou et al. (2022) LREC, SPINOS Reddit stance dataset,
https://github.com/caisa-lab/SPINOS-dataset (SPINOS_official_dataset_v1.1.pkl).

One asynchronous unit per topic, one step per post in timestamp order (stance_not_inferrable
dropped). A post is stance-bearing unless undecided or majority sarcastic; it sets its author's
opinion, and is scored if the author posted before. A user's first stance is backfilled to time 0
and observable after their first post. Edges join a reply's author to the parent's author.
"""
import pickle

import numpy as np

from _panel import RAW, download, sim_ids, unit, write_panel

URL = ("https://raw.githubusercontent.com/caisa-lab/SPINOS-dataset/main/"
       "SPINOS_official_dataset_v1.1.pkl")
STANCE = {"s_against": 0.0, "against": 0.25, "undecided": 0.5, "favor": 0.75, "s_favor": 1.0}

with open(download(URL, RAW / "spinos" / "SPINOS_official_dataset_v1.1.pkl"), "rb") as f:
    data = pickle.load(f)
units = []
for sim_id, topic in zip(sim_ids("spinos"), ["abortion", "brexit", "capitalism", "feminism"]):
    posts = data[data["topic"] == topic]
    author_of = {str(cid): a for cid, a in posts["author_id"].items()}
    df = posts[posts["annotation"].isin(STANCE.keys())].sort_values("timestamp")
    bearing = (df["annotation"] != "undecided") & (df["is_sarcastic"] != "2/3")
    index = {u: i for i, u in enumerate(dict.fromkeys(df["author_id"]))}
    x = np.zeros((len(index), len(df) + 1))
    scored, observable = np.zeros(x.shape, dtype=bool), np.zeros(x.shape, dtype=bool)
    for t, (author, stance, is_bearing) in enumerate(
            zip(df["author_id"], df["annotation"].map(STANCE), bearing)):
        x[:, t + 1] = x[:, t]
        u = index[author]
        if is_bearing:
            x[u, t + 1] = stance
        if not observable[u, t]:
            observable[u, t + 1:] = True
            if is_bearing:
                x[u, :t + 2] = stance
        elif is_bearing:
            scored[u, t] = True
    edges = set()
    for author, parent in zip(df["author_id"], df["parent_id"]):
        p = author_of.get(parent) if isinstance(parent, str) and parent else None
        if p in index and p != author:
            edges.add(tuple(sorted((index[author], index[p]))))
    deltas = np.diff(x).ravel()  # signed nonzero moves, unlike mad_sigma
    nz = deltas[np.abs(deltas) > 1e-8]
    units.append(unit(sim_id, f"spinos_{topic}", x, 1.4826 * float(np.median(np.abs(
        nz - np.median(nz)))), sorted(edges), scored, observable, mode="asynchronous"))
write_panel("spinos", units)
