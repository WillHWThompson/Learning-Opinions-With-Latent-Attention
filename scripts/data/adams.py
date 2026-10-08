"""adams, adams_p02/p05/p08: Adams et al. (2022) PLOS ONE 17(10):e0275473, S2 File,
doi 10.1371/journal.pone.0275473.s006.

Each game is a dyad batch: node 0 is the participant at main.x - other.x, then user.x - other.x
after the advice; node 1 the red circle, pinned at 0 and never scored. Values are divided by
max |x| over all games, so the reliability panels (p = 0.2, 0.5, 0.8) share the pooled scale.
"""
import numpy as np
import pandas as pd

from _panel import RAW, download, mad_sigma, sim_ids, unit, write_panel

URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
       "&id=10.1371/journal.pone.0275473.s006")

df = pd.read_csv(download(URL, RAW / "adams" / "plos_one_data.csv"))
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.dropna(subset=["reliability", "main.x", "user.x", "other.x"])
df = df.sort_values(["sessionId", "timestamp"], kind="stable", ignore_index=True)
x = np.stack([df["main.x"] - df["other.x"], df["user.x"] - df["other.x"]], axis=1)
x = (x / np.abs(x).max()).astype(np.float32)
for name, level in {"adams": None, "adams_p02": 0.2, "adams_p05": 0.5, "adams_p08": 0.8}.items():
    keep = df["reliability"].notna() if level is None else np.isclose(
        df["reliability"], level, rtol=0, atol=1e-9)
    xk = np.stack([x[keep], np.zeros_like(x[keep])], axis=1)
    sigma = mad_sigma(xk[:, 0, 1].astype(np.float64) - xk[:, 0, 0])
    write_panel(name, [unit(sim_ids(name)[0], "plos_one_dyad_anchor0", xk, sigma, [(0, 1)],
                            [[True], [False]], mode=None)])
