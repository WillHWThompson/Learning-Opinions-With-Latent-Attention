"""plos_gauging and plos_counting: Vande Kerckhove et al. (2016) PLOS ONE, S1 Dataset of the
correction, doi 10.1371/journal.pone.0230584.

Every (group of 6, game) is a K6 batch, batch = group * 30 + game, opinions over the answer
scale. A transition is scored when the player answered both rounds and someone else the first.
"""
import io
import zipfile

import numpy as np
import pandas as pd

from _panel import RAW, download, mad_sigma, sim_ids, unit, write_panel

URL = ("https://journals.plos.org/plosone/article/file"
       "?id=10.1371/journal.pone.0230584.s001&type=supplementary")

with zipfile.ZipFile(download(URL, RAW / "plos_2016" / "S1_Dataset.zip")) as archive:
    for game, scale in [("gauging", 100.0), ("counting", 500.0)]:
        with archive.open(f"Xpers_{game}_game_plos_one_header.csv") as f:
            df = pd.read_csv(io.TextIOWrapper(f), header=[0, 1, 2], index_col=[0, 1])
        # (group, player, round, game) -> (group * 30 + game, player, round)
        X = df.to_numpy(dtype=float).reshape(-1, 6, 3, 30).transpose(0, 3, 1, 2).reshape(-1, 6, 3)
        present = ~np.isnan(X)
        others = present.sum(axis=1, keepdims=True) - present
        active = np.ones_like(present)
        active[:, :, :-1] = present[:, :, :-1] & present[:, :, 1:] & (others[:, :, :-1] > 0)
        x = np.nan_to_num(X / scale)
        sigma = mad_sigma(np.diff(x)[active[:, :, :-1]])
        edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        [sim_id] = sim_ids(f"plos_{game}")
        write_panel(f"plos_{game}", [unit(sim_id, f"plos_2016_{game}_k6", x, sigma, edges,
                                          active, present)])
