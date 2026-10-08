"""Score every configuration of both sweeps into figure_data/scores.parquet: python scripts/scores.py

One row per configuration: held-out log-likelihood summed over folds, per-fold scores, the
winning restart's kernel shape and the noise and gate parameters of every fold.
"""
import pandas as pd

from kernel_inference import results, selection

table = pd.concat([selection.scores(results.fits(s)) for s in results.SWEEPS], ignore_index=True)
table.to_parquet(selection.SCORES, index=False)
print(f"wrote {selection.SCORES} ({len(table)} configurations)")
