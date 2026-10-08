# Frozen figure inputs

Four supplementary figures read inputs that this repository cannot regenerate. They were
computed from simulation sweeps run with an earlier version of the fitting code, whose
simulated data and fitted models lived in the project's database. They are kept here as
data, and the figure scripts read them directly.

| File | Figure | Where it came from |
|---|---|---|
| `sbc_summary_v2.json` | `fig_sbc_diagnostic` | Recovery of the sigmoidal bounded-confidence kernel over a grid of true parameters. |
| `sweep_metrics_*.csv` (six) | `fig_tv_sweep` | Recovery error (TV, trajectory-weighted TV, KL) across the true parameter, for six simulation sweeps of 250 fits each. |
| `synthetic_2x2_results.json` | `synthetic_2x2` | Gaussian and Laplace noise, with and without the mover gate, fitted to 30 simulations with a known mover fraction. |
| `rff_vs_mlp.json` | `fig_rff_recovery` | Gain curves of random-feature and neural-network kernels fitted to four simulations on the Les Miserables graph. |
