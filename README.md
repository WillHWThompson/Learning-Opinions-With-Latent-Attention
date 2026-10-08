# Kernel inference for opinion dynamics

Code and data for inferring the interaction kernel of stochastic opinion-dynamics models from
observed opinion trajectories, by expectation maximization with held-out cross-validation. It
reproduces every figure and table of the paper.

## Reproduce the paper

```bash
conda env create -f environment.yml && conda activate mrc-2023 && pip install -e .
snakemake --profile workflow/profiles/slurm           # everything, fits on a GPU cluster
snakemake --profile workflow/profiles/local paper     # figures and tables from fitted results
```

The full run fits 1,152 model configurations on 19 panels (each over 5 folds and up to 4
restarts, about 18,000 fits) and 960 synthetic discrimination datasets, then exports the figure
inputs and draws `output/paper/`. Edit the partitions in `workflow/profiles/slurm/config.yaml`
for your cluster. To skip the fitting, put the released `results/` archive at the repo root and
run the `paper` target. The legacy-style figures use Palatino when it is installed and fall back
to DejaVu Serif otherwise, which changes their glyphs but not their content.

## Layout

| Stage | Code | Output |
|---|---|---|
| Panels | `scripts/data/<source>.py` download and transform raw data | `data/panels/<panel>/` |
| Sweep | `workflow/config/<sweep>.yml`, expanded by `kernel_inference/sweep.py` into `Config`s | |
| Fit | `scripts/run_fit.py <configuration> <sweep>` fits every fold and restart | `results/<sweep>/fits.parquet` |
| Score | `scripts/scores.py` | `figure_data/scores.parquet` |
| Export, draw | `scripts/figures/<module>.py export`, then `scripts/figures/<module>.py` | `figure_data/`, `output/paper/figures/` |
| Synthetic | `scripts/synthetic.py <experiment>` | `figure_data/synthetic_*` |
| Tables | `scripts/tables.py` | `output/paper/tables/` |

`workflow/Snakefile` lists every output with the code that makes it.

The library, `kernel_inference/`:

- `config.py`: every option of a fit and its default, as one pydantic `Config`.
- `model.py`: kernels (bounded confidence, sigmoidal, DeGroot, SAR, Legendre and random Fourier
  bases), noise models (Gaussian, Laplace), the event tensors and the EM.
- `panels.py`, `cv.py`, `fit.py`: load a panel, split it into node-blocked folds, fit one fold.
- `results.py`, `selection.py`: read a sweep's fits, rebuild any fitted model from its stored
  parameters, score configurations and choose the model each figure reports.
- `synthetic.py`: simulators with known truth. `style.py`: colours, names and dataset order.

A configuration is one panel, noise model, representation, kernel and (for random features)
bandwidth, named `<noise>_<arm>/<panel>/<kernel>`. Each row of `fits.parquet` is one fit with
its options, fitted parameters, training likelihood history, held-out score and per-event
held-out log-likelihoods.

Model selection: within each fold the restart with the highest training likelihood is kept; a
configuration's score is its held-out log-likelihood summed over folds per held-out transition;
a panel's model is the best-scoring configuration among those a figure allows, except that on
the random-feature bandwidth grid the smallest bandwidth within one standard error of the best
is taken.

## Data

| Panel | Source | Rebuilt from raw |
|---|---|---|
| plos_gauging, plos_counting | PLOS ONE S1 Dataset, doi 10.1371/journal.pone.0230584 | exactly |
| takacs, takacs_control, takacs_disliking, takacs_disliking_interior, takacs_study1 | DANS, doi 10.17026/DANS-XJZ-P8RK | exactly |
| adams, adams_p02, adams_p05, adams_p08 | PLOS ONE e0275473, S2 File | exactly |
| kozitsin | Harvard Dataverse, doi 10.7910/DVN/H3ZBHR | exactly |
| spinos | github.com/caisa-lab/SPINOS-dataset | exactly |
| senate, house | Voteview and GovInfo BILLSTATUS (pinned dates in `scripts/data/congress.py`) | senate exactly |
| becker_hubless | PNAS SD01, doi 10.1073/pnas.1615978114 (download by hand) | |
| cmv, markets | Reddit threads with a stance model; Manifold Markets | no, committed as fitted |
| synthetic_2e_network | simulated | committed as fitted |

`data/panels/` holds every panel exactly as the fits read it. Four supplementary figures
(`fig_sbc_diagnostic`, `fig_tv_sweep`, `synthetic_2x2`, `fig_rff_recovery`) read the frozen
inputs in `data/figure_data_frozen/`, produced by an earlier version of this code.

## Tests

`pytest` checks that a reference fit reproduces the paper's numbers exactly, that a model rebuilt
from its stored parameters scores like the original fit, and that the panels, folds, sweep and
model selection are as the paper reports.
