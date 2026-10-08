# Learning Opinions with Latent Attention(LOLA)

Code and data for inferring the interaction kernel of stochastic opinion-dynamics models with latent attention mechanims

## Reproduce the paper
```bash
conda env create -f environment.yml && conda activate mrc-2023 && pip install -e .
snakemake --profile workflow/profiles/slurm           # everything, fits on a GPU cluster
snakemake --profile workflow/profiles/local paper     # figures and tables from fitted results
```
