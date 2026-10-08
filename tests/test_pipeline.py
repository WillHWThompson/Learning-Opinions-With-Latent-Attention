"""The claims the paper's numbers rest on."""
from pathlib import Path

import pytest
import torch

from kernel_inference import cv, selection, sweep
from kernel_inference.fit import fit
from kernel_inference.panels import load_panel
from kernel_inference.results import configs, fold, model

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "gaussian_gate_pairwise/takacs_control/bc"


@pytest.fixture(scope="module")
def reference():
    c = configs()[REFERENCE]
    return fit(c, 0, c.phi_init[0])


def test_reference_fit_reproduces_the_paper(reference):
    record, _ = reference
    assert record["phi"] == [0.9366376996040344]
    assert record["lmbda"] == 0.29587325215941473
    assert record["sigma"] == 0.12083463850430642
    assert record["test_mll"] == 37.880878714175545
    assert record["test_n"] == 93


def test_a_model_rebuilt_from_its_parameters_scores_like_the_fit(reference):
    record, _ = reference
    rebuilt, f = model(record), fold(record)
    with torch.no_grad():
        assert float(rebuilt.log_likelihood(f.events(f.test)).sum()) == pytest.approx(record["test_mll"], rel=1e-12)


def test_panels_load():
    panel = load_panel("takacs_control")
    assert tuple(panel.x.shape) == (190, 2, 4)
    assert not panel.transition_mask[..., -1].any()


def test_folds_partition_the_trajectories():
    mask = load_panel("becker_hubless").transition_mask
    tests = [cv.split(mask, 5, k)[1] for k in range(5)]
    assert torch.equal(sum(t.int() for t in tests).bool(), mask)
    sizes = [int((t.int().sum(-1) > 0).sum()) for t in tests]
    assert max(sizes) - min(sizes) <= 1


def test_the_paper_sweep_has_every_configuration():
    assert len(sweep.load(ROOT / "workflow/config/paper.yml")) == 1104


@pytest.mark.skipif(not selection.SCORES.exists(), reason="needs figure_data/scores.parquet")
def test_selection_picks_the_reported_models():
    scores = selection.load_scores()
    picks = {ds: selection.best_model(ds, "pairwise", grain=scores) for ds in ("senate", "markets")}
    assert (picks["senate"].kernel, picks["senate"].noise_variant) == ("random_fourier", "laplace_nogate")
    assert (picks["markets"].kernel, picks["markets"].noise_variant) == ("rzb", "laplace_gate")
