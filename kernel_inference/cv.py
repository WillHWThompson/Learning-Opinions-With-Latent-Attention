"""Cross-validation over trajectories: those with a scored transition are shuffled and dealt into k folds."""
import torch


def split(mask, k, fold):
    """(train, test) transition masks, shape (M, N, T), with fold `fold` of k held out."""
    scored = mask.reshape(-1, mask.shape[-1]).any(1)
    index = scored.nonzero().flatten()
    labels = torch.full(scored.shape, -1, dtype=torch.int64)
    labels[index[torch.randperm(len(index), generator=torch.Generator().manual_seed(0))]] = torch.arange(len(index)) % k
    test = mask & (labels.reshape(mask.shape[:-1]) == fold)[..., None]
    return mask & ~test, test
