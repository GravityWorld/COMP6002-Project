"""Real-execution verification for the pinned correction-head architecture
(correction_head/model.py). Run this in the opencood conda env, on the
real machine -- it needs torch, which isn't available in the cloud
sandbox or the remote-device bridge's own Linux VM this was drafted from
(both checked directly: `import torch` raises ModuleNotFoundError in
each). Until this has been run for real and the printed numbers reported
back, treat model.py as a paper design, same status as P3's crop-caching
code before its own real run.

What this checks, each one required before the Oct 4 architecture gate
can be called passed (scope_and_timeline_oct19.md: "exact architecture
pinned down against a real gossip-round timing before the depth run"):
  1. Forward pass on random data of the exact real shapes (384x7x7 crops,
     the aux-feature dims) runs without shape errors.
  2. count_parameters() on the real instantiated nn.Module matches
     correction_head/count_params.py's arithmetic (388,098) exactly --
     any mismatch means the two have drifted and needs fixing before
     anything else here is trusted.
  3. One simulated local-training step (a few SGD steps on a synthetic
     ~200-fragment batch, matching "buffer ~200 frames" from the one-
     gossip-round spec in task1_datasets_pipeline_dfl.md Sec 4) is timed
     on CPU, since a real car's local step is what the edge-compute
     budget argument rests on -- report the printed per-step and
     per-200-step time back.

Usage (from the MastersProj repo root, opencood conda env active):
    python -m correction_head.verify_on_device
(NOT `python correction_head/verify_on_device.py` directly -- running it as
a bare script puts this file's own directory, not the repo root, at the
front of sys.path, so the `from correction_head.model import ...` line
below can't find its own package. `python -m` runs it as part of the
`correction_head` package instead, from the repo root, which is what
makes that absolute import resolve. The sys.path fallback below also
makes the bare-script form work, belt-and-braces, in case -m is
forgotten.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Fallback for `python correction_head/verify_on_device.py` (bare-script
# invocation): put the repo root (this file's grandparent) on sys.path so
# the absolute `correction_head.model` import below resolves even without
# `-m`. Harmless / a no-op when already run as `-m correction_head.verify_on_device`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_head.model import (
    BEV_CROP_CHANNELS,
    BEV_CROP_SIZE_PX,
    FRAGMENT_AUX_DIM,
    PAIR_AUX_DIM,
    CorrectionHead,
    count_parameters,
)

EXPECTED_PARAM_COUNT = 388_098  # from correction_head/count_params.py, after the Wed 7 Oct 2026 frag_extent_norm feature


def make_synthetic_batch(n_fragments: int = 200, n_pairs: int = 600, device: str = "cpu"):
    crops = torch.randn(n_fragments, BEV_CROP_CHANNELS, BEV_CROP_SIZE_PX, BEV_CROP_SIZE_PX, device=device)
    frag_aux = torch.randn(n_fragments, FRAGMENT_AUX_DIM, device=device)
    pair_index = torch.randint(0, n_fragments, (n_pairs, 2), device=device)
    pair_aux = torch.randn(n_pairs, PAIR_AUX_DIM, device=device)
    labels = torch.randint(0, 2, (n_pairs,), device=device).float()
    return crops, frag_aux, pair_index, pair_aux, labels


def main():
    model = CorrectionHead()

    n_params = count_parameters(model)
    print(f"Instantiated model param count: {n_params:,}")
    print(f"Expected (count_params.py):     {EXPECTED_PARAM_COUNT:,}")
    if n_params != EXPECTED_PARAM_COUNT:
        print("MISMATCH -- model.py and count_params.py have drifted apart. Fix before proceeding.")
    else:
        print("MATCH.")
    print()

    crops, frag_aux, pair_index, pair_aux, labels = make_synthetic_batch()
    logits = model(crops, frag_aux, pair_index, pair_aux)
    print(f"Forward pass OK. logits shape: {tuple(logits.shape)} (expected (600,))")
    assert logits.shape == (600,)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_steps = 5
    t0 = time.perf_counter()
    for _ in range(n_steps):
        crops, frag_aux, pair_index, pair_aux, labels = make_synthetic_batch()
        opt.zero_grad()
        logits = model(crops, frag_aux, pair_index, pair_aux)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    per_step = elapsed / n_steps
    print()
    print(f"{n_steps} local SGD steps (200 fragments / 600 pairs each) on CPU: {elapsed:.3f}s total, {per_step*1000:.1f}ms/step")
    print("Report this number back -- it's the per-round local-training cost the")
    print("edge-compute affordability argument rests on.")


if __name__ == "__main__":
    main()
