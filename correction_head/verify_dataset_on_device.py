"""Real-torch verification for correction_head/dataset.py -- run this in
the opencood conda env on the real machine (python -m
correction_head.verify_dataset_on_device; the sys.path fallback below also
lets `python correction_head/verify_dataset_on_device.py` work, same
belt-and-braces fix verify_on_device.py needed -- see decisions.md, Thu 1
Oct 2026).

This is the one step dataset.py's own test suite (test_dataset.py, 17/17
pass against real repair_labels/crops data) could NOT cover: building a
real torch batch via collate_batches() and feeding it through the real,
execution-verified CorrectionHead (model.py, 388,098 params, after the Wed 7 Oct 2026 frag_extent_norm feature) end to end.
Everything up to the torch tensor conversion has already been confirmed
against real on-disk data in the device-bridge's own (torch-less) Linux
VM -- see dataset.py's module docstring and decisions.md for that run's
output (194 fragments / 174 pairs across the 2 scenarios' currently-
cached crops).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_head import dataset as dataset_module
from correction_head.dataset import build_samples_for_scenario, collate_batches
from correction_head.model import CorrectionHead, count_parameters

# Discovered dynamically from repair_labels/'s own scenario subfolders
# (same pattern cluster_failure_audit.load_qualifying_scenarios() uses off
# the manifest) rather than hardcoded -- this was hardcoded to the original
# 2 qualifying scenarios and silently never updated after the Thu 2 Oct 2026
# OPV2V reassembly unlocked 10 more (found via a real run that only showed
# those same 2 scenarios' samples despite perception_cache.py having cached
# crops for all 12).
def _discover_scenarios():
    return sorted(
        d.name for d in dataset_module.DEFAULT_REPAIR_LABELS_DIR.iterdir()
        if d.is_dir()
    )


SCENARIOS = _discover_scenarios()


def main() -> None:
    dataset_module.assert_matches_model()
    print("dataset.py's duplicated constants match model.py's real ones. OK.")

    all_samples = []
    for scenario_id in SCENARIOS:
        samples = build_samples_for_scenario(scenario_id, arm="A")
        all_samples.extend(samples)
        for s in samples:
            print(f"  {scenario_id} {s.frame_id}/{s.cav_id}: "
                  f"{s.n_fragments} fragments, {s.n_pairs} pairs")

    if not all_samples:
        print("No usable training samples found -- see dataset.py's module "
              "docstring, GAP #2 (crop-caching coverage). Nothing to run "
              "the model on.")
        return

    crops, frag_aux, pair_index, pair_aux, labels = collate_batches(all_samples)
    print(f"\nCollated real batch: crops={tuple(crops.shape)} "
          f"frag_aux={tuple(frag_aux.shape)} pair_index={tuple(pair_index.shape)} "
          f"pair_aux={tuple(pair_aux.shape)} labels={tuple(labels.shape)}")

    model = CorrectionHead()
    model.eval()
    print(f"Instantiated model param count: {count_parameters(model):,}")

    import torch
    with torch.no_grad():
        logits = model(crops, frag_aux, pair_index, pair_aux)
    expected_shape = (pair_index.shape[0],)
    assert tuple(logits.shape) == expected_shape, (logits.shape, expected_shape)
    print(f"\nForward pass OK on the REAL repair-labels/crop-cache data. "
          f"logits shape: {tuple(logits.shape)} (expected {expected_shape})")

    pos_rate = float(labels.mean()) if labels.numel() else float("nan")
    print(f"Real positive (same-object) pair rate in this batch: {pos_rate:.4f}")

    # one real backward step, same spirit as verify_on_device.py's
    # synthetic-batch timing check -- confirms gradients actually flow
    # through dataset.py's real tensors, not just a forward pass.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    optimizer.zero_grad()
    logits = model(crops, frag_aux, pair_index, pair_aux)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"One real backward step on real data OK. loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
