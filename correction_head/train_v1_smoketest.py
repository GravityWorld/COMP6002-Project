"""Correction head v1 training loop -- SMOKE TEST, not a generalisation run.

WHY THIS EXISTS / WHAT IT IS NOT
---------------------------------
Per the Thu 1 Oct 2026 / Fri 2 Oct 2026 decision (decisions.md): with only
2 qualifying scenarios in the current manifest, a real train/val/test
split by scenario (P6's own design -- see make_scenario_splits() in
dataset.py) degenerates (val/test round to empty). Rather than force a
frame-level holdout, which would leak (adjacent frames in the same
scenario share scene geometry -- exactly what P6's own "split by scenario,
never frame" rule exists to avoid), this v1 pass trains on ALL real
samples and tracks TRAIN LOSS ONLY, as an architecture/plumbing sanity
check: does the head learn anything at all from real data (loss goes
down, doesn't NaN, isn't a trivial majority-class collapse), not a
reportable generalisation number. Do not cite this run's numbers as
validation/test metrics anywhere -- there is no held-out data here.

Once more OPV2V scenarios are downloaded and run through P0-P6 (in
progress as of this script's writing), re-run this against
make_scenario_splits() for a real scenario-level split and retire the
all-in-train mode below.

USAGE
-----
    python -m correction_head.train_v1_smoketest [--epochs 200] [--lr 1e-3]

Run from the repo root, opencood conda env active (needs torch + the real
P3/P4 cache on disk -- not runnable in the cloud sandbox or the device-
bridge's own Linux VM, same as verify_on_device.py / verify_dataset_on_device.py).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_head import dataset as dataset_module
from correction_head.dataset import (
    DEFAULT_REPAIR_LABELS_DIR,
    build_samples_for_scenario,
    collate_batches,
)
from correction_head.model import CorrectionHead, count_parameters


def discover_scenarios(repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR) -> list[str]:
    """Scenario IDs with a repair-labels subfolder, discovered from disk --
    not hardcoded, so this script picks up new scenarios automatically
    once more OPV2V data is downloaded and run through P0-P6."""
    return sorted(p.name for p in Path(repair_labels_dir).iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_smoketest_loss.csv"),
    )
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + "\n")

    dataset_module.assert_matches_model()
    print("dataset.py's duplicated constants match model.py's real ones. OK.\n")

    scenarios = discover_scenarios()
    print(f"Discovered {len(scenarios)} scenario(s) with repair labels: {scenarios}")
    if len(scenarios) <= 2:
        print(
            "NOTE: only "
            f"{len(scenarios)} qualifying scenario(s) -- this is the known, "
            "documented degeneracy (decisions.md). Training on ALL samples, "
            "tracking TRAIN LOSS ONLY. Do not treat this run's numbers as a "
            "validation/test metric.\n"
        )

    all_samples = []
    for scenario_id in scenarios:
        samples = build_samples_for_scenario(scenario_id, arm=args.arm)
        all_samples.extend(samples)
        for s in samples:
            print(f"  {scenario_id} {s.frame_id}/{s.cav_id}: "
                  f"{s.n_fragments} fragments, {s.n_pairs} pairs")

    if not all_samples:
        print("No usable training samples found -- see dataset.py's module "
              "docstring (GAP #2). Nothing to train on. Aborting.")
        return

    crops, frag_aux, pair_index, pair_aux, labels = collate_batches(all_samples)
    crops = crops.to(device)
    frag_aux = frag_aux.to(device)
    pair_index = pair_index.to(device)
    pair_aux = pair_aux.to(device)
    labels = labels.to(device)
    n_fragments, n_pairs = crops.shape[0], pair_index.shape[0]
    pos_rate = float(labels.mean())
    print(
        f"\nFull real dataset: {len(all_samples)} (scenario, frame, cav) samples, "
        f"{n_fragments} fragments, {n_pairs} pairs, "
        f"positive (same-object) rate {pos_rate:.4f}\n"
    )

    model = CorrectionHead().to(device)
    print(f"Model param count: {count_parameters(model):,}\n")

    # Class-imbalance guard: with only ~6-7% positive pairs, an unweighted
    # BCE loss lets the model collapse to "always predict negative" and
    # still look like it's minimising loss. pos_weight counteracts that so
    # a shrinking loss here is actually evidence of learning, not of the
    # trivial-majority-class exploit.
    n_pos = float(labels.sum())
    n_neg = float(labels.numel() - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    print(f"BCEWithLogitsLoss pos_weight={pos_weight.item():.2f} ({n_pos:.0f} pos / {n_neg:.0f} neg pairs)\n")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    history = []
    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(crops, frag_aux, pair_index, pair_aux)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = float((preds == labels).float().mean())
            # majority-class baseline for comparison: always predict 0
            majority_acc = 1.0 - pos_rate

        history.append((epoch, float(loss.item()), acc))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:4d}/{args.epochs}  loss={loss.item():.4f}  "
                f"train_acc={acc:.4f}  (majority-class baseline={majority_acc:.4f})"
            )

    elapsed = time.perf_counter() - t0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc"])
        writer.writerows(history)

    first_loss = history[0][1]
    last_loss = history[-1][1]
    early_avg = sum(h[1] for h in history[: max(1, len(history) // 10)]) / max(1, len(history) // 10)
    late_avg = sum(h[1] for h in history[-max(1, len(history) // 10):]) / max(1, len(history) // 10)

    print(f"\n{args.epochs} epochs in {elapsed:.1f}s ({elapsed/args.epochs*1000:.1f}ms/epoch).")
    print(f"Loss: epoch 1 = {first_loss:.4f}  ->  final epoch = {last_loss:.4f}")
    print(f"First-10%-of-run avg loss = {early_avg:.4f}  ->  last-10%-of-run avg loss = {late_avg:.4f}")
    if late_avg < early_avg:
        print("SANITY CHECK PASSED: loss decreased over the run -- the head is learning something from real data.")
    else:
        print("SANITY CHECK FAILED: loss did not decrease -- investigate before trusting this architecture/loader "
              "(check lr, check pos_weight, check for a data/label bug) before reporting this as a working v1.")
    print(f"\nPer-epoch loss/accuracy log written to: {out_csv}")
    print(
        "\nReminder: this is a TRAIN-LOSS-ONLY smoke test on the full degenerate-split "
        "dataset (decisions.md, Thu 1 Oct / Fri 2 Oct 2026). Not a validation number. "
        "Re-run against make_scenario_splits() once enough scenarios exist for a real split."
    )


if __name__ == "__main__":
    main()
