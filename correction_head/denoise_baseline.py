"""Denoise-first baseline -- REAL held-out run, Sat 3 Oct / Sun 4 Oct gate.

Answers the reviewer's obvious question that has to sit next to the learned
correction head's own number: "why not just denoise / distance-threshold
instead of training a model?" (execution_plan_daily.md, Sat 3 Oct: "the
reviewer's obvious 'why not just denoise' question needs an answer at the
same time the head's own number exists.")

This baseline uses ONLY pair_aux[:, 0] -- dist_norm, i.e. centroid_distance_m
/ CANDIDATE_PAIR_RADIUS_M (see correction_head/dataset.py,
build_samples_for_scenario) -- with a single scalar threshold: predict
same-object (label 1) iff dist_norm < threshold. No crops, no fragment
features, no learned weights beyond that one number. This is the simplest
repair rule a reviewer could propose in place of the head: "just merge
fragments that are close together."

Threshold selection: a 1-D grid search over dist_norm in [0, 1] at 0.01
steps, evaluated on TRAIN pairs ONLY (never val) -- same held-out discipline
as train_v1.py, so the two results are comparable side by side on the SAME
val scenarios. The selection criterion is Youden's J (recall - FPR) on
train, maximized -- a standard, self-contained operating-point rule that
does not need to know anything about the learned head's own result (tuning
this baseline's threshold against the head's val FPR would make the
"independent baseline" claim circular).

Reuses train_v1.py's make_scenario_splits()/build_split_batch() with the
SAME default val_frac/test_frac/seed, so both scripts see identical
train/val/test scenario sets -- required for the two numbers to be
comparable at all.

USAGE
-----
    python -m correction_head.denoise_baseline [--val-frac 0.2] [--test-frac 0.2] [--seed 0]

Run from the repo root, opencood conda env active (needs torch + the real
P3/P4 cache on disk, same as train_v1.py -- not runnable in the cloud
sandbox or the device-bridge's own Linux VM).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_head import dataset as dataset_module
from correction_head.dataset import DEFAULT_MANIFEST_PATH, make_scenario_splits
from correction_head.train_v1 import build_split_batch


def confusion_metrics(preds, labels):
    """preds/labels: 1-D numpy float arrays of 0./1. Returns the same metric
    set train_v1.py's evaluate() reports, for a direct side-by-side read."""
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    clean_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "clean_rate": clean_rate, "fpr": fpr, "recall": recall, "precision": precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(_REPO_ROOT / "correction_head" / "runs" / "denoise_baseline_summary.json"),
    )
    args = parser.parse_args()

    import numpy as np

    dataset_module.assert_matches_model()

    split = make_scenario_splits(
        manifest_path=DEFAULT_MANIFEST_PATH,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"Scenario split (seed={args.seed}) -- MUST match train_v1.py's split for the two numbers to be comparable:")
    print(f"  train ({len(split.train_scenarios)}): {sorted(split.train_scenarios)}")
    print(f"  val   ({len(split.val_scenarios)}): {sorted(split.val_scenarios)}")
    print(f"  test  ({len(split.test_scenarios)}): {sorted(split.test_scenarios)}  <- RESERVED, not touched by this script\n")

    if not split.val_scenarios:
        print("ABORTING: val split is empty.")
        return

    train_result = build_split_batch(split.train_scenarios, args.arm)
    val_result = build_split_batch(split.val_scenarios, args.arm)
    if train_result is None or val_result is None:
        print("ABORTING: zero usable samples in train or val split.")
        return

    train_samples, train_batch = train_result
    val_samples, val_batch = val_result
    _, _, _, train_pair_aux, train_labels = train_batch
    _, _, _, val_pair_aux, val_labels = val_batch

    dist_train = train_pair_aux[:, 0].numpy()
    labels_train = train_labels.numpy()
    dist_val = val_pair_aux[:, 0].numpy()
    labels_val = val_labels.numpy()

    print(f"Train: {len(train_samples)} samples, {len(labels_train)} pairs, positive rate {labels_train.mean():.4f}")
    print(f"Val (held-out): {len(val_samples)} samples, {len(labels_val)} pairs, positive rate {labels_val.mean():.4f}\n")

    # Threshold search on TRAIN only.
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t, best_j = None, -1.0
    best_train_metrics = None
    for t in thresholds:
        preds = (dist_train < t).astype(np.float32)
        m = confusion_metrics(preds, labels_train)
        j = m["recall"] - m["fpr"]
        if j > best_j:
            best_j, best_t, best_train_metrics = j, float(t), m

    print(f"Chosen threshold (max Youden's J on TRAIN only): dist_norm < {best_t:.2f}")
    print(f"  TRAIN at this threshold: recall={best_train_metrics['recall']:.4f} "
          f"FPR={best_train_metrics['fpr']:.4f} J={best_j:.4f}\n")

    # Apply frozen threshold to held-out val. No tuning here.
    preds_val = (dist_val < best_t).astype(np.float32)
    v = confusion_metrics(preds_val, labels_val)
    acc_val = float((preds_val == labels_val).mean())
    val_majority = 1.0 - float(labels_val.mean())

    print("=== HELD-OUT (val) RESULT -- denoise-first baseline ===")
    print(f"Val scenarios: {sorted(split.val_scenarios)}")
    print(f"Val pairs: {len(labels_val)}  positive rate: {labels_val.mean():.4f}  "
          f"(TP={v['tp']:.0f} FP={v['fp']:.0f} TN={v['tn']:.0f} FN={v['fn']:.0f})")
    print(f"Val accuracy: {acc_val:.4f}  (majority-class baseline: {val_majority:.4f})")
    print(f"Val CLEAN-rate: {v['clean_rate']:.4f}   Val FPR: {v['fpr']:.4f}")
    print(f"Val recall (catches actual violations): {v['recall']:.4f}   precision: {v['precision']:.4f}")

    summary = {
        "method": "denoise_first_distance_threshold",
        "threshold_dist_norm_lt": best_t,
        "threshold_selection": "max Youden's J (recall - FPR) on TRAIN pairs only",
        "seed": args.seed,
        "train_scenarios": sorted(split.train_scenarios),
        "val_scenarios": sorted(split.val_scenarios),
        "test_scenarios_reserved": sorted(split.test_scenarios),
        "n_train_pairs": len(labels_train),
        "n_val_pairs": len(labels_val),
        "val_positive_rate": float(labels_val.mean()),
        "val_acc": acc_val,
        "val_majority_baseline": val_majority,
        "val_clean_rate": v["clean_rate"],
        "val_fpr": v["fpr"],
        "val_recall": v["recall"],
        "val_precision": v["precision"],
        "val_tp": v["tp"], "val_fp": v["fp"], "val_tn": v["tn"], "val_fn": v["fn"],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nHeld-out summary JSON: {out_json}")
    print(
        "\nReminder: compare this val_clean_rate/val_fpr/val_recall directly against "
        "train_v1.py's own val result -- same scenarios, same seed, same split fracs."
    )


if __name__ == "__main__":
    main()
