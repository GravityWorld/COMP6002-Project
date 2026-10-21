"""Correction head v1 training -- REAL held-out run, Sun 4 Oct gate.

Supersedes train_v1_smoketest.py's train-only mode now that enough real
scenarios exist (12, after the Thu 2 Oct 2026 OPV2V reassembly -- see
decisions.md) for build_scenario_split() to produce a genuine, non-
degenerate train/val/test split by scenario (never by frame -- frame-level
splits leak adjacent frames across the split and inflate every number,
execution_plan_daily.md, Mon 14 Sep). train_v1_smoketest.py is kept as-is
for its own historical record (the Fri 2 Oct 2026 entries); this script is
what the Sun 4 Oct gate's own done-when actually asks for: "the head's
held-out ... number ... written down."

Split: default val_frac=0.2, test_frac=0.2 -> with 12 scenarios, 8 train /
2 val / 2 test. TEST SCENARIOS ARE NEVER TOUCHED BY THIS SCRIPT -- reserved
for the Week 6' results freeze (Tue 13 Oct). This script trains on train
scenarios only and reports a held-out number on the VAL scenarios. Treat
val numbers from repeated runs/tuning as development numbers, not a final
result -- the frozen test set is what the thesis cites.

pos_weight is computed from TRAIN labels only, never val -- using val
statistics to shape the training loss would leak val information into
training, defeating the point of holding it out.

USAGE
-----
    python -m correction_head.train_v1 [--epochs 200] [--lr 1e-3]
        [--val-frac 0.2] [--test-frac 0.2] [--seed 0]

Run from the repo root, opencood conda env active (needs torch + the real
P3/P4 cache on disk -- not runnable in the cloud sandbox or the device-
bridge's own Linux VM, same as every other correction_head/*.py script).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_head import dataset as dataset_module
from correction_head.dataset import (
    DEFAULT_MANIFEST_PATH,
    build_samples_for_scenario,
    build_samples_for_scenario_with_pair_weights,
    build_samples_for_scenario_with_hard_negative_weights,
    collate_batches,
    make_scenario_splits,
)
from correction_head.model import CorrectionHead, count_parameters


def build_split_batch(scenario_ids, arm, violation_lookup=None):
    """All samples across a set of scenario_ids, collated into one batch.
    Returns None (not an empty batch) if the scenario set has zero usable
    samples, so callers can distinguish 'no scenarios in this split' from
    'scenarios exist but nothing collated' -- both real, different cases.

    `violation_lookup` (Thu 8 Oct 2026, GAP #1 close-out): optional
    {(scenario_id, frame_id): (occupancy_violation_m2, perception_violation_m2)}
    dict, as dataset.load_violation_lookup() returns from a real
    P4_defence/compute_violation_scalars.py run. None (default, unchanged
    from before) means every fragment's free_occ_conflict_area/
    tracking_affinity aux features stay zero-filled, same as every run
    before this one."""
    samples = []
    for scenario_id in sorted(scenario_ids):
        samples.extend(
            build_samples_for_scenario(scenario_id, arm=arm, violation_lookup=violation_lookup)
        )
    if not samples:
        return None
    return samples, collate_batches(samples)


def build_split_batch_with_pair_weights(scenario_ids, arm, different_vehicle_weight, violation_lookup=None, other_weight=1.0):
    """Same as build_split_batch(), but also returns a per-pair weight
    tensor aligned with the collated labels -- TRAIN split only (see
    dataset.py's build_pair_type_weights() docstring for why: this is
    ground-truth-derived, train-time-only loss shaping, same status as
    pos_weight, never a model input). Mon 5 Oct 2026 finding (decisions.md):
    the head was dramatically worse than the denoise baseline specifically
    on different-vehicle negative pairs (43.4% FPR vs 3.9%) despite a
    dominant pooled FPR, because that bucket is only ~1.2% of all pairs
    and gets swamped in the gradient by the much larger background-
    background bucket. `different_vehicle_weight` multiplies the loss on
    exactly that bucket; everything else keeps weight 1.0."""
    samples = []
    weight_arrays = []
    for scenario_id in sorted(scenario_ids):
        s, w = build_samples_for_scenario_with_pair_weights(
            scenario_id, arm=arm, different_vehicle_weight=different_vehicle_weight,
            violation_lookup=violation_lookup, other_weight=other_weight,
        )
        samples.extend(s)
        weight_arrays.extend(w)
    if not samples:
        return None
    batch = collate_batches(samples)
    import numpy as np
    import torch

    weights = torch.from_numpy(np.concatenate(weight_arrays, axis=0)) if weight_arrays else torch.zeros(0)
    assert weights.shape[0] == batch[4].shape[0], (
        f"pair-weight/batch misalignment: {weights.shape[0]} weights vs {batch[4].shape[0]} labels -- "
        "build_split_batch_with_pair_weights()'s per-sample weight arrays did not concatenate to the "
        "same length as collate_batches()'s labels. Should not happen: both are built from the same "
        "per-scenario sample lists in the same order."
    )
    return samples, batch, weights


def build_split_batch_with_hard_negative_weights(
    scenario_ids, arm, different_vehicle_weight, hard_negative_weight,
    hard_dist_norm_threshold, hard_size_gap_threshold, violation_lookup=None,
    other_weight=1.0,
):
    """Hard-negative-mining counterpart to build_split_batch_with_pair_weights()
    -- "option 1" from the Tue 7 Oct 2026 discussion (decisions.md): instead
    of a flat different_vehicle_weight on every different-vehicle negative
    pair, give extra weight specifically to the hard subset (spatially far
    AND size-mismatched -- the exact pattern diagnose_false_merges.py found
    the head's real false merges concentrated on)."""
    samples = []
    weight_arrays = []
    for scenario_id in sorted(scenario_ids):
        s, w = build_samples_for_scenario_with_hard_negative_weights(
            scenario_id,
            arm=arm,
            different_vehicle_weight=different_vehicle_weight,
            hard_negative_weight=hard_negative_weight,
            hard_dist_norm_threshold=hard_dist_norm_threshold,
            hard_size_gap_threshold=hard_size_gap_threshold,
            violation_lookup=violation_lookup,
            other_weight=other_weight,
        )
        samples.extend(s)
        weight_arrays.extend(w)
    if not samples:
        return None
    batch = collate_batches(samples)
    import numpy as np
    import torch

    weights = torch.from_numpy(np.concatenate(weight_arrays, axis=0)) if weight_arrays else torch.zeros(0)
    assert weights.shape[0] == batch[4].shape[0], (
        f"pair-weight/batch misalignment: {weights.shape[0]} weights vs {batch[4].shape[0]} labels."
    )
    return samples, batch, weights


def evaluate(model, batch, loss_fn_unweighted, device, diff_vehicle_mask=None):
    """One no-grad forward pass. Returns a dict, not a tuple -- the gate's own
    done-when (execution_plan_daily.md, Sat 3 Oct) asks for "held-out CLEAN-rate
    / FPR", not accuracy. Accuracy is misleading here: at ~1-4% positive rate,
    a model that predicts "clean" for everything already scores >95% accuracy,
    so "DOES NOT BEAT the majority baseline" on raw accuracy does not by itself
    mean the head is useless -- it means accuracy is the wrong lens at this
    imbalance. Following this project's existing FPR convention (P4_defence/
    run_defence_check.py's "clean-condition FPR" -- fraction of truly-clean
    instances wrongly flagged), the pairwise label here is read the same way:
    label 0 = clean pair (no repair needed), label 1 = violation pair (needs
    repair). CLEAN-rate = TN / (TN+FP): of the pairs that are actually clean,
    how many the head correctly leaves alone. FPR = FP / (TN+FP) = 1 -
    CLEAN-rate: of the actually-clean pairs, how many it wrongly flags.
    Recall = TP / (TP+FN): of the pairs that actually need repair, how many
    the head catches -- this is the number that tells you whether the head is
    doing anything at all, since a trivial all-clean predictor gets a perfect
    CLEAN-rate/FPR but zero recall.
    Always uses an UNWEIGHTED BCE loss for the reported held-out loss number
    -- pos_weight is a training-time class-imbalance correction, not
    something that should distort a reported evaluation metric.

    `diff_vehicle_mask` (Tue 7 Oct 2026 fix): optional bool tensor, same
    length as `labels`, marking which VAL pairs are the specific
    different-real-vehicle NEGATIVE bucket (build_pair_type_weights()'s own
    definition). Pooled loss/FPR above are dominated by the background-
    background bucket (~96% of all pairs here) -- a checkpoint picked by
    lowest POOLED val loss can get there by being great at background-
    background while getting WORSE on this specific rare bucket, which is
    exactly the number this whole investigation is trying to minimise (see
    decisions.md, Tue 7 Oct 2026: a "best pooled loss" checkpoint scored
    48.35% different-vehicle FPR, worse than the unweighted 43.41%
    baseline, while pooled background-background FPR looked fine).
    `diff_vehicle_fpr` below is reported as NaN if this mask isn't passed
    or is all-False."""
    crops, frag_aux, pair_index, pair_aux, labels = batch
    model.eval()
    with __import__("torch").no_grad():
        logits = model(crops, frag_aux, pair_index, pair_aux)
        loss = loss_fn_unweighted(logits, labels)
        preds = (logits.sigmoid() > 0.5).float()
        acc = float((preds == labels).float().mean())
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
    model.train()
    pos_rate = float(labels.mean())
    clean_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    diff_vehicle_fpr = float("nan")
    if diff_vehicle_mask is not None:
        with __import__("torch").no_grad():
            n_bucket = float(diff_vehicle_mask.sum())
            if n_bucket > 0:
                bucket_fp = float(((preds == 1) & diff_vehicle_mask).sum())
                diff_vehicle_fpr = bucket_fp / n_bucket

    return {
        "loss": float(loss.item()),
        "acc": acc,
        "pos_rate": pos_rate,
        "n_pairs": int(labels.numel()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "clean_rate": clean_rate,
        "fpr": fpr,
        "recall": recall,
        "diff_vehicle_fpr": diff_vehicle_fpr,
        "precision": precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop training once this many consecutive LOGGED evals (every "
             "--log-every epochs) have passed with no new best VAL loss. "
             "None (default) = disabled, trains the full --epochs count "
             "unchanged from before -- opt in explicitly since it changes "
             "how many epochs a given run actually uses. Safe to combine "
             "with the Tue 7 Oct 2026 best-checkpoint fix: the checkpoint "
             "saved is always the best epoch regardless, this flag only "
             "controls when the loop stops wasting time past it.",
    )
    parser.add_argument(
        "--min-recall-floor",
        type=float,
        default=0.75,
        help="Wed 7 Oct 2026 fix, round 2: selecting the checkpoint by "
             "different-vehicle FPR alone (the first fix, same day) has its "
             "own degenerate failure mode -- a real run picked epoch 5, "
             "basically still at the start of training, where the model is "
             "so conservative it almost never predicts a merge at all. That "
             "gives a great (low) different-vehicle FPR for free, the same "
             "way an all-clean predictor does, but same-vehicle recall "
             "collapsed to 22.89% (established good runs sit around "
             "80-85%, denoise gets 76.31%) -- the model had stopped doing "
             "its actual job. An eval epoch now only COUNTS as a candidate "
             "best checkpoint if its overall VAL recall is >= this floor; "
             "among those, lowest different-vehicle FPR wins, same as "
             "before. If NO logged epoch ever clears the floor, training "
             "falls back to the last epoch and prints a loud warning "
             "instead of silently handing back a checkpoint that doesn't "
             "merge anything.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Try cuDNN's auto-tuned (benchmark=True) conv algorithm "
             "selection instead of the fixed deterministic one. Opt-in "
             "experiment only -- this run's own full-batch tensors never "
             "change shape epoch to epoch, which is exactly the case "
             "cudnn benchmarking is designed for, but it has not been "
             "verified here whether it changes the held-out numbers. "
             "Compare a --cudnn-benchmark run's result against a matching "
             "non-benchmark run before trusting a number produced with it.",
    )
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print/evaluate every N epochs (default 1 = every epoch, "
             "changed from the old default of 10 on Tue 7 Oct 2026 so the "
             "best epoch can be pinpointed exactly rather than to the "
             "nearest 10). Evaluating every epoch costs one extra forward "
             "pass over the val set per epoch -- small next to the train "
             "step's own cost, but not free; raise this back up if epoch "
             "count is large and that overhead starts to matter.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_loss.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_summary.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_model.pt"),
        help="Where to save the trained model's state_dict. Needed by "
             "visualize_corrections.py to actually run the head on real "
             "scenarios after training -- this script did not save one "
             "before Sat 4 Oct 2026.",
    )
    parser.add_argument(
        "--different-vehicle-weight",
        type=float,
        default=1.0,
        help="Loss multiplier (TRAIN pairs only, never a model input) on "
             "NEGATIVE pairs where both fragments are ground-truth DIFFERENT "
             "real vehicles -- the bucket evaluate_pair_types.py found the "
             "head failing on (43.4%% FPR vs denoise's 3.9%%, Mon 5 Oct 2026, "
             "decisions.md). 1.0 (default) = no change from the original "
             "unweighted-by-bucket loss. Try e.g. 5-10 and re-run the full "
             "evaluate_declustering.py / evaluate_pair_types.py chain "
             "afterward to check whether it actually closes the gap "
             "without hurting the other buckets -- not assumed, checked.",
    )
    parser.add_argument(
        "--other-weight",
        type=float,
        default=1.0,
        help="Loss multiplier (TRAIN pairs only) on every pair NOT in the "
             "different-vehicle-negative bucket -- background-background, "
             "background-vehicle, same-vehicle, and mixed/ambiguous pairs. "
             "1.0 (default) = unchanged from the original per-bucket scheme "
             "(only the different-vehicle bucket gets a non-1.0 weight). "
             "Added Fri 9 Oct 2026 after the weight=40.0 different-vehicle "
             "retrain traded same-vehicle recall (88.7%% -> 77.5%%) for the "
             "different-vehicle FPR win (21.84%% -> 3.68%%) -- lets you keep "
             "the other buckets at their own emphasis (e.g. 8.0, matching "
             "the old pre-bugfix different-vehicle-weight run) instead of "
             "letting them fall back to an unweighted 1.0 by default.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=None,
        help="Enables hard-negative mining (Tue 7 Oct 2026 'option 1', "
             "decisions.md) in place of the flat --different-vehicle-weight "
             "scheme: different-vehicle negative pairs that are BOTH "
             "spatially far (dist_norm > --hard-dist-norm-threshold) AND "
             "size-mismatched (abs_delta_log_n > --hard-size-gap-threshold) "
             "get this weight instead of --different-vehicle-weight; all "
             "other different-vehicle negatives still get "
             "--different-vehicle-weight. None (default) = disabled, falls "
             "back to the original flat scheme. Not a hardcoded inference-"
             "time distance cutoff -- purely a training-time sampling/loss "
             "emphasis, the model still decides what to do with a given pair.",
    )
    parser.add_argument("--hard-dist-norm-threshold", type=float, default=0.68)
    parser.add_argument("--hard-size-gap-threshold", type=float, default=1.0)
    parser.add_argument(
        "--violation-lookup",
        type=str,
        default=None,
        help="Thu 8 Oct 2026, GAP #1 close-out: path to the CSV "
             "P4_defence/compute_violation_scalars.py writes (scenario_id, "
             "frame_id, occupancy_violation_m2, perception_violation_m2). "
             "When given, every fragment's free_occ_conflict_area/"
             "tracking_affinity aux features (model.py indices 6/7) are the "
             "REAL frame-level violation magnitudes from P4's occupancy-"
             "consistency + perception-occupancy checks, broadcast to every "
             "fragment in that frame -- instead of the zero-fill every run "
             "before this one used (dataset.py's load_violation_lookup() "
             "returns None and build_sample() zero-fills when this is "
             "omitted, same as always). Run compute_violation_scalars.py "
             "once first to produce the CSV; it needs the opencood env and "
             "is not run automatically by this script.",
    )
    args = parser.parse_args()

    import torch

    # Fixes a real gap found Sat 4 Oct 2026: --seed only seeded the
    # scenario train/val/test split (make_scenario_splits), never torch's
    # own RNG -- so model weight init (and any other stochastic op during
    # training) was never reproducible even at "the same seed". A re-run
    # with identical data/split/hyperparameters produced a materially
    # different held-out result (CLEAN-rate 0.9746->0.9914, FPR
    # 0.0254->0.0086, recall 0.9277->0.8514) purely from this. Seeding both
    # torch and CUDA here makes a given --seed reproduce the same trained
    # model, not just the same data split.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # cuDNN picks convolution algorithms non-deterministically by
        # default regardless of the manual_seed calls above -- this model
        # is almost entirely Conv2d/GroupNorm, so without this a GPU run
        # is still not reproducible even now. Small perf cost, worth it
        # for a number that goes in the thesis.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        if args.cudnn_benchmark:
            print("cudnn.benchmark=True (opt-in speed experiment) -- verify this run's "
                  "numbers against a non-benchmark run before citing them.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + "\n")

    dataset_module.assert_matches_model()
    print("dataset.py's duplicated constants match model.py's real ones. OK.\n")

    violation_lookup = None
    if args.violation_lookup:
        violation_lookup = dataset_module.load_violation_lookup(args.violation_lookup)
        if violation_lookup is None:
            print(f"WARNING: --violation-lookup {args.violation_lookup!r} given but file not "
                  "found/unreadable -- falling back to zero-filled free_occ_conflict_area/"
                  "tracking_affinity features, same as if --violation-lookup had been omitted.\n")
        else:
            print(f"Loaded {len(violation_lookup)} (scenario, frame) violation rows from "
                  f"{args.violation_lookup} -- free_occ_conflict_area/tracking_affinity are REAL "
                  "for every (scenario, frame) covered, zero-filled for any not covered.\n")

    split = make_scenario_splits(
        manifest_path=DEFAULT_MANIFEST_PATH,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"Scenario split (seed={args.seed}):")
    print(f"  train ({len(split.train_scenarios)}): {sorted(split.train_scenarios)}")
    print(f"  val   ({len(split.val_scenarios)}): {sorted(split.val_scenarios)}")
    print(f"  test  ({len(split.test_scenarios)}): {sorted(split.test_scenarios)}  <- RESERVED, not touched by this script\n")

    if not split.val_scenarios:
        print(
            "ABORTING: val split is empty -- same degenerate case P6's own tests "
            "document for too-few scenarios. Not enough scenarios for a held-out "
            "number yet. (Expected to not happen with 12 scenarios at the default "
            "fracs, but checking rather than assuming.)"
        )
        return

    if args.hard_negative_weight is not None:
        train_result = build_split_batch_with_hard_negative_weights(
            split.train_scenarios,
            args.arm,
            args.different_vehicle_weight,
            args.hard_negative_weight,
            args.hard_dist_norm_threshold,
            args.hard_size_gap_threshold,
            violation_lookup=violation_lookup,
            other_weight=args.other_weight,
        )
    else:
        train_result = build_split_batch_with_pair_weights(
            split.train_scenarios, args.arm, args.different_vehicle_weight,
            violation_lookup=violation_lookup, other_weight=args.other_weight,
        )
    # Tue 7 Oct 2026 fix: build val with the SAME per-pair different-vehicle
    # marker build_samples_for_scenario_with_pair_weights() already produces
    # for TRAIN (sentinel weight -1.0, never used arithmetically -- just a
    # marker here, val is always evaluated unweighted). Same scenario order,
    # same crop_lookup construction as build_split_batch() would have used,
    # so val_batch's samples/pair_index/labels are IDENTICAL to what
    # build_split_batch() alone would have returned; this only adds a
    # parallel marker array aligned with it, nothing else changes.
    val_result = build_split_batch_with_pair_weights(
        split.val_scenarios, args.arm, -1.0, violation_lookup=violation_lookup
    )

    if train_result is None:
        print("ABORTING: zero usable training samples in the train split. See dataset.py's GAP #2.")
        return
    if val_result is None:
        print("ABORTING: zero usable samples in the val split (scenarios present but none collated). "
              "Cannot report a held-out number.")
        return

    train_samples, train_batch, train_pair_weights = train_result
    val_samples, val_batch, val_pair_markers = val_result
    if args.hard_negative_weight is not None:
        n_hard = int((train_pair_weights == args.hard_negative_weight).sum())
        n_flat = int((train_pair_weights == args.different_vehicle_weight).sum())
        n_other = int((train_pair_weights == args.other_weight).sum())
        print(f"Hard-negative-mining loss: {n_hard}/{train_pair_weights.shape[0]} TRAIN pairs "
              f"(far + size-mismatched different-vehicle negatives) upweighted by "
              f"{args.hard_negative_weight}x; {n_flat} other different-vehicle negatives "
              f"upweighted by {args.different_vehicle_weight}x; {n_other} remaining pairs "
              f"at other-bucket weight {args.other_weight}x.\n")
    elif args.different_vehicle_weight != 1.0 or args.other_weight != 1.0:
        n_diff = int((train_pair_weights == args.different_vehicle_weight).sum())
        n_other = int((train_pair_weights == args.other_weight).sum())
        print(f"Pair-type-aware loss: {n_diff}/{train_pair_weights.shape[0]} TRAIN pairs "
              f"(different-vehicle negatives) upweighted by {args.different_vehicle_weight}x; "
              f"{n_other} remaining pairs at other-bucket weight {args.other_weight}x.\n")
    train_crops, train_frag_aux, train_pair_index, train_pair_aux, train_labels = train_batch
    val_crops, val_frag_aux, val_pair_index, val_pair_aux, val_labels = val_batch

    for t in (train_crops, train_frag_aux, train_pair_index, train_pair_aux, train_labels):
        t.data = t.data.to(device)
    for t in (val_crops, val_frag_aux, val_pair_index, val_pair_aux, val_labels):
        t.data = t.data.to(device)
    train_pair_weights = train_pair_weights.float().to(device)
    val_diff_vehicle_mask = (val_pair_markers != 1.0).bool().to(device)  # val_pair_markers is already a torch tensor
    print(f"VAL different-vehicle bucket: {int(val_diff_vehicle_mask.sum())}/{val_diff_vehicle_mask.numel()} "
          "pairs -- this is the bucket best-checkpoint selection now tracks, not pooled val loss.\n")

    print(f"Train: {len(train_samples)} samples, {train_crops.shape[0]} fragments, "
          f"{train_pair_index.shape[0]} pairs, positive rate {float(train_labels.mean()):.4f}")
    print(f"Val (held-out): {len(val_samples)} samples, {val_crops.shape[0]} fragments, "
          f"{val_pair_index.shape[0]} pairs, positive rate {float(val_labels.mean()):.4f}\n")

    model = CorrectionHead().to(device)
    print(f"Model param count: {count_parameters(model):,}\n")

    # pos_weight from TRAIN labels only -- using val stats here would leak
    # val information into the training objective.
    n_pos = float(train_labels.sum())
    n_neg = float(train_labels.numel() - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    print(f"BCEWithLogitsLoss pos_weight={pos_weight.item():.2f} ({n_pos:.0f} pos / {n_neg:.0f} neg train pairs)\n")

    # reduction="none" so the pair-type weight (train_pair_weights, all-ones
    # unless --different-vehicle-weight != 1.0) can multiply each pair's own
    # loss before averaging -- pos_weight still handles the separate overall
    # positive/negative class imbalance as before, the two are independent.
    loss_fn_train = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    loss_fn_eval = torch.nn.BCEWithLogitsLoss()  # unweighted, for reporting only
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    history = []
    # Best-checkpoint tracking (Tue 7 Oct 2026 fix): training runs a FIXED
    # epoch count with no early stopping, and the user correctly suspected
    # the model "hadn't fully settled" -- it hadn't regressed either, val
    # loss/FPR oscillate (noisy at this val-set size: only 182
    # different-vehicle pairs) rather than monotonically improving, and the
    # LAST epoch is not guaranteed to be the BEST epoch. Previously this
    # function saved whatever state_dict the model happened to be in after
    # the final epoch; now it tracks the lowest unweighted VAL loss seen at
    # any logged epoch and saves THAT state_dict instead, with the final
    # epoch's numbers still printed/logged for comparison, not hidden.
    best_val_loss = float("inf")
    best_diff_vehicle_fpr = float("inf")
    best_epoch = None
    best_state = None
    evals_without_improve = 0
    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(train_crops, train_frag_aux, train_pair_index, train_pair_aux)
        per_pair_loss = loss_fn_train(logits, train_labels)
        loss = (per_pair_loss * train_pair_weights).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (logits.sigmoid() > 0.5).float()
            train_acc = float((preds == train_labels).float().mean())

        do_eval = epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs
        if do_eval:
            v = evaluate(model, val_batch, loss_fn_eval, device, diff_vehicle_mask=val_diff_vehicle_mask)
            # Tue 7 Oct 2026 fix: select the checkpoint by the DIFFERENT-
            # VEHICLE bucket's own FPR, not pooled val loss. Pooled loss is
            # ~96% background-background pairs here, so "best pooled loss"
            # does not mean "best on the bucket this whole investigation is
            # about" -- a real run picked a pooled-loss-best epoch that
            # scored 48.35% different-vehicle FPR, worse than doing nothing.
            # Wed 7 Oct 2026 fix, round 2: that alone has ITS OWN degenerate
            # case -- an epoch can get a great different-vehicle FPR just by
            # barely merging anything at all (a real run picked epoch 5,
            # recall 22.89%, see --min-recall-floor's own help text). Only
            # count an epoch as a candidate if recall clears the floor.
            clears_recall_floor = v['recall'] >= args.min_recall_floor
            is_best = clears_recall_floor and v['diff_vehicle_fpr'] < best_diff_vehicle_fpr
            if is_best:
                best_diff_vehicle_fpr = v['diff_vehicle_fpr']
                best_val_loss = v['loss']
                best_epoch = epoch
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            epoch_dt = time.perf_counter() - epoch_t0
            print(
                f"epoch {epoch:4d}/{args.epochs}  train_loss={loss.item():.4f} train_acc={train_acc:.4f}  |  "
                f"VAL loss={v['loss']:.4f} acc={v['acc']:.4f} CLEAN-rate={v['clean_rate']:.4f} "
                f"FPR={v['fpr']:.4f} recall={v['recall']:.4f} diffVehFPR={v['diff_vehicle_fpr']:.4f}  |  {epoch_dt:.2f}s"
                + ("  <- best so far" if is_best
                   else "  (below recall floor, not a candidate)" if not clears_recall_floor
                   else "")
            )
            history.append((epoch, float(loss.item()), train_acc, v['loss'], v['acc'], v['clean_rate'], v['fpr'], v['recall'], v['diff_vehicle_fpr'], epoch_dt))
            if not is_best:
                evals_without_improve += 1
            else:
                evals_without_improve = 0
            if args.early_stop_patience is not None and evals_without_improve >= args.early_stop_patience:
                print(f"\nEarly stop: {evals_without_improve} logged evals since the last "
                      f"best different-vehicle FPR (epoch {best_epoch}) >= --early-stop-patience "
                      f"{args.early_stop_patience}. Stopping at epoch {epoch}/{args.epochs}.")
                break
        else:
            epoch_dt = time.perf_counter() - epoch_t0
            history.append((epoch, float(loss.item()), train_acc, None, None, None, None, None, None, epoch_dt))

    elapsed = time.perf_counter() - t0
    epochs_run = epoch  # last epoch actually executed -- equals args.epochs unless early-stopped
    final_epoch_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is None:
        # No logged epoch ever cleared --min-recall-floor -- the fix for
        # one degenerate case (picking a never-merges epoch) must not
        # silently produce a different degenerate case (no checkpoint at
        # all, or a crash on model.load_state_dict(None)). Fall back to
        # the last epoch's weights and say so loudly rather than hiding it.
        print(f"\nWARNING: no logged epoch reached --min-recall-floor {args.min_recall_floor} "
              f"(best recall seen was still below it). Falling back to the LAST epoch's weights "
              f"({args.epochs}) -- this run's checkpoint has NOT been selected by different-vehicle "
              "FPR at all. Consider lowering --min-recall-floor or training longer.")
        best_state = final_epoch_state
        best_epoch = epochs_run
    print(f"\nBest different-vehicle FPR {best_diff_vehicle_fpr:.4f} (VAL loss {best_val_loss:.4f}) "
          f"at epoch {best_epoch}/{args.epochs} (final epoch was {args.epochs}) -- checkpoint below "
          "is the BEST-epoch weights BY THIS BUCKET, not necessarily the last-epoch weights and not "
          "necessarily the lowest-pooled-loss epoch either (see evaluate()'s own docstring, "
          "Tue 7 Oct 2026, for why pooled loss was dropped as the selection criterion).")
    model.load_state_dict(best_state)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_clean_rate", "val_fpr", "val_recall", "val_diff_vehicle_fpr", "epoch_seconds"])
        writer.writerows(history)

    v = evaluate(model, val_batch, loss_fn_eval, device, diff_vehicle_mask=val_diff_vehicle_mask)  # model now holds BEST-epoch weights
    final_val_majority = 1.0 - v["pos_rate"]

    # For transparency only: what the OLD behaviour (save last-epoch weights
    # unconditionally) would have reported, so a reader can see the gap
    # between best-epoch and last-epoch rather than just trusting this is
    # an improvement.
    model.load_state_dict(final_epoch_state)
    v_final_epoch = evaluate(model, val_batch, loss_fn_eval, device, diff_vehicle_mask=val_diff_vehicle_mask)
    model.load_state_dict(best_state)  # restore best for the real report/checkpoint below

    print(f"\n{epochs_run} epochs in {elapsed:.1f}s ({elapsed/epochs_run*1000:.1f}ms/epoch)."
          + (f" (--epochs was {args.epochs}; stopped early.)" if epochs_run != args.epochs else ""))
    print(f"Best epoch: {best_epoch} (pooled VAL loss {best_val_loss:.4f}, pooled FPR {v['fpr']:.4f}, "
          f"different-vehicle FPR {v['diff_vehicle_fpr']:.4f}).  "
          f"Last epoch {args.epochs} would have given different-vehicle FPR {v_final_epoch['diff_vehicle_fpr']:.4f} "
          f"(pooled FPR {v_final_epoch['fpr']:.4f}) -- checkpoint saved below uses the BEST epoch "
          "by different-vehicle FPR.")
    print(f"\n=== HELD-OUT (val) RESULT, seed={args.seed}, epoch={best_epoch} (best) ===")
    print(f"Val scenarios: {sorted(split.val_scenarios)}")
    print(f"Val pairs: {v['n_pairs']}  positive rate: {v['pos_rate']:.4f}  "
          f"(TP={v['tp']:.0f} FP={v['fp']:.0f} TN={v['tn']:.0f} FN={v['fn']:.0f})")
    print(f"Val loss (unweighted BCE): {v['loss']:.4f}")
    print(f"Val accuracy: {v['acc']:.4f}  (majority-class baseline: {final_val_majority:.4f})")
    print(f"Val CLEAN-rate: {v['clean_rate']:.4f}   Val FPR: {v['fpr']:.4f}")
    print(f"Val recall (catches actual violations): {v['recall']:.4f}   precision: {v['precision']:.4f}")
    beat_baseline = v['acc'] > final_val_majority
    print(f"{'BEATS' if beat_baseline else 'DOES NOT BEAT'} the majority-class baseline on raw accuracy "
          "(expected to be a weak/misleading signal at this positive rate -- CLEAN-rate/FPR/recall above "
          "are the numbers the Sat 3 Oct / Sun 4 Oct gate's done-when actually names).")

    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_diff_vehicle_fpr": best_diff_vehicle_fpr,
        "best_val_loss": best_val_loss,
        "val_diff_vehicle_fpr": v["diff_vehicle_fpr"],
        "final_epoch_val_loss": v_final_epoch["loss"],
        "final_epoch_val_fpr": v_final_epoch["fpr"],
        "final_epoch_diff_vehicle_fpr": v_final_epoch["diff_vehicle_fpr"],
        "different_vehicle_weight": args.different_vehicle_weight,
        "other_weight": args.other_weight,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_dist_norm_threshold": args.hard_dist_norm_threshold,
        "hard_size_gap_threshold": args.hard_size_gap_threshold,
        "train_scenarios": sorted(split.train_scenarios),
        "val_scenarios": sorted(split.val_scenarios),
        "test_scenarios_reserved": sorted(split.test_scenarios),
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "n_val_pairs": v["n_pairs"],
        "val_positive_rate": v["pos_rate"],
        "val_loss": v["loss"],
        "val_acc": v["acc"],
        "val_majority_baseline": final_val_majority,
        "val_clean_rate": v["clean_rate"],
        "val_fpr": v["fpr"],
        "val_recall": v["recall"],
        "val_precision": v["precision"],
        "val_tp": v["tp"], "val_fp": v["fp"], "val_tn": v["tn"], "val_fn": v["fn"],
        "beats_majority_baseline": beat_baseline,
        "elapsed_seconds": elapsed,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),  # BEST-epoch weights, see best_epoch below
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_diff_vehicle_fpr": best_diff_vehicle_fpr,
            "best_val_loss": best_val_loss,
            "val_scenarios": sorted(split.val_scenarios),
            "train_scenarios": sorted(split.train_scenarios),
            "arm": args.arm,
            "different_vehicle_weight": args.different_vehicle_weight,
            "other_weight": args.other_weight,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_dist_norm_threshold": args.hard_dist_norm_threshold,
        "hard_size_gap_threshold": args.hard_size_gap_threshold,
        },
        checkpoint_path,
    )
    print(f"\nPer-epoch log: {out_csv}")
    print(f"Held-out summary JSON: {args.out_json}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(
        "\nReminder: val scenarios are development numbers, re-run/tunable. The "
        "test scenarios above were never touched by this script -- reserved for "
        "the Week 6' results freeze (execution_plan_daily.md, Tue 13 Oct)."
    )


if __name__ == "__main__":
    main()
