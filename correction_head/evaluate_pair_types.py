"""Per-pair-type FPR/recall breakdown -- Mon 5 Oct 2026.

The headline held-out CLEAN-rate/FPR/recall numbers (train_v1.py,
denoise_baseline.py) are computed over ALL held-out pairs pooled together.
Most of those pairs are background-background (two fragments, neither
touching a real vehicle) -- this dominates the pool, since most of what
DBSCAN fragments isn't vehicles at all (see the before/after galleries).
A method that is excellent at background-background pairs and weak at
telling two DIFFERENT real vehicles apart can still show a great pooled
FPR, because the easy majority class swamps the number. The aggregate
false-merge count (evaluate_declustering.py) already caught this in
practice: the head makes FAR more vehicle-crossing false merges than the
denoise baseline (47 vs 6 across 43 held-out units) despite having a much
better pooled FPR (0.99% vs 48.5%).

This script splits every held-out PAIR (not just realized merges) into
four buckets using ground truth point_vehicle_id (no guessing -- this is
the same oracle label the project's whole ground truth is built on):
  - background-background: neither fragment touches a real vehicle
  - background-vehicle: exactly one fragment touches a real vehicle
  - same-vehicle: both fragments touch the SAME real vehicle (should merge)
  - different-vehicle: both fragments touch DIFFERENT real vehicles
    (should NEVER merge -- this is where a false merge comes from)
  - mixed/ambiguous: a fragment whose points touch >1 real vehicle (rare;
    reported separately, excluded from the four buckets above so it
    doesn't contaminate a clean category)
A fragment's "vehicle identity" is read directly from its points' own
point_vehicle_id (majority of non-background points, if any) -- not
inferred from distance or appearance, so this split cannot be circular
with what either method is being scored on.

Then reports CLEAN-rate/FPR/recall/precision SEPARATELY within each
bucket, for both the correction head and the denoise baseline, on the
SAME pairs. The number that matters most is each method's FPR on
different-vehicle pairs -- how often it wrongly says "merge" when the two
fragments are actually two different real cars.

USAGE
-----
    python -m correction_head.evaluate_pair_types \
        --checkpoint correction_head/runs/v1_holdout_model.pt \
        --out-json correction_head/runs/pair_type_breakdown.json

Run from the repo root, opencood conda env (needs torch).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from correction_head.dataset import (
    DEFAULT_CROPS_DIR,
    DEFAULT_REPAIR_LABELS_DIR,
    discover_frame_cav_pairs,
    load_degradation_config,
    load_scenario_crops,
)
from correction_head.model import CorrectionHead
from correction_head.visualize_corrections import BACKGROUND_VEHICLE_ID, build_sample_with_fragment_ids

BUCKETS = ["background-background", "background-vehicle", "same-vehicle", "different-vehicle"]


def fragment_vehicle_ids(fragment_ids_in_order, point_fragment_id, point_vehicle_id):
    """{fragment_id: None (background) | int vehicle_id | "mixed"} using
    ONLY the real ground-truth point_vehicle_id -- majority non-background
    vehicle id among that fragment's points if its points touch exactly
    one real vehicle; "mixed" if they touch more than one (rare, kept
    visible rather than silently assigned to either)."""
    out = {}
    for frag_id in fragment_ids_in_order.tolist():
        vids = point_vehicle_id[point_fragment_id == frag_id]
        real = vids[vids != BACKGROUND_VEHICLE_ID]
        if len(real) == 0:
            out[frag_id] = None
        else:
            distinct = set(real.tolist())
            out[frag_id] = distinct.pop() if len(distinct) == 1 else "mixed"
    return out


def classify_pair(vid_i, vid_j):
    if vid_i == "mixed" or vid_j == "mixed":
        return "mixed/ambiguous"
    if vid_i is None and vid_j is None:
        return "background-background"
    if vid_i is None or vid_j is None:
        return "background-vehicle"
    return "same-vehicle" if vid_i == vid_j else "different-vehicle"


def confusion(preds, labels):
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    clean_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return dict(n=int(len(labels)), tp=tp, fp=fp, tn=tn, fn=fn,
                clean_rate=clean_rate, fpr=fpr, recall=recall, precision=precision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_model.pt"))
    parser.add_argument("--scenarios", type=str, nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--denoise-summary", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "denoise_baseline_summary.json"))
    parser.add_argument("--denoise-threshold", type=float, default=None)
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument("--crops-dir", type=str, default=str(DEFAULT_CROPS_DIR))
    parser.add_argument("--repair-labels-dir", type=str, default=str(DEFAULT_REPAIR_LABELS_DIR))
    parser.add_argument("--out-json", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "pair_type_breakdown.json"))
    parser.add_argument(
        "--violation-lookup",
        type=str,
        default=None,
        help="Thu 8 Oct 2026 fix: path to the CSV compute_violation_scalars.py writes. "
             "Pass the SAME lookup the checkpoint was trained with (train_v1.py's own "
             "--violation-lookup), or free_occ_conflict_area/tracking_affinity will be "
             "zero-filled here regardless of what the model actually learned on -- this "
             "was a real bug (this script had no way to pass this at all, so today's "
             "occviol-checkpoint evals were run against the wrong frag_aux inputs) "
             "until today.",
    )
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CorrectionHead().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_scenarios = set(ckpt.get("val_scenarios", []))

    if args.denoise_threshold is not None:
        denoise_threshold = args.denoise_threshold
    else:
        p = Path(args.denoise_summary)
        denoise_threshold = json.loads(p.read_text())["threshold_dist_norm_lt"] if p.exists() else 0.68
    print(f"Denoise threshold: dist_norm < {denoise_threshold:.2f}")

    scenarios = args.scenarios if args.scenarios else sorted(val_scenarios)
    degradation_config = load_degradation_config()
    repair_labels_dir = Path(args.repair_labels_dir)

    violation_lookup = None
    if args.violation_lookup:
        from correction_head.dataset import load_violation_lookup
        violation_lookup = load_violation_lookup(args.violation_lookup)
        if violation_lookup is None:
            print(f"WARNING: --violation-lookup {args.violation_lookup} not found -- "
                  f"free_occ_conflict_area/tracking_affinity will be zero-filled.")
        else:
            print(f"Loaded {len(violation_lookup)} (scenario, frame) violation rows from {args.violation_lookup}")

    # accumulate raw pairs across ALL held-out units, tagged with their bucket
    all_labels, all_head_scores, all_denoise_dist, all_buckets = [], [], [], []
    mixed_count = 0
    n_units = 0

    for scenario_id in scenarios:
        try:
            crop_lookup, _ = load_scenario_crops(args.crops_dir, scenario_id, args.arm)
        except FileNotFoundError as e:
            print(f"Skipping {scenario_id}: {e}")
            continue
        for frame_id, cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
            built = build_sample_with_fragment_ids(
                scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, args.arm, degradation_config,
                violation_lookup=violation_lookup,
            )
            if built is None:
                continue
            crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order = built
            if pair_index.shape[0] == 0:
                continue
            points_path = repair_labels_dir / scenario_id / f"{frame_id}_{cav_id}_points.npz"
            if not points_path.exists():
                continue
            d = np.load(points_path)
            vid_of_frag = fragment_vehicle_ids(fragment_ids_in_order, d["fragment_id"], d["point_vehicle_id"])

            with torch.no_grad():
                t_crops = torch.from_numpy(crops).float().to(device)
                t_frag_aux = torch.from_numpy(frag_aux).float().to(device)
                t_pair_index = torch.from_numpy(pair_index).long().to(device)
                t_pair_aux = torch.from_numpy(pair_aux).float().to(device)
                head_scores = model(t_crops, t_frag_aux, t_pair_index, t_pair_aux).sigmoid().cpu().numpy()
            dist_norm = pair_aux[:, 0]

            for (i, j), lbl, hs, dn in zip(pair_index, labels, head_scores, dist_norm):
                frag_i = fragment_ids_in_order[i]
                frag_j = fragment_ids_in_order[j]
                bucket = classify_pair(vid_of_frag[frag_i], vid_of_frag[frag_j])
                if bucket == "mixed/ambiguous":
                    mixed_count += 1
                    continue
                all_labels.append(lbl)
                all_head_scores.append(hs)
                all_denoise_dist.append(dn)
                all_buckets.append(bucket)
            n_units += 1

    all_labels = np.array(all_labels, dtype=np.float32)
    all_head_scores = np.array(all_head_scores, dtype=np.float32)
    all_denoise_dist = np.array(all_denoise_dist, dtype=np.float32)
    all_buckets = np.array(all_buckets)

    head_preds = (all_head_scores > args.threshold).astype(np.float32)
    denoise_preds = (all_denoise_dist < denoise_threshold).astype(np.float32)

    print(f"\n{n_units} held-out units, {len(all_labels)} pairs classified "
          f"({mixed_count} mixed/ambiguous pairs excluded)\n")
    print(f"Pair counts by bucket: {dict(Counter(all_buckets.tolist()))}\n")

    results = {}
    header = f"{'Bucket':<22} {'n':>6}  {'method':<10} {'CLEAN-rate':>10} {'FPR':>8} {'recall':>8} {'precision':>10}"
    print(header)
    print("-" * len(header))
    for bucket in BUCKETS:
        mask = all_buckets == bucket
        n = int(mask.sum())
        if n == 0:
            print(f"{bucket:<22} {0:>6}  (no pairs in this bucket)")
            continue
        head_m = confusion(head_preds[mask], all_labels[mask])
        denoise_m = confusion(denoise_preds[mask], all_labels[mask])
        results[bucket] = {"head": head_m, "denoise": denoise_m}
        print(f"{bucket:<22} {n:>6}  {'head':<10} {head_m['clean_rate']*100:>9.2f}% {head_m['fpr']*100:>7.2f}% "
              f"{head_m['recall']*100:>7.2f}% {head_m['precision']*100:>9.2f}%")
        print(f"{'':<22} {'':>6}  {'denoise':<10} {denoise_m['clean_rate']*100:>9.2f}% {denoise_m['fpr']*100:>7.2f}% "
              f"{denoise_m['recall']*100:>7.2f}% {denoise_m['precision']*100:>9.2f}%")

    print("\nThe number that matters most: FPR on 'different-vehicle' pairs above -- how often each "
          "method wrongly merges two DIFFERENT real cars. Everything else is diluted by background pairs.")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_units": n_units, "n_pairs": int(len(all_labels)), "mixed_excluded": mixed_count,
                   "bucket_counts": dict(Counter(all_buckets.tolist())), "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
