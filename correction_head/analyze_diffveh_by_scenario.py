"""Per-SCENARIO different-vehicle FPR breakdown -- Fri 9 Oct 2026.

evaluate_pair_types.py pools all held-out scenarios into one different-
vehicle bucket and reports one FPR number. After the 34-scenario data
expansion, that pooled number (15.53%) regressed vs the old 12-scenario
best (7.69%/7.14%), but the comparison isn't apples-to-apples -- the held-
out scenarios are completely different this time (7 new val scenarios vs
the old 2). This script reuses evaluate_pair_types.py's exact same loop
and bucket logic, but tags each pair with its scenario_id before pooling,
so you can see whether the regression is a BROAD effect (every scenario
a bit worse) or CONCENTRATED (one or two bad scenarios dragging the
pooled number down).

USAGE (same args as evaluate_pair_types.py, run from repo root, opencood
conda env):
    python -m correction_head.analyze_diffveh_by_scenario \
        --checkpoint correction_head/checkpoints/v5_expanded34scen_model.pt \
        --violation-lookup P4_defence/cache/violation_scalars.csv
"""

from __future__ import annotations

import argparse
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
from correction_head.evaluate_pair_types import (
    BUCKETS,
    classify_pair,
    confusion,
    fragment_vehicle_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_model.pt"))
    parser.add_argument("--scenarios", type=str, nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument("--crops-dir", type=str, default=str(DEFAULT_CROPS_DIR))
    parser.add_argument("--repair-labels-dir", type=str, default=str(DEFAULT_REPAIR_LABELS_DIR))
    parser.add_argument("--violation-lookup", type=str, default=None)
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

    scenarios = args.scenarios if args.scenarios else sorted(val_scenarios)
    degradation_config = load_degradation_config()
    repair_labels_dir = Path(args.repair_labels_dir)

    violation_lookup = None
    if args.violation_lookup:
        from correction_head.dataset import load_violation_lookup
        violation_lookup = load_violation_lookup(args.violation_lookup)
        if violation_lookup is None:
            print(f"WARNING: --violation-lookup {args.violation_lookup} not found -- zero-filling.")
        else:
            print(f"Loaded {len(violation_lookup)} (scenario, frame) violation rows from {args.violation_lookup}")

    all_labels, all_head_scores, all_buckets, all_scenario_tags = [], [], [], []
    mixed_count = 0

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

            for (i, j), lbl, hs in zip(pair_index, labels, head_scores):
                frag_i = fragment_ids_in_order[i]
                frag_j = fragment_ids_in_order[j]
                bucket = classify_pair(vid_of_frag[frag_i], vid_of_frag[frag_j])
                if bucket == "mixed/ambiguous":
                    mixed_count += 1
                    continue
                all_labels.append(lbl)
                all_head_scores.append(hs)
                all_buckets.append(bucket)
                all_scenario_tags.append(scenario_id)

    all_labels = np.array(all_labels, dtype=np.float32)
    all_head_scores = np.array(all_head_scores, dtype=np.float32)
    all_buckets = np.array(all_buckets)
    all_scenario_tags = np.array(all_scenario_tags)
    head_preds = (all_head_scores > args.threshold).astype(np.float32)

    print(f"\n{mixed_count} mixed/ambiguous pairs excluded. "
          f"Per-scenario different-vehicle breakdown ({len(scenarios)} scenarios):\n")

    rows = []
    for scenario_id in scenarios:
        smask = (all_scenario_tags == scenario_id) & (all_buckets == "different-vehicle")
        n = int(smask.sum())
        if n == 0:
            rows.append((scenario_id, 0, float("nan")))
            continue
        m = confusion(head_preds[smask], all_labels[smask])
        rows.append((scenario_id, n, m["fpr"]))

    rows.sort(key=lambda r: (-(r[2] if r[2] == r[2] else -1)))  # NaNs last implicitly via -1 sort trick reversed below
    rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1), reverse=True)

    print(f"{'scenario_id':<22} {'n pairs':>8} {'FPR (head)':>12}")
    print("-" * 44)
    total_fp = 0
    total_n = 0
    for scenario_id, n, fpr in rows:
        if n == 0:
            print(f"{scenario_id:<22} {n:>8} {'(no diff-veh pairs)':>12}")
            continue
        print(f"{scenario_id:<22} {n:>8} {fpr*100:>11.2f}%")
        total_n += n
    print("-" * 44)
    print(f"Pooled: {total_n} different-vehicle pairs across {len([r for r in rows if r[1] > 0])} scenarios with any.")


if __name__ == "__main__":
    main()
