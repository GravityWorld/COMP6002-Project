"""Aggregate false-merge evaluation, EVERY held-out unit -- Mon 5 Oct 2026.

One frame's picture (visualize_corrections.py's gallery) is illustrative,
not evidence -- it's a single data point, and one real example already
showed it can randomly favor the weaker method (the denoise baseline had
0 "false merges" on one specific frame purely because its 90+ merges
there happened to land on background-background pairs, not because it's
actually good -- see decisions.md). This script answers the question that
picture can't: scored across EVERY held-out (frame, cav) unit in BOTH val
scenarios -- not a 2-per-scenario sample -- does the pattern hold?

Reports, for both methods, on the IDENTICAL fragments/pairs used for the
official held-out CLEAN-rate/FPR/recall run (train_v1.py / denoise_
baseline.py), a different and complementary metric: how many predicted
groups provably contain >=2 DIFFERENT real vehicles (ground truth
point_vehicle_id) -- a false merge in the real, end-to-end declustering
sense, counted after the union-find post-process, not a wrong pairwise
label in isolation. This is closer to "how many real mistakes would this
make if deployed" than the pairwise metric is, at the cost of being a
coarser number (one false merge can come from many wrong pairs at once,
or vice versa) -- it belongs ALONGSIDE the pairwise CLEAN-rate/FPR/recall
already in decisions.md, not in place of it. Neither number alone is the
whole story.

USAGE
-----
    python -m correction_head.evaluate_declustering \
        --checkpoint correction_head/runs/v1_holdout_model.pt \
        --out-json correction_head/runs/false_merge_summary.json

Run from the repo root, opencood conda env (needs torch -- not runnable
in the cloud sandbox or the device-bridge VM, same as every other
correction_head/*.py script).
"""

from __future__ import annotations

import argparse
import json
import sys
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
# Reuses the exact same fragment-building / union-find / false-merge logic
# the gallery script uses -- deliberately NOT reimplemented here, so the
# two tools can never silently drift apart on what counts as a false merge.
from correction_head.visualize_corrections import (
    BACKGROUND_VEHICLE_ID,
    UnionFind,
    build_sample_with_fragment_ids,
    false_merge_group_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_model.pt"))
    parser.add_argument("--scenarios", type=str, nargs="*", default=None,
                         help="Defaults to the checkpoint's own held-out val scenarios -- the honest choice.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Head sigmoid merge threshold.")
    parser.add_argument("--denoise-summary", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "denoise_baseline_summary.json"))
    parser.add_argument("--denoise-threshold", type=float, default=None)
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument("--crops-dir", type=str, default=str(DEFAULT_CROPS_DIR))
    parser.add_argument("--repair-labels-dir", type=str, default=str(DEFAULT_REPAIR_LABELS_DIR))
    parser.add_argument("--out-json", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "false_merge_summary.json"))
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run `python -m correction_head.train_v1` first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CorrectionHead().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_scenarios = set(ckpt.get("val_scenarios", []))
    print(f"Loaded checkpoint: seed={ckpt.get('seed')} epochs={ckpt.get('epochs')}, held out {sorted(val_scenarios)}")

    if args.denoise_threshold is not None:
        denoise_threshold = args.denoise_threshold
    else:
        p = Path(args.denoise_summary)
        if not p.exists():
            print(f"No denoise summary at {p}, falling back to the logged threshold 0.68.")
            denoise_threshold = 0.68
        else:
            denoise_threshold = json.loads(p.read_text())["threshold_dist_norm_lt"]
    print(f"Denoise baseline threshold: dist_norm < {denoise_threshold:.2f}\n")

    scenarios = args.scenarios if args.scenarios else sorted(val_scenarios)
    if not scenarios:
        print("No scenarios given and checkpoint has no recorded val_scenarios -- pass --scenarios explicitly.")
        return

    degradation_config = load_degradation_config()
    repair_labels_dir = Path(args.repair_labels_dir)

    totals = dict(
        n_units=0, n_units_skipped=0,
        total_raw_fragments=0, total_scored_fragments=0, total_unscored_fragments=0,
        head_total_groups=0, denoise_total_groups=0,
        head_total_false_merges=0, denoise_total_false_merges=0,
        head_units_with_false_merge=0, denoise_units_with_false_merge=0,
    )
    per_unit_rows = []

    for scenario_id in scenarios:
        try:
            crop_lookup, _ = load_scenario_crops(args.crops_dir, scenario_id, args.arm)
        except FileNotFoundError as e:
            print(f"Skipping {scenario_id}: {e}")
            continue

        units = discover_frame_cav_pairs(repair_labels_dir, scenario_id)  # NO cap -- every unit
        for frame_id, cav_id in units:
            built = build_sample_with_fragment_ids(
                scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, args.arm, degradation_config
            )
            if built is None:
                totals["n_units_skipped"] += 1
                continue
            crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order = built

            points_path = repair_labels_dir / scenario_id / f"{frame_id}_{cav_id}_points.npz"
            if not points_path.exists():
                totals["n_units_skipped"] += 1
                continue
            d = np.load(points_path)
            point_fragment_id = d["fragment_id"]
            point_vehicle_id = d["point_vehicle_id"]

            n_frag = crops.shape[0]
            uf_head = UnionFind(n_frag)
            uf_denoise = UnionFind(n_frag)
            if pair_index.shape[0] > 0:
                with torch.no_grad():
                    t_crops = torch.from_numpy(crops).float().to(device)
                    t_frag_aux = torch.from_numpy(frag_aux).float().to(device)
                    t_pair_index = torch.from_numpy(pair_index).long().to(device)
                    t_pair_aux = torch.from_numpy(pair_aux).float().to(device)
                    head_scores = model(t_crops, t_frag_aux, t_pair_index, t_pair_aux).sigmoid().cpu().numpy()
                dist_norm = pair_aux[:, 0]
                for (i, j), hs, dn in zip(pair_index, head_scores, dist_norm):
                    if hs > args.threshold:
                        uf_head.union(int(i), int(j))
                    if dn < denoise_threshold:
                        uf_denoise.union(int(i), int(j))

            head_group_of_frag = dict(zip(fragment_ids_in_order.tolist(), uf_head.groups().tolist()))
            denoise_group_of_frag = dict(zip(fragment_ids_in_order.tolist(), uf_denoise.groups().tolist()))

            def frag_groups_to_points(group_of_frag):
                g = np.full(len(point_fragment_id), -1, dtype=np.int64)
                for frag_id, grp in group_of_frag.items():
                    g[point_fragment_id == frag_id] = grp
                return g

            head_point_group = frag_groups_to_points(head_group_of_frag)
            denoise_point_group = frag_groups_to_points(denoise_group_of_frag)

            head_false = false_merge_group_ids(head_point_group, point_vehicle_id)
            denoise_false = false_merge_group_ids(denoise_point_group, point_vehicle_id)

            n_scored = n_frag
            n_raw = len(set(point_fragment_id[point_fragment_id >= 0].tolist()))
            n_head_groups = len(set(g for g in head_group_of_frag.values()))
            n_denoise_groups = len(set(g for g in denoise_group_of_frag.values()))

            totals["n_units"] += 1
            totals["total_raw_fragments"] += n_raw
            totals["total_scored_fragments"] += n_scored
            totals["total_unscored_fragments"] += (n_raw - n_scored)
            totals["head_total_groups"] += n_head_groups
            totals["denoise_total_groups"] += n_denoise_groups
            totals["head_total_false_merges"] += len(head_false)
            totals["denoise_total_false_merges"] += len(denoise_false)
            totals["head_units_with_false_merge"] += int(len(head_false) > 0)
            totals["denoise_units_with_false_merge"] += int(len(denoise_false) > 0)

            per_unit_rows.append(dict(
                scenario_id=scenario_id, frame_id=frame_id, cav_id=cav_id,
                n_raw_fragments=n_raw, n_scored_fragments=n_scored,
                head_groups=n_head_groups, denoise_groups=n_denoise_groups,
                head_false_merges=len(head_false), denoise_false_merges=len(denoise_false),
            ))
            print(f"{scenario_id} frame {frame_id} CAV {cav_id}: "
                  f"raw={n_raw} scored={n_scored} | "
                  f"head groups={n_head_groups} false={len(head_false)} | "
                  f"denoise groups={n_denoise_groups} false={len(denoise_false)}")

    print(f"\n=== AGGREGATE, {totals['n_units']} held-out units across {scenarios} "
          f"({totals['n_units_skipped']} skipped: no crops/points) ===")
    print(f"Total raw fragments: {totals['total_raw_fragments']}  "
          f"(scored: {totals['total_scored_fragments']}, unscored/no-crop: {totals['total_unscored_fragments']})")
    print(f"\n{'Metric':<45} {'Denoise baseline':>18} {'Correction head':>18}")
    print(f"{'Total merged groups produced':<45} {totals['denoise_total_groups']:>18} {totals['head_total_groups']:>18}")
    print(f"{'Total FALSE merges (span >=2 real vehicles)':<45} {totals['denoise_total_false_merges']:>18} {totals['head_total_false_merges']:>18}")
    print(f"{'Units with >=1 false merge':<45} "
          f"{totals['denoise_units_with_false_merge']:>14}/{totals['n_units']:<3} "
          f"{totals['head_units_with_false_merge']:>14}/{totals['n_units']:<3}")

    summary = dict(totals=totals, per_unit=per_unit_rows, scenarios=scenarios,
                   threshold=args.threshold, denoise_threshold=denoise_threshold,
                   checkpoint=str(ckpt_path))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nReminder: this is a DIFFERENT metric from the pairwise CLEAN-rate/FPR/recall already "
          "in decisions.md -- report both. A false merge here can come from one wrong pair or several, "
          "and a unit with 0 false merges can still have wrong pairs that didn't happen to cross a "
          "real vehicle boundary (e.g. two background fragments wrongly merged).")


if __name__ == "__main__":
    main()
