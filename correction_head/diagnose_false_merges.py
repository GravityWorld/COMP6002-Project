"""Pin down the exact pair(s) behind each head false merge -- Mon 5 Oct 2026.

The gallery (visualize_corrections.py) shows the SHAPE of a false merge
but packs hundreds of fragments into one scatter plot -- you can see THAT
a red dashed hull spans two cars, not WHY the head fired on that specific
pair. This script finds, for each false-merge group, the exact pair (i, j)
that first joined two different real vehicles together (walking the union
operations in score order, same as the model/post-process actually does),
and prints the numbers the head actually had to work with for that pair:

  - dist_norm       : the ONLY explicit distance signal in pair_aux
                       (pair_aux is just [dist_norm, abs_delta_log_n] --
                       PAIR_AUX_DIM=2, see dataset.py). This is also
                       exactly what the denoise baseline thresholds on.
  - abs_delta_log_n  : the other explicit pair feature (point-count gap).
  - head_score       : the head's sigmoid output for this pair.
  - denoise verdict  : would dist_norm alone have merged this pair.
  - the two fragments' own vehicle ids (ground truth) and point counts.

If the culprit pairs have a LOW dist_norm (genuinely close together in
space), that's a hard, defensible case -- two real cars parked close.
If dist_norm is NOT small yet the head still merged, that means the head
is leaning on the CNN crop embedding (visual similarity) and overriding
its own distance input -- evidence for a minimum-distance guard fix
rather than (or alongside) a loss reweighting fix.

USAGE
-----
    python -m correction_head.diagnose_false_merges \
        --checkpoint correction_head/runs/v1_holdout_model.pt \
        --units 2021_08_22_07_52_02:000071:3513 2021_08_22_07_52_02:000071:3495

Run from the repo root, opencood conda env (needs torch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from correction_head.dataset import (
    DEFAULT_CROPS_DIR,
    DEFAULT_REPAIR_LABELS_DIR,
    load_degradation_config,
    load_scenario_crops,
)
from correction_head.model import CorrectionHead
from correction_head.visualize_corrections import (
    BACKGROUND_VEHICLE_ID,
    UnionFind,
    build_sample_with_fragment_ids,
)
from correction_head.evaluate_pair_types import fragment_vehicle_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_model.pt"))
    parser.add_argument("--units", type=str, nargs="+", required=True,
                         help="scenario_id:frame_id:cav_id triples -- e.g. from evaluate_declustering.py's "
                              "--out-json per_unit list, sorted by head_false_merges.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument("--crops-dir", type=str, default=str(DEFAULT_CROPS_DIR))
    parser.add_argument("--repair-labels-dir", type=str, default=str(DEFAULT_REPAIR_LABELS_DIR))
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
    print(f"Loaded checkpoint: seed={ckpt.get('seed')} epochs={ckpt.get('epochs')}\n")

    degradation_config = load_degradation_config()
    repair_labels_dir = Path(args.repair_labels_dir)
    crop_cache = {}

    for unit in args.units:
        parts = unit.split(":")
        if len(parts) != 3:
            print(f"Skipping malformed unit {unit!r}: expected scenario_id:frame_id:cav_id")
            continue
        scenario_id, frame_id, cav_id = parts

        if scenario_id not in crop_cache:
            try:
                crop_cache[scenario_id], _ = load_scenario_crops(args.crops_dir, scenario_id, args.arm)
            except FileNotFoundError as e:
                print(f"Skipping {scenario_id}: {e}")
                continue
        crop_lookup = crop_cache[scenario_id]

        built = build_sample_with_fragment_ids(
            scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, args.arm, degradation_config
        )
        if built is None:
            print(f"Skipping {scenario_id}/{frame_id}/{cav_id}: no usable fragments.")
            continue
        crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order = built

        points_path = repair_labels_dir / scenario_id / f"{frame_id}_{cav_id}_points.npz"
        if not points_path.exists():
            print(f"Skipping {scenario_id}/{frame_id}/{cav_id}: no {points_path.name}.")
            continue
        d = np.load(points_path)
        point_fragment_id = d["fragment_id"]
        point_vehicle_id = d["point_vehicle_id"]
        frag_vid = fragment_vehicle_ids(fragment_ids_in_order, point_fragment_id, point_vehicle_id)
        frag_npoints = {fid: int((point_fragment_id == fid).sum()) for fid in fragment_ids_in_order.tolist()}

        n_frag = crops.shape[0]
        if pair_index.shape[0] == 0:
            print(f"{scenario_id} frame {frame_id} CAV {cav_id}: no pairs, skipping.")
            continue

        with torch.no_grad():
            t_crops = torch.from_numpy(crops).float().to(device)
            t_frag_aux = torch.from_numpy(frag_aux).float().to(device)
            t_pair_index = torch.from_numpy(pair_index).long().to(device)
            t_pair_aux = torch.from_numpy(pair_aux).float().to(device)
            head_scores = model(t_crops, t_frag_aux, t_pair_index, t_pair_aux).sigmoid().cpu().numpy()
        dist_norm = pair_aux[:, 0]
        abs_delta_log_n = pair_aux[:, 1]

        # fragment_ids_in_order[i] is the real fragment id for row i of crops/frag_aux.
        frag_id_of_row = fragment_ids_in_order.tolist()

        # Walk unions in score-descending order (most-confident merges first,
        # same effective behavior as a threshold cut) tracking, group by
        # group, the set of distinct real vehicle ids it currently contains.
        # The first pair that brings a SECOND distinct vehicle id into a
        # group is the culprit for that group's false merge.
        order = np.argsort(-head_scores)
        uf = UnionFind(n_frag)
        group_vids = {i: set() for i in range(n_frag)}
        for i in range(n_frag):
            vid = frag_vid[frag_id_of_row[i]]
            if vid is not None and vid != "mixed":
                group_vids[i].add(vid)

        culprits = []
        for k in order:
            if head_scores[k] <= args.threshold:
                break
            i, j = int(pair_index[k, 0]), int(pair_index[k, 1])
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:
                continue
            merged_vids = group_vids[ri] | group_vids[rj]
            was_clean = len(group_vids[ri]) <= 1 and len(group_vids[rj]) <= 1
            becomes_dirty = len(merged_vids) >= 2
            if was_clean and becomes_dirty:
                culprits.append(dict(
                    row_i=i, row_j=j,
                    frag_i=frag_id_of_row[i], frag_j=frag_id_of_row[j],
                    vid_i=frag_vid[frag_id_of_row[i]], vid_j=frag_vid[frag_id_of_row[j]],
                    npts_i=frag_npoints[frag_id_of_row[i]], npts_j=frag_npoints[frag_id_of_row[j]],
                    head_score=float(head_scores[k]), dist_norm=float(dist_norm[k]),
                    abs_delta_log_n=float(abs_delta_log_n[k]),
                ))
            uf.union(i, j)
            new_root = uf.find(i)
            group_vids[new_root] = merged_vids

        print(f"=== {scenario_id} frame {frame_id} CAV {cav_id}: {len(culprits)} culprit pair(s) ===")
        if not culprits:
            print("  (no NEW different-vehicle merges found by this walk -- check --threshold matches the run)")
        for c in culprits:
            denoise_would_merge = "YES (denoise would ALSO wrongly merge this)" if c["dist_norm"] < 0.68 else \
                                   "no (denoise's distance rule would correctly reject this)"
            print(f"  fragment {c['frag_i']} (vehicle {c['vid_i']}, {c['npts_i']} pts)  <->  "
                  f"fragment {c['frag_j']} (vehicle {c['vid_j']}, {c['npts_j']} pts)")
            print(f"    head_score={c['head_score']:.3f}  dist_norm={c['dist_norm']:.3f}  "
                  f"abs_delta_log_n={c['abs_delta_log_n']:.3f}")
            print(f"    would denoise (dist_norm<0.68) also merge this pair? {denoise_would_merge}")
        print()


if __name__ == "__main__":
    main()
