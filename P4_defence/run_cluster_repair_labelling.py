"""
P4 -> P5 -- End-to-end cluster-repair labelling: for every qualifying
(scenario, frame, CAV), runs the real pipeline (RANSAC ground removal +
DBSCAN clustering, exactly cluster_failure_audit.py's remove_ground_plane_seeded
/ cluster_points) and P5_labelling's oracle-labelling functions
(label_point_vehicle_membership / build_cluster_repair_target /
build_candidate_fragments / fragment_majority_vehicle_id /
pairwise_same_object_affinity) to write the correction head's actual
training data to disk.

This is the "Next" item decisions.md flagged after committing the audit
script, the RANSAC seed fix, and the P5 oracle-labelling functions
(25 Sep 2026 entry) -- see task1_datasets_pipeline_dfl.md §2/§4 for the
spec this implements.

Ground-truth vehicle inclusion uses the EXACT SAME criteria as
cluster_failure_audit.py's audit_frame() (range <= MAX_AUDIT_RANGE_M,
raw_in_box >= MIN_RAW_POINTS_IN_BOX) -- deliberately, so the vehicles this
script treats as real, labellable objects are exactly the ones the
already-reported 996-instance CLEAN/MERGED/SPLIT/GROUND_SWALLOWED audit is
built from. Diffing this script's own raw_cluster_id against
repaired_cluster_id per (scenario, frame, cav) output reproduces that same
taxonomy as a by-product, not a separately re-implemented rule.

Per (scenario, frame, cav), writes three files under --out-dir:
  <scenario>/<frame>_<cav>_points.npz     -- per-point arrays: xyz (of the
      non-ground points only), raw_cluster_id, point_vehicle_id,
      repaired_cluster_id, fragment_id. This is the point-level supervision
      (repaired_cluster_id alone already fully captures MERGED/SPLIT/
      GROUND_SWALLOWED repair, including splitting a single MERGED raw
      cluster -- see labelling.py's module docstring on why fragment-level
      affinity alone can't express that).
  <scenario>/<frame>_<cav>_fragments.parquet -- one row per candidate
      fragment (fragment_id, centroid_x/y, n_points, vehicle_id,
      is_recovered_noise).
  <scenario>/<frame>_<cav>_pairs.parquet  -- one row per candidate fragment
      pair within CANDIDATE_PAIR_RADIUS_M of each other (fragment_i,
      fragment_j, centroid_distance_m, affinity_label) -- the correction
      head's actual training pairs (task1_datasets_pipeline_dfl.md §4).

Usage (from the MastersProj repo root; same deps as cluster_failure_audit.py
-- numpy, scikit-learn, shapely, open3d, pyyaml, pandas, NOT the full
opencood conda env, no torch needed):
    python P4_defence/run_cluster_repair_labelling.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data"

IMPORTANT -- do not commit output run against a partial local data mirror as
if it were the authoritative labelled set (same caution as
cluster_failure_audit.py's own vehicle_audit.json / frame_summary.json):
run this against the complete dataset on the real machine before treating
--out-dir's contents as the correction head's actual training data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# See P0_scenario_indexing/index.py's module comment: OPV2V's official
# train/validate/test splits dump numpy scalars into frame yaml, which
# yaml.safe_load() refuses -- reusing OpenCOOD's own loader (same fix).
from opencood.hypes_yaml.yaml_utils import load_yaml as _opencood_load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
sys.path.insert(0, str(REPO_ROOT / "P5_labelling"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402

from cluster_failure_audit import (  # noqa: E402
    remove_ground_plane_seeded,
    cluster_points,
    project_vehicle_to_lidar,
    points_in_rotated_box_xy,
    load_qualifying_scenarios,
    MAX_AUDIT_RANGE_M,
    MIN_RAW_POINTS_IN_BOX,
)
from consistency_layer import DBSCAN_EPS_M  # noqa: E402
from labelling import (  # noqa: E402
    label_point_vehicle_membership,
    build_cluster_repair_target,
    build_candidate_fragments,
    fragment_majority_vehicle_id,
    pairwise_same_object_affinity,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cache" / "repair_labels"

# Looser than the real pipeline's DBSCAN_MIN_SAMPLES (10) -- see
# labelling.build_candidate_fragments' docstring for why: this is what gives
# GROUND_SWALLOWED vehicles (entirely DBSCAN noise under the real threshold)
# a recoverable fragment identity at all.
NOISE_MIN_SAMPLES = 3

# Max centroid distance (metres) between two fragments to be scored as a
# candidate pair. Bigger than a single vehicle's own footprint -- so a SPLIT
# vehicle's two raw-cluster fragments, or a GROUND_SWALLOWED vehicle's
# recovered fragment sitting next to a MERGED neighbour, fall inside it --
# but small enough not to waste pairs on distant, unrelated fragments.
# Tunable; revisit once the correction head's own BEV-crop neighbourhood
# radius is pinned down (task1_datasets_pipeline_dfl.md's still-open
# architecture item).
CANDIDATE_PAIR_RADIUS_M = 4.0


import functools as _functools


@_functools.lru_cache(maxsize=None)
def resolve_scenario_dir(opv2v_root: str, scenario_id: str) -> Path:
    """
    OPV2V extracts sometimes put a scenario directly under opv2v_root (e.g.
    culver_data), sometimes one level deeper under a split subfolder
    (train/validate/test) -- mirrors P0's own _discover_scenarios()
    two-level search (P0_scenario_indexing/index.py) so a manifest
    scenario_id resolves correctly regardless of which OPV2V extract it
    came from. Found by running this script against the combined
    train/validate/test+culver_data root (Thu 2 Oct 2026) and getting 0
    audited instances -- this function previously assumed every scenario
    sat directly under opv2v_root, which broke once scenarios came from
    more than one subfolder.
    """
    root = Path(opv2v_root)
    direct = root / scenario_id
    if direct.is_dir():
        return direct
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidate = child / scenario_id
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(
        f"scenario {scenario_id!r} not found under {opv2v_root} "
        "(checked directly under opv2v_root and one level deeper)"
    )


def label_frame(opv2v_root: str, scenario_id: str, frame_id: str, cav_id: str):
    """
    Runs the real pipeline + full P5 oracle-labelling chain for one
    (scenario, frame, CAV). Returns None if the yaml/pcd files aren't
    present locally (same skip behaviour as cluster_failure_audit.audit_frame).
    Otherwise returns (points_out: dict of arrays, fragments_df, pairs_df).
    """
    scenario_dir = resolve_scenario_dir(opv2v_root, scenario_id)
    yaml_path = scenario_dir / cav_id / f"{frame_id}.yaml"
    pcd_path = scenario_dir / cav_id / f"{frame_id}.pcd"
    if not (yaml_path.exists() and pcd_path.exists()):
        return None

    meta = _opencood_load_yaml(str(yaml_path))
    lidar_pose = meta["lidar_pose"]
    vehicles = meta["vehicles"]

    raw = mask_ego_points(pcd_to_np(str(pcd_path)))
    non_ground = remove_ground_plane_seeded(raw)
    raw_cluster_id = cluster_points(non_ground)
    points_xy = non_ground[:, :2]

    # Same vehicle-inclusion criteria as cluster_failure_audit.audit_frame --
    # see module docstring for why that matters.
    gt_boxes_xy = {}
    for vid, content in vehicles.items():
        if vid == cav_id:
            continue
        corners = project_vehicle_to_lidar(content, lidar_pose)
        box_xy = corners[:4, :2]
        z_lo, z_hi = corners[:, 2].min(), corners[:, 2].max()
        r = float(np.linalg.norm(box_xy.mean(axis=0)))
        if r > MAX_AUDIT_RANGE_M:
            continue
        in_box_xy_raw = points_in_rotated_box_xy(raw[:, :2], box_xy)
        in_box_z_raw = (raw[:, 2] >= z_lo - 0.1) & (raw[:, 2] <= z_hi + 0.1)
        raw_in_box = int(np.sum(in_box_xy_raw & in_box_z_raw))
        if raw_in_box < MIN_RAW_POINTS_IN_BOX:
            continue
        gt_boxes_xy[vid] = box_xy

    point_vehicle_id = label_point_vehicle_membership(points_xy, gt_boxes_xy)
    repair_target = build_cluster_repair_target(raw_cluster_id, point_vehicle_id)
    fragment_id = build_candidate_fragments(
        points_xy, raw_cluster_id, noise_eps_m=DBSCAN_EPS_M, noise_min_samples=NOISE_MIN_SAMPLES
    )
    frag_vehicle = fragment_majority_vehicle_id(fragment_id, point_vehicle_id)
    frag_ids_sorted = sorted(frag_vehicle)

    if not frag_ids_sorted:
        fragments_df = pd.DataFrame(
            columns=["fragment_id", "centroid_x", "centroid_y", "n_points", "vehicle_id", "is_recovered_noise"]
        )
        pairs_df = pd.DataFrame(columns=["fragment_i", "fragment_j", "centroid_distance_m", "affinity_label"])
    else:
        raw_cluster_ids_present = set(raw_cluster_id[raw_cluster_id >= 0].tolist())
        centroids = {f: points_xy[fragment_id == f].mean(axis=0) for f in frag_ids_sorted}
        npoints = {f: int(np.sum(fragment_id == f)) for f in frag_ids_sorted}

        fragments_df = pd.DataFrame([
            {
                "fragment_id": int(f),
                "centroid_x": float(centroids[f][0]),
                "centroid_y": float(centroids[f][1]),
                "n_points": npoints[f],
                "vehicle_id": int(frag_vehicle[f]),
                "is_recovered_noise": f not in raw_cluster_ids_present,
            }
            for f in frag_ids_sorted
        ])

        vehicle_arr = np.array([frag_vehicle[f] for f in frag_ids_sorted])
        centroid_arr = np.array([centroids[f] for f in frag_ids_sorted])

        pair_a, pair_b, pair_dist = [], [], []
        for a in range(len(frag_ids_sorted)):
            for b in range(a + 1, len(frag_ids_sorted)):
                d = float(np.linalg.norm(centroid_arr[a] - centroid_arr[b]))
                if d <= CANDIDATE_PAIR_RADIUS_M:
                    pair_a.append(a)
                    pair_b.append(b)
                    pair_dist.append(d)

        if pair_a:
            affinity = pairwise_same_object_affinity(vehicle_arr, np.array(pair_a), np.array(pair_b))
        else:
            affinity = np.array([], dtype=bool)

        pairs_df = pd.DataFrame({
            "fragment_i": [frag_ids_sorted[a] for a in pair_a],
            "fragment_j": [frag_ids_sorted[b] for b in pair_b],
            "centroid_distance_m": pair_dist,
            "affinity_label": list(affinity),
        })

    points_out = {
        "xyz": non_ground,
        "raw_cluster_id": raw_cluster_id,
        "point_vehicle_id": point_vehicle_id,
        "repaired_cluster_id": repair_target.repaired_cluster_id,
        "fragment_id": fragment_id,
    }
    return points_out, fragments_df, pairs_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_qualifying_scenarios()
    print(f"Labelling {len(scenarios)} qualifying scenario(s): {sorted(scenarios)}")

    n_frames = 0
    n_fragments = 0
    n_recovered_noise_fragments = 0
    n_pairs = 0
    n_pairs_affinity_true = 0

    for scenario_id, (cavs, frames) in scenarios.items():
        scenario_out = out_dir / scenario_id
        scenario_out.mkdir(parents=True, exist_ok=True)
        for frame_id in frames:
            for cav_id in cavs:
                out = label_frame(args.opv2v_root, scenario_id, frame_id, cav_id)
                if out is None:
                    continue
                points_out, fragments_df, pairs_df = out

                stem = f"{frame_id}_{cav_id}"
                np.savez_compressed(scenario_out / f"{stem}_points.npz", **points_out)
                fragments_df.to_parquet(scenario_out / f"{stem}_fragments.parquet")
                pairs_df.to_parquet(scenario_out / f"{stem}_pairs.parquet")

                n_frames += 1
                n_fragments += len(fragments_df)
                if len(fragments_df):
                    n_recovered_noise_fragments += int(fragments_df["is_recovered_noise"].sum())
                n_pairs += len(pairs_df)
                if len(pairs_df):
                    n_pairs_affinity_true += int(pairs_df["affinity_label"].sum())

    print(f"\nLabelled {n_frames} (scenario, frame, cav) rows -> {out_dir}")
    print(f"Total candidate fragments: {n_fragments}"
          f" ({n_recovered_noise_fragments} recovered from noise, "
          f"{100 * n_recovered_noise_fragments / n_fragments:.1f}%)" if n_fragments else "")
    print(f"Total candidate pairs: {n_pairs}"
          f" ({n_pairs_affinity_true} same-object, "
          f"{100 * n_pairs_affinity_true / n_pairs:.1f}%)" if n_pairs else "")


if __name__ == "__main__":
    main()
