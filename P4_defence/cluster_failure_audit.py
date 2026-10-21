"""
P4 -- Clustering-failure audit: quantify where the consistency layer's RANSAC
ground-removal + DBSCAN clustering (consistency_layer.py's remove_ground_plane
/ cluster_points, exact same params: RANSAC residual 0.2m, DBSCAN eps=0.75m,
min_samples=10) diverges from real OPV2V ground-truth vehicle boxes.

This is the audit behind the Thu 24 Sep 2026 decisions.md entry and the
clustering-failure design now written into task1_datasets_pipeline_dfl.md,
scope_and_timeline_oct19.md, and execution_plan_daily.md. It was originally
built and run in a cloud sandbox (no opencood conda env / GPU available
there) and is committed here unchanged in method, only adapted to the repo's
own conventions (argparse --opv2v-root, REPO_ROOT-relative imports/output,
matching run_consistency_check.py's style).

Deliberately reuses only the pure-numpy pieces of OpenCOOD's own utilities
(pcd_utils.pcd_to_np/mask_ego_points, transformation_utils.x_to_world/
x1_to_x2 -- none of these import torch) so this script runs in ANY Python
env with numpy/sklearn/shapely/open3d/pyyaml, no torch or GPU needed.
create_bbx() is the one exception kept duplicated below rather than imported
from opencood.utils.box_utils, because that module imports torch at the top
of the file just to define a handful of unrelated functions -- importing it
here would defeat the point of a torch-free audit script for a single
Py3-only helper.

Classification per (CAV, frame, GT-vehicle) instance, requiring >=5 raw
points inside the GT box and range <= --max-range (default 60m, since points
get very sparse well before OPV2V's ~120m LiDAR clip range -- this keeps the
audit to vehicles that plausibly should be well-sensed, rather than
confounding "clustering failure" with "too far to ever have enough points"):
  - CLEAN            : exactly one DBSCAN cluster, uniquely assigned to that vehicle
  - MERGED           : that vehicle's cluster is shared with >=1 other GT vehicle
  - SPLIT            : that vehicle's non-ground points span >=2 distinct clusters
  - GROUND_SWALLOWED : real, well-sensed vehicle, but ground-removal +
                        clustering leaves it with zero surviving cluster
                        (either too few non-ground points to ever cluster, or
                        enough points that still end up entirely as DBSCAN
                        noise -- both are recorded, see non_ground_in_box)

Usage (from the MastersProj repo root; needs numpy, scikit-learn, shapely,
open3d, pyyaml, pandas -- NOT the full opencood conda env, no torch needed):
    python P4_defence/cluster_failure_audit.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data"

Reads the qualifying scenario list from the P0 manifest
(P0_scenario_indexing/cache/manifest.parquet) rather than hardcoding scenario
IDs, so this stays correct if more qualifying scenarios are added later.
Writes vehicle_audit.json (per-instance rows) and frame_summary.json
(per-frame ground-removal/cluster-count stats) to --out-dir (default
P4_defence/cache/audit/), plus prints the same headline breakdown reported
in decisions.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# See P0_scenario_indexing/index.py's module comment: OPV2V's official
# train/validate/test splits dump numpy scalars into frame yaml, which
# yaml.safe_load() refuses -- reusing OpenCOOD's own loader (same fix).
from opencood.hypes_yaml.yaml_utils import load_yaml as _opencood_load_yaml
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402
from opencood.utils.transformation_utils import x_to_world, x1_to_x2  # noqa: E402

from consistency_layer import (  # noqa: E402
    RANSAC_RESIDUAL_THRESHOLD_M,
    DBSCAN_EPS_M,
    DBSCAN_MIN_SAMPLES,
)

MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cache" / "audit"

# Vehicles farther than this from the sensor are excluded from the audit --
# not a "clustering failure" if a vehicle is simply too far/sparse to ever be
# a fair test of clustering, as opposed to sensing.
MAX_AUDIT_RANGE_M = 60.0
MIN_RAW_POINTS_IN_BOX = 5  # need at least this many raw points to call it "sensed"

# Evenly-spaced sample of frames per qualifying scenario -- matches the
# original audit's coverage (6 frames x all CAVs x each qualifying scenario).
N_FRAMES_PER_SCENARIO = 6


def create_bbx(extent):
    """Duplicated from opencood.utils.box_utils.create_bbx (pure Py3/numpy,
    but that module imports torch at the top just for unrelated functions --
    see module docstring for why it's not imported here instead)."""
    return np.array([[extent[0], -extent[1], -extent[2]],
                      [extent[0], extent[1], -extent[2]],
                      [-extent[0], extent[1], -extent[2]],
                      [-extent[0], -extent[1], -extent[2]],
                      [extent[0], -extent[1], extent[2]],
                      [extent[0], extent[1], extent[2]],
                      [-extent[0], extent[1], extent[2]],
                      [-extent[0], -extent[1], extent[2]]])


def points_in_rotated_box_xy(p_xy, box_corner_xy):
    """box_corner_xy: 4 corners (bottom face) of the box footprint, in the
    winding order create_bbx() produces: 0,1,2,3 = bottom face."""
    edge1 = box_corner_xy[1] - box_corner_xy[0]
    edge2 = box_corner_xy[3] - box_corner_xy[0]
    p_rel = p_xy - box_corner_xy[0].reshape(1, -1)
    l1 = (p_rel @ edge1) / np.dot(edge1, edge1)
    l2 = (p_rel @ edge2) / np.dot(edge2, edge2)
    return (l1 >= 0) & (l1 <= 1) & (l2 >= 0) & (l2 <= 1)


def project_vehicle_to_lidar(object_content, lidar_pose):
    location = object_content["location"]
    rotation = object_content["angle"]
    center = object_content["center"]
    extent = object_content["extent"]
    object_pose = [location[0] + center[0], location[1] + center[1],
                   location[2] + center[2], rotation[0], rotation[1], rotation[2]]
    object2lidar = x1_to_x2(object_pose, lidar_pose)
    bbx = create_bbx(extent).T
    bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
    bbx_lidar = np.dot(object2lidar, bbx).T[:, :3]
    return bbx_lidar  # (8, 3), corners 0-3 = bottom face


def remove_ground_plane_seeded(points: np.ndarray):
    """
    Same RANSAC ground-removal as consistency_layer.py's remove_ground_plane,
    but with random_state=0 so THIS AUDIT's own numbers are reproducible.

    consistency_layer.py's real RANSACRegressor() call still has no
    random_state set as of this commit -- its ground-removal (and everything
    downstream: cluster membership, cluster IDs) is NOT deterministic across
    runs on identical input. Confirmed empirically while building this audit
    (re-running against the same frame/CAV produced different non_ground
    sets and different cluster ID numbers between runs). That is tracked as
    a separate, real bug (see decisions.md, Thu 24 Sep 2026 entry) to be
    fixed in consistency_layer.py itself, independent of this audit.
    """
    xy = points[:, :2]
    z = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=RANSAC_RESIDUAL_THRESHOLD_M, random_state=0)
    ransac.fit(xy, z)
    inlier_mask = ransac.inlier_mask_
    return points[~inlier_mask]


def cluster_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=int)
    xy = points[:, :2]
    return DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)


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


def audit_frame(opv2v_root: str, scenario_id: str, frame_id: str, cav_id: str):
    scenario_dir = resolve_scenario_dir(opv2v_root, scenario_id)
    yaml_path = scenario_dir / cav_id / f"{frame_id}.yaml"
    pcd_path = scenario_dir / cav_id / f"{frame_id}.pcd"
    if not (yaml_path.exists() and pcd_path.exists()):
        return None

    meta = _opencood_load_yaml(str(yaml_path))
    lidar_pose = meta["lidar_pose"]
    vehicles = meta["vehicles"]

    raw = pcd_to_np(str(pcd_path))
    raw = mask_ego_points(raw)
    non_ground = remove_ground_plane_seeded(raw)
    labels = cluster_points(non_ground)

    # Precompute per-vehicle GT box (lidar frame) + 2D footprint corners.
    gt_boxes = {}
    for vid, content in vehicles.items():
        if vid == cav_id:
            continue
        corners = project_vehicle_to_lidar(content, lidar_pose)
        center_xy = corners[:4, :2].mean(axis=0)
        r = float(np.linalg.norm(center_xy))
        if r > MAX_AUDIT_RANGE_M:
            continue
        gt_boxes[vid] = {"corners": corners, "range_m": r}

    results = []
    for vid, box in gt_boxes.items():
        box_xy = box["corners"][:4, :2]  # bottom-face 4 corners
        z_lo, z_hi = box["corners"][:, 2].min(), box["corners"][:, 2].max()

        in_box_xy_raw = points_in_rotated_box_xy(raw[:, :2], box_xy)
        in_box_z_raw = (raw[:, 2] >= z_lo - 0.1) & (raw[:, 2] <= z_hi + 0.1)
        raw_in_box = int(np.sum(in_box_xy_raw & in_box_z_raw))

        if raw_in_box < MIN_RAW_POINTS_IN_BOX:
            continue  # not meaningfully sensed at all -- exclude (occlusion/sparsity, not a clustering failure)

        in_box_xy_ng = points_in_rotated_box_xy(non_ground[:, :2], box_xy)
        in_box_z_ng = (non_ground[:, 2] >= z_lo - 0.1) & (non_ground[:, 2] <= z_hi + 0.1)
        ng_mask = in_box_xy_ng & in_box_z_ng
        ng_in_box = int(np.sum(ng_mask))
        labels_in_box = labels[ng_mask]
        cluster_ids_in_box = sorted(set(labels_in_box.tolist()) - {-1})

        if ng_in_box < MIN_RAW_POINTS_IN_BOX or len(cluster_ids_in_box) == 0:
            status = "GROUND_SWALLOWED"
        elif len(cluster_ids_in_box) >= 2:
            status = "SPLIT"
        else:
            status = "CLEAN_SO_FAR"  # merge status resolved in a second pass below

        results.append({
            "scenario": scenario_id, "frame": frame_id, "cav": cav_id,
            "vehicle_id": vid, "range_m": box["range_m"],
            "raw_in_box": raw_in_box, "non_ground_in_box": ng_in_box,
            "ground_removed_frac": 1 - ng_in_box / raw_in_box if raw_in_box else None,
            "cluster_ids": cluster_ids_in_box, "status": status,
        })

    # Second pass: MERGED detection -- does a cluster used by this vehicle's
    # box also cover >=1 point of a DIFFERENT vehicle's box?
    vid_to_clusters = {r["vehicle_id"]: set(r["cluster_ids"]) for r in results}
    for r in results:
        if r["status"] != "CLEAN_SO_FAR":
            continue
        my_clusters = vid_to_clusters[r["vehicle_id"]]
        shared = any(
            my_clusters & other_clusters
            for other_vid, other_clusters in vid_to_clusters.items()
            if other_vid != r["vehicle_id"]
        )
        r["status"] = "MERGED" if shared else "CLEAN"

    return results, {
        "n_raw": len(raw), "n_non_ground": len(non_ground),
        "frac_ground_removed": 1 - len(non_ground) / len(raw),
        "n_clusters": len(set(labels.tolist()) - {-1}),
        "n_noise": int(np.sum(labels == -1)),
    }


def load_qualifying_scenarios():
    """Every scenario_id with n_cav >= 3 in the P0 manifest, plus its CAV
    list, taken from the manifest's own row rather than hardcoded."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found -- run P0's build_manifest() first "
            "(see P0_scenario_indexing/index.py --opv2v-root ...)."
        )
    df = pd.read_parquet(MANIFEST_PATH)
    scenarios = {}
    for scenario_id, group in df.groupby("scenario_id"):
        row = group.iloc[0]
        cav_ids = list(row["cav_ids"])
        frame_ids = sorted(group["frame_id"].unique())
        if len(frame_ids) == 0:
            continue
        # Evenly-spaced sample of N_FRAMES_PER_SCENARIO frames across the scenario.
        idx = np.linspace(0, len(frame_ids) - 1, min(N_FRAMES_PER_SCENARIO, len(frame_ids)))
        sampled = [frame_ids[int(round(i))] for i in idx]
        scenarios[scenario_id] = (cav_ids, sorted(set(sampled)))
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_qualifying_scenarios()
    print(f"Auditing {len(scenarios)} qualifying scenario(s): {sorted(scenarios)}")

    all_results = []
    frame_summaries = []
    for scenario_id, (cavs, frames) in scenarios.items():
        for frame_id in frames:
            for cav_id in cavs:
                out = audit_frame(args.opv2v_root, scenario_id, frame_id, cav_id)
                if out is None:
                    continue
                results, summary = out
                summary.update({"scenario": scenario_id, "frame": frame_id, "cav": cav_id})
                frame_summaries.append(summary)
                all_results.extend(results)

    with open(out_dir / "vehicle_audit.json", "w") as f:
        json.dump(all_results, f, indent=1)
    with open(out_dir / "frame_summary.json", "w") as f:
        json.dump(frame_summaries, f, indent=1)

    n = len(all_results)
    status_counts = Counter(r["status"] for r in all_results)
    print(f"\nTotal audited (CAV, frame, GT-vehicle) instances: {n}")
    for k, v in status_counts.items():
        print(f"  {k}: {v} ({100*v/n:.1f}%)")

    print("\nRange stats by status (mean range_m):")
    for k in status_counts:
        rs = [r["range_m"] for r in all_results if r["status"] == k]
        print(f"  {k}: mean={np.mean(rs):.1f}m  median={np.median(rs):.1f}m  n={len(rs)}")

    print("\nGround-removed-fraction stats by status (of the box's raw points):")
    for k in status_counts:
        gs = [r["ground_removed_frac"] for r in all_results if r["status"] == k and r["ground_removed_frac"] is not None]
        if gs:
            print(f"  {k}: mean={np.mean(gs):.2f}  median={np.median(gs):.2f}")

    print(f"\nWrote {out_dir / 'vehicle_audit.json'} and {out_dir / 'frame_summary.json'}")


if __name__ == "__main__":
    main()
