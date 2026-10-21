"""
Diagnostic for the compute_fp_gap.py anomaly (17 Sep 2026): arm B's mean
violation magnitude dropped to 0.0121 m^2 vs arm A's 0.7770 m^2 -- a ~60x
DECREASE under degradation, the opposite of what benign sensor noise should
do to an occupancy-consistency check (it should push things toward looking
MORE inconsistent, not less).

This isolates ONE degraded CAV in ONE frame and prints point counts,
cluster counts, and occupied/free polygon areas before vs after
degradation, so we can tell apart two explanations:
  (a) fog degradation is starving the clustering step -- the degraded CAV's
      point cloud thins out so much that it contributes almost nothing to
      either occupied or free space, so the consistency check has nothing
      left to contradict (a genuine, reportable failure mode: "data
      poverty makes the defence go blind, not paranoid").
  (b) an actual bug -- e.g. degradation applied to the wrong axis, or
      cav_id in the config not matching a real cav_id in this scenario's
      frames (in which case inject_fog/inject_compression would silently
      never run, and the "clean" and "degraded" describe() blocks below
      would be near-identical instead of collapsed).

Usage (from the MastersProj repo root, opencood conda env active):
    python P4_defence\\diagnose_degradation.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data" ^
        --scenario-id 2021_09_03_20_08_37 --frame-id 003464 --cav-id 1854 ^
        --deg-type fog --severity 1 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
sys.path.insert(0, str(REPO_ROOT / "P1_alignment"))
sys.path.insert(0, str(REPO_ROOT / "P2_degradation"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402

from consistency_layer import remove_ground_plane, cluster_points, clusters_to_polygons, get_free_space  # noqa: E402
from degrade import inject_fog, inject_compression  # noqa: E402


def describe(label: str, points) -> None:
    non_ground = remove_ground_plane(points)
    labels = cluster_points(non_ground)
    n_unique_labels = len(set(labels.tolist())) if len(labels) else 0
    occupied = clusters_to_polygons(non_ground, labels)
    occupied_area = sum(p.area for p in occupied)
    free = get_free_space(points)
    print(
        f"  [{label:8s}] n_points={len(points):5d}  n_points_non_ground={len(non_ground):5d}  "
        f"n_unique_cluster_labels={n_unique_labels:3d} (may include a -1 noise label, "
        f"depends on cluster_points' convention)  n_occupied_polys={len(occupied):2d}  "
        f"occupied_area_total={occupied_area:9.4f}m2  free_area={free.area:9.4f}m2"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--cav-id", required=True)
    parser.add_argument("--deg-type", default="fog", choices=["fog", "compression"])
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    pcd_path = Path(args.opv2v_root) / args.scenario_id / args.cav_id / f"{args.frame_id}.pcd"
    if not pcd_path.exists():
        raise FileNotFoundError(
            f"{pcd_path} does not exist -- this means --cav-id {args.cav_id!r} is not a real "
            f"CAV folder in this scenario. If that's the case here, that's exactly the bug: "
            f"the degradation config's cav_id doesn't match a real cav_id, so "
            f"apply_degradation_if_assigned() would have silently found no match for every "
            f"frame and never degraded anything -- which would NOT explain the observed drop "
            f"in violation magnitude, so something else would need explaining instead."
        )
    points = pcd_to_np(str(pcd_path))
    points = mask_ego_points(points)
    print(
        f"scenario={args.scenario_id} frame={args.frame_id} cav={args.cav_id} "
        f"deg={args.deg_type} severity={args.severity} seed={args.seed}"
    )
    describe("clean", points)

    if args.deg_type == "fog":
        degraded = inject_fog(points, args.severity, args.seed)
    else:
        degraded = points.copy()
        if degraded.shape[1] >= 4:
            degraded[:, 3] = inject_compression(degraded[:, 3], args.severity, args.seed)
    describe("degraded", degraded)

    print(
        "\nInterpretation: if n_points and occupied_area_total both collapse sharply from "
        "clean to degraded, that supports 'data starvation defeats the check by silencing it' "
        "-- a real, reportable failure mode, not a bug. If n_points/occupied_area are nearly "
        "identical between clean and degraded, the degradation likely isn't being applied to "
        "this cav_id at all (check the --cav-id against the degradation_config_arm_b.json key "
        "and the actual cav_id folder names under this scenario)."
    )


if __name__ == "__main__":
    main()
