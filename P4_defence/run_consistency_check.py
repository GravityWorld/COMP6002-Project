"""
P4 — Run the standalone consistency layer against one real qualifying scenario
from the P0 manifest.

Today's done-when checks (execution_plan_daily.md, Thu 3 / Fri 4 Sep):
  1. Plot clusters from one real point cloud -- eyeball check: do the clusters
     look like plausible object groupings on real data?
  2. conflicting_area() passes its synthetic test -- already verified via the
     unit tests in test_consistency_layer.py (run separately, no pytest
     needed: `python -m P4_defence.test_consistency_layer` won't work as a
     script since it has no __main__; run via pytest if installed, or ask
     for a runner script if not).

This script covers (1), plus a real (non-synthetic) conflicting-area run
between two actual CAVs in one frame, as a sanity check beyond the synthetic
unit test.

Usage (from the MastersProj repo root, opencood conda env active):
    python P4_defence/run_consistency_check.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data" ^
        --scenario 2021_09_03_20_08_37

Uses whichever frame_id/cav_ids the P0 manifest (P0_scenario_indexing/cache/
manifest.parquet) recorded for that scenario -- run P0's build_manifest()
first if that file doesn't exist yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; the PNG is the deliverable, not a live window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402

from consistency_layer import (  # noqa: E402
    remove_ground_plane,
    cluster_points,
    clusters_to_polygons,
    conflicting_area,
    CONFLICT_AREA_THRESHOLD_M2,
)

MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"
OUT_DIR = Path(__file__).resolve().parent / "cache"


def load_frame_and_cavs(scenario_id: str) -> tuple[str, list[str]]:
    """Pull one qualifying frame_id and its cav_ids for `scenario_id` from the P0 manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found -- run P0's build_manifest() first "
            "(see P0_scenario_indexing/index.py --opv2v-root ...)."
        )
    df = pd.read_parquet(MANIFEST_PATH)
    rows = df[df["scenario_id"] == scenario_id]
    if rows.empty:
        raise ValueError(
            f"'{scenario_id}' has no qualifying (n_cav >= 3) rows in the manifest. "
            f"Scenarios available: {sorted(df['scenario_id'].unique())[:10]}..."
        )
    row = rows.iloc[0]
    return row["frame_id"], list(row["cav_ids"])


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


def pcd_path(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> Path:
    # Standard OpenCOOD/OPV2V layout: {root}/{scenario}/{cav_id}/{frame_id}.pcd
    # (scenario may sit one level deeper under a split subfolder -- see resolve_scenario_dir)
    return resolve_scenario_dir(opv2v_root, scenario_id) / cav_id / f"{frame_id}.pcd"


def run_one_cav(points: np.ndarray, label: str):
    raw_n = len(points)
    points = mask_ego_points(points)  # drop the sensing vehicle's own body/roof returns
    # OPV2V LiDAR is 360 degrees; a handful of far-range returns (building tops,
    # sky reflections, sparse noise) can span hundreds of metres and blow out
    # any plot's autoscale even though the scene of interest is much tighter.
    # Report the raw radial-range spread so it's visible, not silently clipped.
    r = np.linalg.norm(points[:, :2], axis=1)
    print(
        f"[{label}] radial range: p50={np.percentile(r, 50):.1f}m, "
        f"p95={np.percentile(r, 95):.1f}m, p99.9={np.percentile(r, 99.9):.1f}m, "
        f"max={r.max():.1f}m"
    )

    non_ground = remove_ground_plane(points)
    frac_ground = 1 - len(non_ground) / len(points)
    labels = cluster_points(non_ground)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    print(
        f"[{label}] {raw_n} raw -> {len(points)} pts (post ego-mask) -> "
        f"{len(non_ground)} non-ground ({frac_ground:.0%} removed as ground) "
        f"-> {n_clusters} clusters, {n_noise} noise pts"
    )
    if frac_ground < 0.5:
        print(
            f"  !! only {frac_ground:.0%} removed as ground -- for a street scene this "
            "is low (expect ground to dominate raw LiDAR returns). RANSAC's single "
            "global-plane fit may be under-fitting a sloped/curved road surface -- "
            "worth checking against CAD's actual ground-removal approach."
        )
    if n_clusters > 30:
        print(
            f"  !! {n_clusters} clusters is a lot for one frame -- likely "
            "over-segmentation (DBSCAN_EPS_M may be too small, splitting single "
            "objects into fragments) rather than 84 real distinct objects. "
            "Check the plot's zoomed-in view before trusting this count."
        )
    polygons = clusters_to_polygons(non_ground, labels)
    return non_ground, labels, polygons


def plot_clusters(non_ground: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    xy = non_ground[:, :2]
    noise = labels == -1
    n_clusters = len(set(labels) - {-1})

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Left: full extent, so you can see any far-range outliers driving autoscale.
    ax = axes[0]
    ax.scatter(xy[noise, 0], xy[noise, 1], s=2, c="lightgray")
    ax.scatter(xy[~noise, 0], xy[~noise, 1], s=4, c=labels[~noise], cmap="tab20")
    ax.set_title(f"{title}\nfull extent")
    ax.set_xlabel("x (m, lidar frame)")
    ax.set_ylabel("y (m, lidar frame)")
    ax.axis("equal")

    # Right: zoomed to the 1st-99th percentile of non-ground points -- this is
    # the view that actually answers "do clusters look like plausible objects".
    if len(xy) > 0:
        lo, hi = np.percentile(xy, [1, 99], axis=0)
        pad = 0.1 * max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
        ax2 = axes[1]
        ax2.scatter(xy[noise, 0], xy[noise, 1], s=3, c="lightgray")
        ax2.scatter(xy[~noise, 0], xy[~noise, 1], s=8, c=labels[~noise], cmap="tab20")
        ax2.set_xlim(lo[0] - pad, hi[0] + pad)
        ax2.set_ylim(lo[1] - pad, hi[1] + pad)
        ax2.set_title(f"zoomed (1st-99th pct), {n_clusters} clusters")
        ax2.set_xlabel("x (m, lidar frame)")
        ax2.set_ylabel("y (m, lidar frame)")
        ax2.axis("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument(
        "--scenario",
        default="2021_09_03_20_08_37",
        help="Must be a confirmed-qualifying (n_cav >= 3) scenario_id present in the manifest",
    )
    args = parser.parse_args()

    frame_id, cav_ids = load_frame_and_cavs(args.scenario)
    print(f"Scenario {args.scenario}, frame {frame_id}, CAVs {cav_ids}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # First two CAVs are enough for today's check: one eyeballed cluster plot
    # per CAV, plus one pairwise conflicting-area number.
    cav_a, cav_b = cav_ids[0], cav_ids[1]
    polygons_by_cav = {}
    for cav_id in (cav_a, cav_b):
        pcd_file = pcd_path(args.opv2v_root, args.scenario, cav_id, frame_id)
        if not pcd_file.exists():
            raise FileNotFoundError(f"Expected point cloud not found: {pcd_file}")
        points = pcd_to_np(str(pcd_file))
        non_ground, labels, polygons = run_one_cav(points, f"CAV {cav_id}")
        plot_clusters(
            non_ground,
            labels,
            f"{args.scenario} / frame {frame_id} / CAV {cav_id}",
            OUT_DIR / f"clusters_{args.scenario}_{frame_id}_cav{cav_id}.png",
        )
        polygons_by_cav[cav_id] = polygons

    result = conflicting_area(polygons_by_cav[cav_a], polygons_by_cav[cav_b])
    print(f"\nConflicting area between CAV {cav_a} and CAV {cav_b} (RAW, UNALIGNED frames):")
    print(f"  total_conflicting_area_m2 = {result.total_conflicting_area_m2:.2f}")
    print(f"  conflict_polygon_count    = {result.conflict_polygon_count}")
    print(f"  largest_conflict_area_m2  = {result.largest_conflict_area_m2:.2f}")
    print(f"  flagged (> {CONFLICT_AREA_THRESHOLD_M2} m^2)     = {result.flagged}")
    print(
        "  NOTE: each CAV's points are still in its own local sensor frame -- "
        "P1 alignment (transform to a shared ego frame via GT pose) hasn't run "
        "yet. A huge number here is EXPECTED right now, not a defence failure -- "
        "it becomes a meaningful number only after P1. Don't compare it to the "
        "~2.7 m^2 threshold until then."
    )
    print(
        "\nDone-when check: open the two PNGs in P4_defence/cache/ -- look at the "
        "RIGHT (zoomed) panel specifically. Do the clusters look like plausible "
        "object groupings (vehicles, poles, curbs), not one giant blob or dozens "
        "of fragments of the same object? If yes, today's consistency-layer "
        "done-when is met. If clusters look fragmented, DBSCAN_EPS_M in "
        "consistency_layer.py (currently 0.75m) may need to go up."
    )


if __name__ == "__main__":
    main()
