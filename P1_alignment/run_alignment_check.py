"""
P1 — Run alignment against one real qualifying scenario from the P0 manifest.

Today's done-when (execution_plan_daily.md, Sat 5 Sep): aligned multi-CAV
point clouds render correctly overlaid for one scenario. Eyeball a plot --
misalignment is obvious visually and invisible in metrics.

This is the real-data counterpart to test_align.py's synthetic ground-truth
tests (identity pose, known translation, known yaw rotation -- all passing
as of today's session).

Usage (from the MastersProj repo root, opencood conda env active):
    python P1_alignment/run_alignment_check.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data" ^
        --scenario 2021_09_03_20_08_37
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# See P0_scenario_indexing/index.py's module comment: OPV2V's official
# train/validate/test splits dump numpy scalars into frame yaml, which
# yaml.safe_load() refuses -- reusing OpenCOOD's own loader (same fix).
from opencood.hypes_yaml.yaml_utils import load_yaml as _opencood_load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402

from align import align_to_ego  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"
OUT_DIR = Path(__file__).resolve().parent / "cache"


def load_frame_and_cavs(scenario_id: str) -> tuple[str, list[str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run P0's build_manifest() first.")
    df = pd.read_parquet(MANIFEST_PATH)
    rows = df[df["scenario_id"] == scenario_id]
    if rows.empty:
        raise ValueError(f"'{scenario_id}' has no qualifying rows in the manifest.")
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


def frame_yaml_path(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> Path:
    return resolve_scenario_dir(opv2v_root, scenario_id) / cav_id / f"{frame_id}.yaml"


def frame_pcd_path(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> Path:
    return resolve_scenario_dir(opv2v_root, scenario_id) / cav_id / f"{frame_id}.pcd"


def load_lidar_pose(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> np.ndarray:
    yaml_path = frame_yaml_path(opv2v_root, scenario_id, cav_id, frame_id)
    data = _opencood_load_yaml(str(yaml_path))
    if "lidar_pose" not in data:
        raise KeyError(f"'lidar_pose' missing from {yaml_path} -- check this extract's yaml schema.")
    return np.asarray(data["lidar_pose"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--scenario", default="2021_09_03_20_08_37")
    args = parser.parse_args()

    frame_id, cav_ids = load_frame_and_cavs(args.scenario)
    # P0's ego convention (verified 17 Sep against OpenCOOD's own loader):
    # lowest numeric CAV id = ego, since no frame yaml declares 'ego' explicitly.
    ego_id = min(cav_ids, key=int)
    print(f"Scenario {args.scenario}, frame {frame_id}, CAVs {cav_ids}, ego={ego_id}")

    ego_pose = load_lidar_pose(args.opv2v_root, args.scenario, ego_id, frame_id)
    print(f"ego_pose (x,y,z,roll,yaw,pitch) = {ego_pose}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(15, 7))
    colors = plt.cm.tab10.colors

    before_all, after_all = [], []
    for i, cav_id in enumerate(cav_ids):
        pcd_file = frame_pcd_path(args.opv2v_root, args.scenario, cav_id, frame_id)
        points = pcd_to_np(str(pcd_file))
        points = mask_ego_points(points)

        cav_pose = load_lidar_pose(args.opv2v_root, args.scenario, cav_id, frame_id)
        print(f"CAV {cav_id} pose = {cav_pose}  ({len(points)} pts)")

        aligned = align_to_ego(points, cav_pose=cav_pose, ego_pose=ego_pose)

        color = colors[i % len(colors)]
        # 1st-99th percentile zoom per CAV's own raw points, unioned across
        # CAVs after the loop -- for now just collect for a shared window.
        before_all.append((points[:, :2], color, cav_id))
        after_all.append((aligned[:, :2], color, cav_id))

    def plot_side(ax, data, title):
        all_pts = np.vstack([xy for xy, _, _ in data])
        lo, hi = np.percentile(all_pts, [1, 99], axis=0)
        pad = 0.1 * max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
        for xy, color, cav_id in data:
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c=[color], alpha=0.5, label=f"CAV {cav_id}")
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.legend(markerscale=5, fontsize=8)

    plot_side(
        ax_before, before_all,
        f"BEFORE alignment (each CAV's own local frame)\n{args.scenario} / frame {frame_id}",
    )
    plot_side(
        ax_after, after_all,
        f"AFTER alignment (all CAVs -> ego {ego_id}'s frame)\n{args.scenario} / frame {frame_id}",
    )

    plt.tight_layout()
    out_path = OUT_DIR / f"alignment_{args.scenario}_{frame_id}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out_path}")
    print(
        "\nDone-when check: in the AFTER panel, the CAVs' point clouds should "
        "overlap on shared structure (road edges, building walls, parked "
        "vehicles seen by more than one CAV) -- if they still look like "
        "separate, offset blobs the way the BEFORE panel does, alignment "
        "isn't working and the pose convention or x1_to_x2 direction needs "
        "re-checking. If they visibly line up on shared geometry, today's "
        "done-when is met."
    )


if __name__ == "__main__":
    main()
