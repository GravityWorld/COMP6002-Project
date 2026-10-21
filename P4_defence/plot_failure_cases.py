"""
P4 -- Illustrative plot of the three clustering-failure modes (MERGED, SPLIT,
GROUND_SWALLOWED) found by cluster_failure_audit.py, on real OPV2V data.

Hardcodes three specific (scenario, frame, cav, vehicle_ids) cases that were
found and manually verified during the Thu 24 Sep 2026 audit (see
decisions.md) to be clean illustrations of each failure mode. Re-run
cluster_failure_audit.py and inspect vehicle_audit.json if you want to swap
in different examples later (e.g. after the RANSAC random_state fix changes
which exact cases fall into MERGED/SPLIT/GROUND_SWALLOWED).

Usage (from the MastersProj repo root):
    python P4_defence/plot_failure_cases.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# See P0_scenario_indexing/index.py's module comment: OPV2V's official
# train/validate/test splits dump numpy scalars into frame yaml, which
# yaml.safe_load() refuses -- reusing OpenCOOD's own loader (same fix).
from opencood.hypes_yaml.yaml_utils import load_yaml as _opencood_load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402

from cluster_failure_audit import (  # noqa: E402
    remove_ground_plane_seeded,
    cluster_points,
    project_vehicle_to_lidar,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cache" / "audit"

CASES = [
    ("MERGED — 1 cluster spans 2 vehicles", "2021_09_03_20_08_37", "003464", "1854", ["1931", "1959"]),
    ("SPLIT — 1 vehicle spread across 2 clusters", "2021_09_03_20_08_37", "003464", "1854", ["1889"]),
    ("GROUND-SWALLOWED — real object, zero surviving cluster", "2021_09_03_20_08_37", "003464", "1845", ["1924"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for ax, (title, scenario, frame, cav, veh_ids) in zip(axes, CASES):
        yaml_path = Path(args.opv2v_root) / scenario / cav / f"{frame}.yaml"
        pcd_path = Path(args.opv2v_root) / scenario / cav / f"{frame}.pcd"
        meta = _opencood_load_yaml(str(yaml_path))
        lidar_pose = meta["lidar_pose"]
        raw = mask_ego_points(pcd_to_np(str(pcd_path)))
        non_ground = remove_ground_plane_seeded(raw)
        labels = cluster_points(non_ground)

        boxes = [project_vehicle_to_lidar(meta["vehicles"][int(vid)], lidar_pose) for vid in veh_ids]
        centers = np.array([b[:4, :2].mean(axis=0) for b in boxes])
        cx, cy = centers.mean(axis=0)
        half = 12  # metres, zoom window

        noise = labels == -1
        ax.scatter(non_ground[noise, 0], non_ground[noise, 1], s=18, c="dimgray", marker="x",
                   label="noise/unclustered (DBSCAN label -1)")
        ax.scatter(non_ground[~noise, 0], non_ground[~noise, 1], s=8, c=labels[~noise], cmap="tab20")

        for vid, box in zip(veh_ids, boxes):
            xy = box[:4, :2]
            xy_closed = np.vstack([xy, xy[0]])
            ax.plot(xy_closed[:, 0], xy_closed[:, 1], "r-", linewidth=1.8)
            ax.annotate(f"GT veh {vid}", xy=xy.mean(axis=0), color="red", fontsize=9, fontweight="bold")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f"{title}\n{scenario} / {frame} / CAV{cav}", fontsize=10)
        ax.set_xlabel("x (m, CAV-local lidar frame)")
        ax.set_ylabel("y (m)")

    plt.tight_layout()
    out_path = out_dir / "clustering_failure_examples.png"
    plt.savefig(out_path, dpi=150)
    print("saved", out_path)


if __name__ == "__main__":
    main()
