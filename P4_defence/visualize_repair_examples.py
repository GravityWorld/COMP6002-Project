"""
P4/P5 -- Real-data gallery: shows the actual clustering-failure gap found by
run_cluster_repair_labelling.py's real output, in one figure.

Unlike explain_clustering_failure_mechanisms.py (synthetic, illustrates WHY
the failures happen in principle), this script loads the REAL per-point
labels (*_points.npz) that run_cluster_repair_labelling.py wrote from the
actual complete OPV2V dataset, picks one real, representative example of
each failure mode, and plots what DBSCAN actually did to a real vehicle
next to what the P5 oracle chain (point_vehicle_id / fragment_id /
repaired_cluster_id) says should have happened.

Classification per real vehicle re-derives the same CLEAN/MERGED/SPLIT/
GROUND_SWALLOWED taxonomy as cluster_failure_audit.py, but here it's a
by-product of diffing raw_cluster_id against point_vehicle_id -- no
re-running of the detection pipeline needed, since run_cluster_repair_
labelling.py already computed everything this script reads.

Usage (from the MastersProj repo root, same light deps as the labelling
script -- numpy, pandas, shapely, matplotlib; no opencood/torch import at
all, since this only reads the .npz files the labelling run already wrote):
    python P4_defence/visualize_repair_examples.py \
        --labels-dir P4_defence/cache/repair_labels \
        --out P4_defence/cache/audit/real_repair_examples.png

Caveat carried over from run_cluster_repair_labelling.py: only trust this
figure's "found in the real dataset" framing when --labels-dir was produced
by a run against the COMPLETE dataset, not the partial local mirror.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPoint

BACKGROUND_VEHICLE_ID = -1
NOISE_CLUSTER_ID = -1

FRAME_CAV_RE = re.compile(r"^(?P<frame>\d+)_(?P<cav>.+)_points\.npz$")


def find_npz_files(labels_dir: Path):
    for scenario_dir in sorted(p for p in labels_dir.iterdir() if p.is_dir()):
        for npz_path in sorted(scenario_dir.glob("*_points.npz")):
            m = FRAME_CAV_RE.match(npz_path.name)
            if not m:
                continue
            yield scenario_dir.name, m.group("frame"), m.group("cav"), npz_path


def classify_file(npz_path: Path):
    """
    Returns a list of per-vehicle records for one (scenario, frame, cav) file:
    dicts with vid, category, n_points, positives (set of raw cluster ids
    the vehicle's points touch, excluding noise), n_fragments.
    Also returns raw_cluster_to_vehicles: dict[cluster_id] -> set(vid) for
    MERGED lookups.
    """
    d = np.load(npz_path)
    raw_cluster_id = d["raw_cluster_id"]
    point_vehicle_id = d["point_vehicle_id"]
    fragment_id = d["fragment_id"]

    vids = sorted(set(point_vehicle_id.tolist()) - {BACKGROUND_VEHICLE_ID})

    raw_cluster_to_vehicles = {}
    for cid in set(raw_cluster_id[raw_cluster_id >= 0].tolist()):
        mask = raw_cluster_id == cid
        vs = set(point_vehicle_id[mask].tolist()) - {BACKGROUND_VEHICLE_ID}
        if vs:
            raw_cluster_to_vehicles[cid] = vs

    records = []
    for vid in vids:
        mask = point_vehicle_id == vid
        n_points = int(mask.sum())
        positives = set(raw_cluster_id[mask].tolist()) - {NOISE_CLUSTER_ID}
        n_fragments = len(set(fragment_id[mask].tolist()))

        if not positives:
            category = "GROUND_SWALLOWED"
        elif len(positives) >= 2:
            category = "SPLIT"
        else:
            (only_cluster,) = positives
            if len(raw_cluster_to_vehicles.get(only_cluster, {vid})) >= 2:
                category = "MERGED"
            else:
                category = "CLEAN"

        records.append(dict(
            vid=vid, category=category, n_points=n_points,
            positives=positives, n_fragments=n_fragments,
        ))
    return records, raw_cluster_to_vehicles


def scan_all(labels_dir: Path):
    """Scans every points.npz once; returns candidate lists per category,
    each entry carrying enough to re-load and plot later without re-scanning."""
    candidates = {"GROUND_SWALLOWED": [], "SPLIT": [], "MERGED": []}
    merged_clusters_seen = set()

    for scenario_id, frame_id, cav_id, npz_path in find_npz_files(labels_dir):
        records, raw_cluster_to_vehicles = classify_file(npz_path)
        for rec in records:
            key = (scenario_id, frame_id, cav_id)
            if rec["category"] == "GROUND_SWALLOWED":
                candidates["GROUND_SWALLOWED"].append(dict(
                    key=key, npz_path=npz_path, vid=rec["vid"],
                    n_points=rec["n_points"], n_fragments=rec["n_fragments"],
                ))
            elif rec["category"] == "SPLIT" and len(rec["positives"]) == 2:
                candidates["SPLIT"].append(dict(
                    key=key, npz_path=npz_path, vid=rec["vid"],
                    n_points=rec["n_points"], positives=rec["positives"],
                ))
        # MERGED: one candidate per (raw cluster, vehicle-pair), not per vehicle
        for cid, vs in raw_cluster_to_vehicles.items():
            if len(vs) == 2 and (scenario_id, frame_id, cav_id, cid) not in merged_clusters_seen:
                merged_clusters_seen.add((scenario_id, frame_id, cav_id, cid))
                candidates["MERGED"].append(dict(
                    key=key, npz_path=npz_path, cid=cid, vids=sorted(vs),
                ))
    return candidates


def pick_ground_swallowed(cands):
    # Prefer a decent point count AND a multi-fragment recovery story
    # (matches the Sat 26 Sep finding: recovery doesn't collapse to 1 piece).
    scored = [c for c in cands if 15 <= c["n_points"] <= 90]
    multi = [c for c in scored if c["n_fragments"] >= 2]
    pool = multi if multi else (scored if scored else cands)
    return max(pool, key=lambda c: c["n_points"])


def pick_split(cands):
    # Same pitfall as MERGED: a vehicle box sitting next to a wall/structure
    # can inflate one of its two raw-cluster counts into the hundreds+ (the
    # box catches adjacent structure points too), which reads as "a building
    # split into two clusters," not "one car split into two." Restrict to a
    # vehicle-scale total and prefer a roughly balanced two-way split so both
    # clusters are clearly visible car-sized clusters, not one dominant blob.
    scored = []
    for c in cands:
        if not (15 <= c["n_points"] <= 150):
            continue
        d = np.load(c["npz_path"])
        vmask = d["point_vehicle_id"] == c["vid"]
        counts = [int(((d["raw_cluster_id"] == cid) & vmask).sum()) for cid in c["positives"]]
        balance = min(counts) / max(counts) if max(counts) else 0
        scored.append(dict(c, balance=balance))
    pool = scored if scored else [dict(c, balance=0) for c in cands]
    return max(pool, key=lambda c: (c["balance"], c["n_points"]))


def pick_merged(cands, labels_dir):
    # A raw cluster shared by 2 GT vehicle boxes is sometimes a genuine
    # vehicle-vehicle merge, but can also be a large building/wall cluster
    # that happens to sweep across the edges of two unrelated, distant
    # vehicle boxes (this dataset's busy-intersection scenes have plenty of
    # 1000+ point wall clusters -- see Wed 16 Sep's log). Prefer the
    # SMALLEST qualifying shared cluster: that's the one most likely to be
    # an actual two-nearby-vehicles case, not a structural artifact, and
    # it's the only one small enough to read clearly in a zoomed panel.
    scored = []
    for c in cands:
        d = np.load(c["npz_path"])
        mask = d["raw_cluster_id"] == c["cid"]
        pvid = d["point_vehicle_id"][mask]
        n0 = int((pvid == c["vids"][0]).sum())
        n1 = int((pvid == c["vids"][1]).sum())
        total_cluster = int(mask.sum())
        if n0 >= 5 and n1 >= 5:
            scored.append(dict(c, n0=n0, n1=n1, total_cluster=total_cluster))
    pool = scored if scored else [dict(c, n0=0, n1=0, total_cluster=10**9) for c in cands]
    return min(pool, key=lambda c: c["total_cluster"])


def hull_xy(points_xy):
    if len(points_xy) < 3:
        return points_xy
    hull = MultiPoint(points_xy).convex_hull
    if hull.geom_type != "Polygon":
        return points_xy
    return np.array(hull.exterior.coords)


def crop_mask(points_xy, center, half_extent):
    return (
        (np.abs(points_xy[:, 0] - center[0]) <= half_extent)
        & (np.abs(points_xy[:, 1] - center[1]) <= half_extent)
    )


CONTEXT_ZOOM_M = 12.0  # matches plot_failure_cases.py's fixed zoom radius


def draw_context(ax, points_xy, raw_cluster_id, crop_m):
    """
    Real scene context, same convention as plot_failure_cases.py /
    run_consistency_check.py: every OTHER real DBSCAN cluster in the crop is
    drawn in its own tab20 colour (so you can see this case sitting among
    its actual neighbouring clusters, not stripped down to flat gray), and
    noise stays a small dimgray 'x'. The case being illustrated is drawn on
    top of this by the caller, so it isn't excluded here.
    """
    noise = crop_m & (raw_cluster_id == -1)
    clustered = crop_m & (raw_cluster_id >= 0)
    ax.scatter(points_xy[noise, 0], points_xy[noise, 1], s=10, c="dimgray",
               marker="x", linewidths=0.8, zorder=1, label="_nolegend_")
    if clustered.any():
        ax.scatter(points_xy[clustered, 0], points_xy[clustered, 1], s=8,
                   c=raw_cluster_id[clustered], cmap="tab20", zorder=1,
                   label="_nolegend_")


def panel_ground_swallowed_raw(ax, rec):
    d = np.load(rec["npz_path"])
    xy = d["xyz"][:, :2]
    pvid = d["point_vehicle_id"]
    vmask = pvid == rec["vid"]
    center = xy[vmask].mean(axis=0)
    crop_m = crop_mask(xy, center, half_extent=CONTEXT_ZOOM_M)

    draw_context(ax, xy, d["raw_cluster_id"], crop_m)
    ax.scatter(xy[vmask, 0], xy[vmask, 1], s=42, c="crimson", marker="x",
               linewidths=1.8, zorder=3, label="real vehicle's points\n(all DBSCAN noise)")
    hull = hull_xy(xy[vmask])
    ax.add_patch(MplPolygon(hull, closed=True, fill=False, edgecolor="crimson",
                             linewidth=2.2, linestyle="--", zorder=3))
    ax.set_title("GROUND_SWALLOWED -- what DBSCAN actually produced (in scene context)",
                 fontsize=11, fontweight="bold")
    scen, frame, cav = rec["key"]
    ax.text(0.02, 0.02,
            f"real vehicle {rec['vid']}  |  {rec['n_points']} pts, "
            f"0 survive as a cluster\n{scen} / frame {frame} / CAV {cav}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def panel_ground_swallowed_recovered(ax, rec):
    d = np.load(rec["npz_path"])
    xy = d["xyz"][:, :2]
    pvid = d["point_vehicle_id"]
    frag = d["fragment_id"]
    vmask = pvid == rec["vid"]
    center = xy[vmask].mean(axis=0)
    crop_m = crop_mask(xy, center, half_extent=CONTEXT_ZOOM_M)

    draw_context(ax, xy, d["raw_cluster_id"], crop_m)
    frag_ids = sorted(set(frag[vmask].tolist()))
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(frag_ids):
        m = vmask & (frag == f)
        ax.scatter(xy[m, 0], xy[m, 1], s=46, color=cmap(i % 10), zorder=3,
                   edgecolors="black", linewidths=0.4,
                   label=f"recovered fragment {f} ({int(m.sum())} pts)")
    hull = hull_xy(xy[vmask])
    ax.add_patch(MplPolygon(hull, closed=True, fill=False, edgecolor="crimson",
                             linewidth=2.2, linestyle="--", zorder=3, label="real vehicle (1 object)"))
    ax.set_title(f"...after the noise-recovery pass: {len(frag_ids)} separate fragments",
                 fontsize=11, fontweight="bold")
    ax.text(0.02, 0.02,
            "the repair head's job: learn these fragments\nhave affinity -- they're the same object",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def panel_merged(ax, rec):
    d = np.load(rec["npz_path"])
    xy = d["xyz"][:, :2]
    raw_cluster_id = d["raw_cluster_id"]
    pvid = d["point_vehicle_id"]
    cmask = raw_cluster_id == rec["cid"]
    center = xy[cmask].mean(axis=0)
    cluster_half_span = float(np.max(np.abs(xy[cmask] - center))) if cmask.any() else 3.0
    half_extent = max(CONTEXT_ZOOM_M, cluster_half_span + 2.0)
    crop_m = crop_mask(xy, center, half_extent=half_extent)

    draw_context(ax, xy, raw_cluster_id, crop_m)
    ax.scatter(xy[cmask, 0], xy[cmask, 1], s=20, c="#4c72b0", zorder=3,
               edgecolors="black", linewidths=0.3,
               label=f"DBSCAN cluster {rec['cid']} ({int(cmask.sum())} pts, 1 cluster)")

    colors = ["crimson", "darkorange"]
    for i, vid in enumerate(rec["vids"]):
        vmask = cmask & (pvid == vid)
        hull = hull_xy(xy[vmask])
        ax.add_patch(MplPolygon(hull, closed=True, fill=False, edgecolor=colors[i],
                                 linewidth=2.2, linestyle="--", zorder=3,
                                 label=f"real vehicle {vid} ({int(vmask.sum())} pts)"))
    ax.set_title("MERGED -- one DBSCAN cluster, two real vehicles (in scene context)",
                 fontsize=11, fontweight="bold")
    scen, frame, cav = rec["key"]
    ax.text(0.02, 0.02, f"{scen} / frame {frame} / CAV {cav}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def panel_split(ax, rec):
    d = np.load(rec["npz_path"])
    xy = d["xyz"][:, :2]
    raw_cluster_id = d["raw_cluster_id"]
    pvid = d["point_vehicle_id"]
    vmask = pvid == rec["vid"]
    center = xy[vmask].mean(axis=0)
    crop_m = crop_mask(xy, center, half_extent=CONTEXT_ZOOM_M)

    draw_context(ax, xy, raw_cluster_id, crop_m)
    colors = ["#4c72b0", "#55a868"]
    for i, cid in enumerate(sorted(rec["positives"])):
        m = vmask & (raw_cluster_id == cid)
        ax.scatter(xy[m, 0], xy[m, 1], s=34, color=colors[i % 2], zorder=3,
                   edgecolors="black", linewidths=0.3,
                   label=f"DBSCAN cluster {cid} ({int(m.sum())} pts)")
    hull = hull_xy(xy[vmask])
    ax.add_patch(MplPolygon(hull, closed=True, fill=False, edgecolor="crimson",
                             linewidth=2.2, linestyle="--", zorder=3,
                             label=f"real vehicle {rec['vid']} (1 object)"))
    ax.set_title("SPLIT -- one real vehicle, two DBSCAN clusters (in scene context)",
                 fontsize=11, fontweight="bold")
    scen, frame, cav = rec["key"]
    ax.text(0.02, 0.02, f"{scen} / frame {frame} / CAV {cav}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def compute_summary_stats(labels_dir: Path):
    """Re-aggregates the same headline numbers run_cluster_repair_labelling.py
    printed at the end of its own run, straight from the fragments/pairs
    parquet files -- so the figure's caption always matches whatever
    --labels-dir it was actually built from, not a stale hardcoded number."""
    n_rows = 0
    n_fragments = 0
    n_recovered = 0
    n_pairs = 0
    n_pairs_affinity_true = 0
    for scenario_dir in labels_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        for frag_path in scenario_dir.glob("*_fragments.parquet"):
            n_rows += 1
            fdf = pd.read_parquet(frag_path)
            n_fragments += len(fdf)
            if len(fdf):
                n_recovered += int(fdf["is_recovered_noise"].sum())
            pairs_path = frag_path.with_name(frag_path.name.replace("_fragments.", "_pairs."))
            if pairs_path.exists():
                pdf = pd.read_parquet(pairs_path)
                n_pairs += len(pdf)
                if len(pdf):
                    n_pairs_affinity_true += int(pdf["affinity_label"].sum())
    return dict(n_rows=n_rows, n_fragments=n_fragments, n_recovered=n_recovered,
                n_pairs=n_pairs, n_pairs_affinity_true=n_pairs_affinity_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default="P4_defence/cache/repair_labels")
    parser.add_argument("--out", default="P4_defence/cache/audit/real_repair_examples.png")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    candidates = scan_all(labels_dir)

    n_gs, n_split, n_merged = (len(candidates["GROUND_SWALLOWED"]),
                                len(candidates["SPLIT"]), len(candidates["MERGED"]))
    print(f"Real candidates found -- GROUND_SWALLOWED: {n_gs}  SPLIT: {n_split}  MERGED: {n_merged}")
    if not (n_gs and n_split and n_merged):
        raise SystemExit(
            "Not enough real examples of every category in this labels-dir "
            "to build the gallery -- run against more scenarios/frames, or "
            "the complete dataset, first."
        )

    gs_rec = pick_ground_swallowed(candidates["GROUND_SWALLOWED"])
    split_rec = pick_split(candidates["SPLIT"])
    merged_rec = pick_merged(candidates["MERGED"], labels_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    panel_ground_swallowed_raw(axes[0, 0], gs_rec)
    panel_ground_swallowed_recovered(axes[0, 1], gs_rec)
    panel_merged(axes[1, 0], merged_rec)
    panel_split(axes[1, 1], split_rec)

    stats = compute_summary_stats(labels_dir)
    pct_recovered = 100 * stats["n_recovered"] / stats["n_fragments"] if stats["n_fragments"] else 0.0
    pct_affinity = 100 * stats["n_pairs_affinity_true"] / stats["n_pairs"] if stats["n_pairs"] else 0.0

    fig.suptitle(
        "Real clustering failures found in the actual OPV2V run\n"
        "(from run_cluster_repair_labelling.py's real output -- not synthetic)",
        fontsize=15, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.925,
        f"{stats['n_rows']} (scenario, frame, CAV) rows labelled  |  "
        f"{stats['n_fragments']} candidate fragments ({pct_recovered:.1f}% recovered from noise)  |  "
        f"{stats['n_pairs']} candidate pairs ({pct_affinity:.1f}% same-object)",
        ha="center", fontsize=10.5, color="#333333",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    print(f"\nGROUND_SWALLOWED example: vehicle {gs_rec['vid']} in {gs_rec['key']}, "
          f"{gs_rec['n_points']} pts, recovered into {gs_rec['n_fragments']} fragments")
    print(f"SPLIT example: vehicle {split_rec['vid']} in {split_rec['key']}, "
          f"clusters {sorted(split_rec['positives'])}")
    print(f"MERGED example: cluster {merged_rec['cid']} in {merged_rec['key']}, "
          f"vehicles {merged_rec['vids']} ({merged_rec['n0']} + {merged_rec['n1']} pts)")


if __name__ == "__main__":
    main()
