"""
P4 -- Explaining the clustering-failure MECHANISMS visually, with fully
synthetic, controlled scenes (NOT real OPV2V data).

Why this exists: cluster_failure_audit.py's real, quantified result
(996 real (CAV, frame, GT-vehicle) instances -- 77.9% CLEAN / 18.0%
GROUND_SWALLOWED / 3.1% SPLIT / 1.0% MERGED, see decisions.md 24 Sep 2026)
tells you THAT the real pipeline's RANSAC+DBSCAN clustering fails at these
rates. It doesn't make it easy to SEE why -- a real frame has occlusion,
several nearby objects, and sensor noise all tangled together, so the
specific geometric reason a given cluster failed is hard to isolate by eye.
This script isolates each failure mode into one minimal, controlled scene,
built with the real pipeline's own DBSCAN parameters (imported from
consistency_layer.py, not re-typed), so the mechanism is unambiguous.

Four mechanisms, four panels -- each is deliberately synthetic/illustrative,
not a calibrated LiDAR simulator or a real-data reproduction; see each
scene-builder function's own docstring for exactly what is illustrative vs.
what mirrors the real pipeline:

  1. GROUND_SWALLOWED (structural sparsity) -- a real, well-formed vehicle
     simply doesn't get enough LiDAR returns to reach DBSCAN_MIN_SAMPLES,
     because point count on a fixed-size object falls off with range (a
     LiDAR is an angular-sampling sensor: the same angular slice covers
     proportionally less of a distant object). Matches decisions.md's own
     finding that GROUND_SWALLOWED is range-correlated (mean range 46.7m vs
     26.5m for CLEAN) and that 86 of 179 real GROUND_SWALLOWED cases were
     exactly this -- too few points, structurally, to ever reach
     min_samples. The near/far vehicles are drawn SIDE BY SIDE here for
     visual comparison, not at their true ~60m separation -- the point is
     the density difference the range causes, not the literal geometry of
     the gap between them (that would need a panel many times wider than
     the vehicles themselves to stay to scale, at the cost of readability).

  2. GROUND_SWALLOWED (density, not count) -- a vehicle can clear
     DBSCAN_MIN_SAMPLES in raw point COUNT and still end up entirely
     DBSCAN noise, if those points are spread out enough that no single
     point has min_samples neighbours within eps of it. Count is necessary
     but not sufficient -- matches the other 91 of 179 real cases (>=10
     non-ground points, still all noise).

  3. MERGED -- two real, physically distinct vehicles parked close enough
     that the gap between their nearest points is smaller than
     DBSCAN_EPS_M, so eps-neighbourhood chaining links them into ONE
     cluster. Matches decisions.md's finding that real MERGED cases have a
     close mean range (15.2m) consistent with an eps-driven chaining effect
     between nearby vehicles, not a ground-removal artifact.

  4. SPLIT -- one real vehicle, partially occluded from the sensor's
     viewpoint (by another vehicle or a pole in the line of sight), leaving
     only two disconnected patches of its surface visible -- and those
     patches sit farther apart than DBSCAN_EPS_M, so DBSCAN reports TWO
     clusters for one real object.

All four panels work in the XY (bird's-eye) footprint only, exactly like
the real pipeline's own cluster_points() (consistency_layer.py runs DBSCAN
on the non-ground points' x,y only) -- height/z is irrelevant to this
mechanism and left out entirely for clarity. Panels 3 and 4 ARE drawn to
true relative scale (the gap sizes are the whole point); panels 1 and 2 are
not, as noted above and on the figure itself.

Usage (from the MastersProj repo root -- no --opv2v-root, no real data,
fully self-contained and reproducible anywhere numpy/sklearn/matplotlib are
installed):
    python P4_defence/explain_clustering_failure_mechanisms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN

REPO_ROOT = Path(__file__).resolve().parent.parent
# No sys.path insert needed for consistency_layer -- it's a sibling file in
# this same directory (P4_defence/), which Python already puts on sys.path
# automatically when this script is run directly (`python
# P4_defence/explain_clustering_failure_mechanisms.py`), same convention as
# cluster_failure_audit.py.
from consistency_layer import DBSCAN_EPS_M, DBSCAN_MIN_SAMPLES  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cache" / "audit"
RNG_SEED = 0


def expected_point_count_at_range(range_m: float, base_count: float = 220.0, base_range_m: float = 10.0) -> float:
    """
    Illustrative only -- NOT a calibrated LiDAR beam simulation. Approximates
    the real, physical reason a fixed-size object's point count falls with
    range: a LiDAR samples at fixed ANGULAR resolution (both azimuth and
    elevation), so the same angular slice covers proportionally MORE of a
    nearby object's surface than a far one, in both dimensions at once --
    hence point count on a fixed-size target falls off roughly as
    1/range^2, the same solid-angle argument that makes distant objects
    subtend less of the sensor's field of view. base_count is chosen only
    to sit in the same ballpark as this project's own real near-range
    vehicle point counts (hundreds of raw points well within 20m -- see
    decisions.md's Wed 16 Sep entries), not fit to any specific real
    measurement.
    """
    return base_count * (base_range_m / range_m) ** 2


def make_dense_object(center_xy, n_points, width=4.2, depth=1.8, seed=0):
    """A well-sensed vehicle: n_points scattered densely and fairly
    uniformly across a width x depth footprint (a plausible car silhouette
    in BEV) -- represents a CLEAN or near-range case."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-width / 2, width / 2, n_points) + center_xy[0]
    y = rng.uniform(-depth / 2, depth / 2, n_points) + center_xy[1]
    return np.column_stack([x, y])


def make_sparse_spread_object(center_xy, n_points, min_spacing, seed=0, jitter_frac=0.12):
    """
    Places exactly n_points on a jittered grid, spaced comfortably above
    min_spacing -- used for the "density, not count" panel. This guarantees
    both the exact requested point count AND that every point's nearest
    neighbour sits above min_spacing, deterministically -- unlike rejection
    sampling, which silently under-delivers points once the requested count
    approaches the area's packing limit (caught during testing: an earlier
    version of this function asked for 12 points in too little room and
    only ever placed 10). A real vehicle's surviving returns wouldn't
    literally sit on a grid, so a small random jitter is added for a more
    natural look; jitter_frac is kept small enough that it can't undo the
    spacing guarantee (worst case shrinks spacing by 2*jitter_frac, and the
    base spacing multiplier below leaves comfortable headroom above
    min_spacing for that).
    """
    rng = np.random.default_rng(seed)
    spacing = min_spacing * 1.4  # headroom so jitter can't pull spacing back below min_spacing
    ncols = int(np.ceil(np.sqrt(n_points * 1.6)))  # wider than tall, car-like proportions
    xs, ys = [], []
    r = 0
    while len(xs) < n_points:
        for c in range(ncols):
            if len(xs) >= n_points:
                break
            xs.append(c * spacing)
            ys.append(r * spacing)
        r += 1
    xs = np.array(xs) - np.mean(xs) + center_xy[0]
    ys = np.array(ys) - np.mean(ys) + center_xy[1]
    xs = xs + rng.uniform(-spacing * jitter_frac, spacing * jitter_frac, len(xs))
    ys = ys + rng.uniform(-spacing * jitter_frac, spacing * jitter_frac, len(ys))
    return np.column_stack([xs, ys])


def mean_nearest_neighbour_distance(points_xy):
    if len(points_xy) < 2:
        return float("nan")
    dists = []
    for i, p in enumerate(points_xy):
        d = np.linalg.norm(points_xy - p, axis=1)
        d[i] = np.inf
        dists.append(d.min())
    return float(np.mean(dists))


def run_dbscan(points_xy):
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=int)
    return DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(points_xy)


def draw_panel(ax, points_xy, labels, gt_boxes, sensor_xy=None, occluder=None, title="", note="", aspect_equal=True):
    noise = labels == -1
    if np.any(~noise):
        ax.scatter(points_xy[~noise, 0], points_xy[~noise, 1], s=26, c=labels[~noise], cmap="tab10", zorder=3)
    if np.any(noise):
        ax.scatter(points_xy[noise, 0], points_xy[noise, 1], s=34, c="dimgray", marker="x",
                   label="DBSCAN noise (label -1)", zorder=3)

    for (cx, cy, w, d, color, gt_label) in gt_boxes:
        rect = Rectangle((cx - w / 2, cy - d / 2), w, d, fill=False, edgecolor=color,
                          linewidth=2.0, linestyle="--", zorder=2)
        ax.add_patch(rect)
        # Label BELOW the box, not above -- keeps the top of the panel free
        # for the legend, which otherwise collides with an above-box label.
        ax.annotate(gt_label, (cx, cy - d / 2 - 0.35), color=color, fontsize=8.5, fontweight="bold",
                    ha="center", va="top")

    if occluder is not None:
        cx, cy, w, d = occluder
        rect = Rectangle((cx - w / 2, cy - d / 2), w, d, facecolor="black", alpha=0.18,
                          edgecolor="black", hatch="//", zorder=1)
        ax.add_patch(rect)
        ax.annotate("occluder", (cx, cy), fontsize=8, ha="center", va="center",
                    style="italic", fontweight="bold")

    if sensor_xy is not None:
        ax.scatter(*sensor_xy, marker="*", s=220, c="gold", edgecolor="black", zorder=4,
                   label="sensor (ego CAV, illustrative position)")

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if aspect_equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.25)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.92))


def panel_ground_swallowed_structural(ax):
    near_range, far_range = 12.0, 70.0
    gap_between_shown = 8.0  # metres -- purely compositional spacing between the two side-by-side vehicles
    near_center = (-(4.2 / 2 + gap_between_shown / 2), 0.0)
    far_center = ((4.2 / 2 + gap_between_shown / 2), 0.0)

    n_near = round(expected_point_count_at_range(near_range))
    n_far = max(1, round(expected_point_count_at_range(far_range)))

    near_pts = make_dense_object(near_center, n_near, seed=RNG_SEED)
    far_pts = make_dense_object(far_center, n_far, seed=RNG_SEED + 1)
    points = np.vstack([near_pts, far_pts])
    labels = run_dbscan(points)

    far_labels = labels[len(near_pts):]
    far_all_noise = np.all(far_labels == -1)

    note = (
        f"near @ {near_range:.0f}m from sensor: ~{n_near} pts -> clusters fine\n"
        f"far  @ {far_range:.0f}m from sensor: ~{n_far} pts (< min_samples={DBSCAN_MIN_SAMPLES})\n"
        f"far vehicle: {'ALL NOISE (swallowed)' if far_all_noise else 'clustered'}\n"
        f"point count ~ 1/range^2 (illustrative model)\n"
        f"[shown side-by-side for comparison -- not to true scale]"
    )
    draw_panel(
        ax, points, labels,
        gt_boxes=[(*near_center, 4.2, 1.8, "tab:blue", f"GT vehicle, {near_range:.0f}m away"),
                  (*far_center, 4.2, 1.8, "tab:red", f"GT vehicle, {far_range:.0f}m away")],
        title="GROUND_SWALLOWED -- structural sparsity\n(too few points to ever reach min_samples)",
        note=note,
        aspect_equal=False,
    )
    return far_all_noise, n_near, n_far


def panel_ground_swallowed_density(ax):
    center = (0.0, 0.0)
    n_points = DBSCAN_MIN_SAMPLES + 2  # clears min_samples in raw COUNT
    min_spacing = DBSCAN_EPS_M * 1.2  # deliberately over eps

    pts = make_sparse_spread_object(center, n_points, min_spacing, seed=RNG_SEED)
    labels = run_dbscan(pts)
    all_noise = np.all(labels == -1)
    mean_nn = mean_nearest_neighbour_distance(pts)

    note = (
        f"{len(pts)} points total (>= min_samples={DBSCAN_MIN_SAMPLES})\n"
        f"mean nearest-neighbour spacing: {mean_nn:.2f}m > eps={DBSCAN_EPS_M}m\n"
        f"status: {'ALL NOISE (swallowed despite count)' if all_noise else 'clustered'}\n"
        f"count alone is not sufficient -- spacing matters too"
    )
    draw_panel(
        ax, pts, labels,
        gt_boxes=[(*center, 4.2, 1.8, "tab:red", "GT vehicle (moderate range)")],
        sensor_xy=(0.0, -4.0),
        title="GROUND_SWALLOWED -- density, not count\n(enough points, too spread out)",
        note=note,
    )
    return all_noise, mean_nn


def panel_merged(ax):
    gap = 0.3  # metres, deliberately < DBSCAN_EPS_M
    car_w, car_d = 4.2, 1.8
    center_a = (-(car_w / 2 + gap / 2), 0.0)
    center_b = ((car_w / 2 + gap / 2), 0.0)

    pts_a = make_dense_object(center_a, 90, width=car_w, depth=car_d, seed=RNG_SEED)
    pts_b = make_dense_object(center_b, 90, width=car_w, depth=car_d, seed=RNG_SEED + 1)
    points = np.vstack([pts_a, pts_b])
    labels = run_dbscan(points)

    n_clusters = len(set(labels.tolist()) - {-1})
    merged = n_clusters == 1

    note = (
        f"gap between vehicles: {gap}m < eps={DBSCAN_EPS_M}m\n"
        f"DBSCAN cluster count: {n_clusters} (for 2 real vehicles)\n"
        f"status: {'MERGED into one cluster' if merged else 'correctly separated'}\n"
        f"eps-neighbourhood chaining bridges the gap\n"
        f"[drawn to true relative scale]"
    )
    draw_panel(
        ax, points, labels,
        gt_boxes=[(*center_a, car_w, car_d, "tab:blue", "GT vehicle A"),
                  (*center_b, car_w, car_d, "tab:red", "GT vehicle B")],
        sensor_xy=(0.0, -4.0),
        title="MERGED -- eps-bridging across a small gap\n(two real vehicles, one DBSCAN cluster)",
        note=note,
    )
    return merged, n_clusters


def panel_split(ax):
    car_w, car_d = 4.6, 1.8
    center = (0.0, 0.0)
    occlusion_gap = 1.2  # metres, deliberately > DBSCAN_EPS_M

    front_center = (center[0] - car_w / 4 - occlusion_gap / 4, center[1])
    rear_center = (center[0] + car_w / 4 + occlusion_gap / 4, center[1])
    patch_w = (car_w - occlusion_gap) / 2

    front_pts = make_dense_object(front_center, 45, width=patch_w, depth=car_d, seed=RNG_SEED)
    rear_pts = make_dense_object(rear_center, 45, width=patch_w, depth=car_d, seed=RNG_SEED + 1)
    points = np.vstack([front_pts, rear_pts])
    labels = run_dbscan(points)

    n_clusters = len(set(labels.tolist()) - {-1})
    split = n_clusters >= 2

    note = (
        f"occlusion gap: {occlusion_gap}m > eps={DBSCAN_EPS_M}m\n"
        f"DBSCAN cluster count: {n_clusters} (for 1 real vehicle)\n"
        f"status: {'SPLIT across ' + str(n_clusters) + ' clusters' if split else 'correctly merged'}\n"
        f"occluder breaks line of sight to the car's middle\n"
        f"[drawn to true relative scale]"
    )
    draw_panel(
        ax, points, labels,
        gt_boxes=[(*center, car_w, car_d, "tab:red", "GT vehicle (one real object)")],
        occluder=(center[0], center[1] - car_d / 2 - 1.3, occlusion_gap * 1.8, 1.4),
        sensor_xy=(0.0, -4.0),
        title="SPLIT -- occlusion breaks one vehicle in two\n(1 real vehicle, >=2 DBSCAN clusters)",
        note=note,
    )
    return split, n_clusters


def main():
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    fig.suptitle(
        f"Why RANSAC+DBSCAN clustering fails on real LiDAR data\n"
        f"(synthetic, controlled scenes -- real pipeline constants: "
        f"eps={DBSCAN_EPS_M}m, min_samples={DBSCAN_MIN_SAMPLES})",
        fontsize=14, fontweight="bold",
    )

    far_swallowed, n_near, n_far = panel_ground_swallowed_structural(axes[0, 0])
    density_swallowed, mean_nn = panel_ground_swallowed_density(axes[0, 1])
    merged, n_clusters_merged = panel_merged(axes[1, 0])
    split, n_clusters_split = panel_split(axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / "clustering_failure_mechanisms_synthetic.png"
    plt.savefig(out_path, dpi=150)

    print("=" * 70)
    print("SYNTHETIC MECHANISM DEMO -- not real data, see module docstring")
    print("=" * 70)
    print(f"Real pipeline constants used: DBSCAN_EPS_M={DBSCAN_EPS_M}, "
          f"DBSCAN_MIN_SAMPLES={DBSCAN_MIN_SAMPLES} (imported from consistency_layer.py)")
    print()
    print(f"[1] GROUND_SWALLOWED (structural): near vehicle ~{n_near} pts -> clusters fine. "
          f"far vehicle ~{n_far} pts -> {'ALL NOISE' if far_swallowed else 'clustered (unexpected)'}. "
          f"Matches real finding: GROUND_SWALLOWED mean range 46.7m vs 26.5m for CLEAN.")
    print(f"[2] GROUND_SWALLOWED (density): {DBSCAN_MIN_SAMPLES + 2} pts (clears min_samples) but "
          f"mean nearest-neighbour spacing {mean_nn:.2f}m > eps -> "
          f"{'ALL NOISE despite count' if density_swallowed else 'clustered (unexpected)'}. "
          f"Matches real finding: 91/179 real GROUND_SWALLOWED cases had >=10 non-ground points "
          f"and still ended up entirely noise.")
    print(f"[3] MERGED: 0.3m gap (< eps={DBSCAN_EPS_M}m) between two real vehicles -> "
          f"{n_clusters_merged} DBSCAN cluster(s) for 2 real vehicles "
          f"({'MERGED' if merged else 'correctly separated (unexpected)'}). "
          f"Matches real finding: real MERGED cases have a close mean range (15.2m), "
          f"consistent with eps-driven chaining between nearby vehicles.")
    print(f"[4] SPLIT: 1.2m occlusion gap (> eps={DBSCAN_EPS_M}m) across one real vehicle -> "
          f"{n_clusters_split} DBSCAN cluster(s) for 1 real vehicle "
          f"({'SPLIT' if split else 'correctly merged (unexpected)'}).")
    print()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
