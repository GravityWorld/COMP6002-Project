"""
P4 — Standalone occupancy-consistency layer.

Per scope_and_timeline_oct19.md and execution_plan_daily.md (Thu 3 / Fri 4 Sep):
this half of the defence needs NO GPU — RANSAC ground-plane removal + DBSCAN
clustering + shapely polygons on raw point clouds. It de-risks independently
of the detector (which already works, per Sat 5 Sep's log).

This file has a real, runnable implementation, not just stubs — get today's
two done-when checks working against it:
  1. Plot clusters from one real point cloud.
  2. conflicting_area() passes a synthetic test where you know ground truth.

Full-depth instrumentation (execution_plan_daily.md, Sat 12 Sep) logs more
scalars than conflicting_area alone — that comes later in W2. Today is just:
does the geometry pipeline work end to end.

Before trusting these defaults: read CAD's released code (arXiv:2309.12955)
to confirm the ~2.7 m² threshold and the exact RANSAC/DBSCAN parameters they
use — this file's constants are reasonable starting points, not verified
against the paper yet.
"""

from __future__ import annotations  # Py3.7 compat: allows list[Polygon] etc. without erroring

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

# Starting defaults — CONFIRM against CAD's released code before relying on these.
RANSAC_RESIDUAL_THRESHOLD_M = 0.2   # ground-plane inlier tolerance
DBSCAN_EPS_M = 0.75
DBSCAN_MIN_SAMPLES = 10
CONFLICT_AREA_THRESHOLD_M2 = 2.7    # per task1_datasets_pipeline_dfl.md §5 — verify exact figure in CAD's code


def remove_ground_plane(points: np.ndarray) -> np.ndarray:
    """
    RANSAC-fit a ground plane on z as a function of (x, y) and drop inliers.
    points: (N, 3+) array, first 3 columns x, y, z.
    Returns the non-ground points.
    """
    xy = points[:, :2]
    z = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=RANSAC_RESIDUAL_THRESHOLD_M)
    ransac.fit(xy, z)
    inlier_mask = ransac.inlier_mask_
    return points[~inlier_mask]


def cluster_points(points: np.ndarray) -> np.ndarray:
    """
    DBSCAN cluster the non-ground points (x, y only — clustering in 2D
    footprint is standard for occupancy-style checks; revisit if CAD's code
    clusters in 3D).
    Returns integer cluster labels, -1 for noise.
    """
    xy = points[:, :2]
    labels = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
    return labels


def clusters_to_polygons(points: np.ndarray, labels: np.ndarray) -> list[Polygon]:
    """Convex hull per non-noise cluster -> list of shapely Polygons."""
    polygons = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_xy = points[labels == label][:, :2]
        if len(cluster_xy) < 3:
            continue  # can't form a polygon from < 3 points
        hull = MultiPoint(cluster_xy).convex_hull
        if isinstance(hull, Polygon):
            polygons.append(hull)
    return polygons


@dataclass
class ConflictResult:
    total_conflicting_area_m2: float
    conflict_polygon_count: int
    largest_conflict_area_m2: float
    flagged: bool


def conflicting_area(polygons_a: list[Polygon], polygons_b: list[Polygon]) -> ConflictResult:
    """
    Pairwise conflicting-area computation between two CAVs' occupancy maps.

    "Conflict" = area where CAV A claims occupancy but CAV B's own occupancy
    map disagrees within the same footprint (or vice versa) — i.e. symmetric
    difference of the unioned occupancy polygons. This is a first-pass
    definition; cross-check it against CAD's actual definition when you read
    their code (today's other task).
    """
    union_a = unary_union(polygons_a) if polygons_a else Polygon()
    union_b = unary_union(polygons_b) if polygons_b else Polygon()

    sym_diff = union_a.symmetric_difference(union_b)

    if sym_diff.is_empty:
        return ConflictResult(0.0, 0, 0.0, False)

    # symmetric_difference can return a Polygon or MultiPolygon
    geoms = list(sym_diff.geoms) if hasattr(sym_diff, "geoms") else [sym_diff]
    areas = [g.area for g in geoms]

    total = sum(areas)
    return ConflictResult(
        total_conflicting_area_m2=total,
        conflict_polygon_count=len(geoms),
        largest_conflict_area_m2=max(areas) if areas else 0.0,
        flagged=total > CONFLICT_AREA_THRESHOLD_M2,
    )
