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

CAD's actual released code was read on 16 Sep 2026 (arXiv:2309.12955, repo
github.com/zqzqz/AdvCollaborativePerception) -- see decisions.md for the full
comparison. Findings that matter for this file:
  - RANSAC ground-plane removal matches CAD's own method almost exactly,
    including the same 0.2m residual threshold (mvp/tools/ground_detection.py
    get_ground_plane_ransac). No change needed.
  - CAD's actual flag decision does NOT diff two CAVs' cluster/blob maps
    against each other the way conflicting_area() below does. It compares a
    CAV's claimed OCCUPIED space against another CAV's claimed FREE space
    (computed via ray-casting -- everything up to a LiDAR ray's first hit is
    "confirmed empty"), plus the actual detector's predicted boxes. That is
    a materially different (and more faithful) check than blob-vs-blob.
  - conflicting_area() is kept below as the original simplified version (and
    its synthetic tests still document known ground truth), but
    get_free_space() + free_vs_occupied_violation() below are the actual
    flag rule used going forward -- a deliberately simplified version of
    CAD's real free/occupied logic (no lane-area filtering, no detector
    wiring yet -- P3 doesn't exist yet, see P3_perception/perception_cache.py).
    Document this simplification explicitly in the thesis; it is a known,
    chosen scope cut, not an oversight.
  - The ~2.7 m² threshold was NOT found anywhere in CAD's own code (no
    literal "2.7" constant, no DBSCAN either) -- it traces back to a
    different paper's analysis of CAD (task1_datasets_pipeline_dfl.md §5),
    not CAD's own source. Kept as the working threshold since it's the
    number the project's docs already committed to, but it is a borrowed
    estimate, not a verified constant from CAD itself.
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

    random_state is fixed (added 25 Sep 2026) because RANSACRegressor's
    internal random sampling otherwise draws from global numpy state, which
    differs run to run -- confirmed empirically (24 Sep 2026, see
    decisions.md) that re-running ground removal + clustering on the
    IDENTICAL frame/CAV twice produced different inlier masks and different
    downstream cluster ID assignments. A defence whose ground-removal isn't
    deterministic on identical input is a real bug independent of anything
    else this file does -- fixed here rather than left as a known issue.
    """
    xy = points[:, :2]
    z = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=RANSAC_RESIDUAL_THRESHOLD_M, random_state=0)
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
    if len(points) == 0:
        # A CAV can genuinely see nothing but flat ground in a frame (e.g.
        # a very clean stretch of road) -- zero non-ground points is a real
        # case, not an error. DBSCAN itself throws on an empty array, so
        # short-circuit here rather than letting every caller guard this.
        return np.zeros((0,), dtype=int)
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


# ---------------------------------------------------------------------------
# Free-space + free-vs-occupied flag rule (added 16 Sep 2026, after reading
# CAD's actual code) -- see this file's module docstring for why this
# replaces conflicting_area() as the flag rule going forward.
# ---------------------------------------------------------------------------

FREE_SPACE_ANGLE_BINS = 360   # angular resolution of the ray-cast sweep
FREE_SPACE_MAX_RANGE_M = 120.0  # matches OPV2V's standard LiDAR clip range


def get_free_space(
    points: np.ndarray,
    angle_bins: int = FREE_SPACE_ANGLE_BINS,
    max_range: float = FREE_SPACE_MAX_RANGE_M,
) -> Polygon:
    """
    2D ray-cast "confirmed empty" polygon from the sensor's own local origin
    (0, 0) -- a simplified version of CAD's free-space idea
    (mvp/tools/polygon_space.py get_free_space/get_free_space_fast).

    For each of `angle_bins` angular slices around the sensor, this assumes
    the LiDAR swept cleanly out to the nearest return in that direction (or
    `max_range` if nothing was seen) -- everything inside that radius, in
    that direction, is "free": something else claiming to be there would
    contradict what this sensor actually observed.

    points: (N, 3+) array in the SENSOR'S OWN LOCAL frame (origin at the
        sensor). Pass the full point cloud (ground included) -- physically a
        LiDAR ray can't see past its first hit, ground or object, so ground
        returns still correctly bound how far "confirmed empty" extends in
        that direction. This is a deliberate simplification vs CAD's own
        lane-area / ground handling (see module docstring) -- CAD separates
        true open space from the ground plane's own footprint more
        carefully; this version doesn't, to stay simple.

    Returns a single star-shaped Polygon (valid by construction: vertices
    are strictly ordered by increasing angle around a full circle).
    """
    xy = points[:, :2]
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])  # (-pi, pi]

    bin_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)
    bin_idx = np.clip(np.digitize(theta, bin_edges) - 1, 0, angle_bins - 1)

    min_range_per_bin = np.full(angle_bins, max_range, dtype=float)
    np.minimum.at(min_range_per_bin, bin_idx, np.minimum(r, max_range))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vertices = np.column_stack(
        [min_range_per_bin * np.cos(bin_centers), min_range_per_bin * np.sin(bin_centers)]
    )
    return Polygon(vertices)


@dataclass
class FreeOccupiedViolation:
    a_occupied_in_b_free_m2: float  # CAV A claims occupied space CAV B's sensor swept empty
    b_occupied_in_a_free_m2: float  # CAV B claims occupied space CAV A's sensor swept empty
    total_violation_m2: float
    flagged: bool


def free_vs_occupied_violation(
    occupied_a: list[Polygon],
    free_a: Polygon,
    occupied_b: list[Polygon],
    free_b: Polygon,
) -> FreeOccupiedViolation:
    """
    CAD-style cross-check (simplified -- see module docstring): flag when one
    CAV's claimed-occupied space sits inside another CAV's claimed-free
    (swept-clean) space. Unlike conflicting_area()'s blob-vs-blob diff, this
    compares occupancy claims against actual sensor coverage, which is
    closer to CAD's real logic -- still missing CAD's lane-area filtering
    and its use of the detector's actual predicted boxes (P3 not built yet).
    """
    union_occupied_a = unary_union(occupied_a) if occupied_a else Polygon()
    union_occupied_b = unary_union(occupied_b) if occupied_b else Polygon()

    a_in_b_free = union_occupied_a.intersection(free_b).area
    b_in_a_free = union_occupied_b.intersection(free_a).area
    total = a_in_b_free + b_in_a_free

    return FreeOccupiedViolation(
        a_occupied_in_b_free_m2=a_in_b_free,
        b_occupied_in_a_free_m2=b_in_a_free,
        total_violation_m2=total,
        flagged=total > CONFLICT_AREA_THRESHOLD_M2,
    )


# ---------------------------------------------------------------------------
# Perception-occupancy check (added 17 Sep 2026, once P3's detector cache
# existed) -- the other half of CAD's two-part defence, previously an open
# TODO. See run_defence_check.py's module docstring for the full scope note
# on why this checks the FUSED detector's claims against each CAV's own free
# space, rather than CAV-vs-CAV detector claims the way CAD's own per-CAV
# early-fusion design does: P3's cache (built on OpenCOOD's LateFusionDataset
# + inference_late_fusion) only ever produces ONE fused detection set per
# frame -- there is no "which CAV's report produced this box" information
# left to cross-check CAV-against-CAV the way the occupancy-consistency half
# above does. This is a real, deliberate simplification, not an oversight.
# ---------------------------------------------------------------------------

DETECTION_SCORE_THRESHOLD = 0.5  # arbitrary, not calibrated -- flag as such


def box_corners_to_polygon(corners: np.ndarray) -> Polygon:
    """
    corners: (8, 3) array of box corner coordinates, OpenCOOD's standard
    8-corner box format. Returns the box's 2D footprint as the convex hull
    over the corners' (x, y) -- sufficient here since only footprint overlap
    with a free-space polygon matters, not the box's height.
    """
    xy = corners[:, :2]
    hull = MultiPoint(xy).convex_hull
    if isinstance(hull, Polygon):
        return hull
    return Polygon()  # degenerate (e.g. collinear/duplicate corners) -> zero-area


def detections_to_polygons(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    score_threshold: float = DETECTION_SCORE_THRESHOLD,
) -> list[Polygon]:
    """
    pred_boxes: (N, 8, 3) predicted box corners, P3's cache format.
    pred_scores: (N,) confidence scores, same order as pred_boxes.
    Returns footprint polygons for every box at or above score_threshold.
    """
    polygons = []
    for box, score in zip(pred_boxes, pred_scores):
        if score < score_threshold:
            continue
        poly = box_corners_to_polygon(np.asarray(box))
        if not poly.is_empty:
            polygons.append(poly)
    return polygons


@dataclass
class PerceptionOccupancyViolation:
    detected_in_free_m2: float
    per_source_m2: dict
    flagged: bool


def perception_occupancy_violation(
    detection_polygons: list[Polygon],
    free_space_by_source: dict,
) -> PerceptionOccupancyViolation:
    """
    Flags when the (fused) detector's claimed-occupied boxes fall inside
    space that some individual CAV's own raw sensor swept as confirmed free
    -- a detection appearing where a CAV's own direct observation says
    nothing is there. free_space_by_source: {cav_id: free-space Polygon},
    one entry per CAV already aligned to the shared (ego) frame -- reuse the
    same free-space polygons the occupancy-consistency check already
    computed per CAV, don't recompute.
    """
    union_detected = unary_union(detection_polygons) if detection_polygons else Polygon()
    per_source = {}
    total = 0.0
    for source_id, free_poly in free_space_by_source.items():
        overlap = union_detected.intersection(free_poly).area
        per_source[source_id] = overlap
        total += overlap
    return PerceptionOccupancyViolation(
        detected_in_free_m2=total,
        per_source_m2=per_source,
        flagged=total > CONFLICT_AREA_THRESHOLD_M2,
    )
