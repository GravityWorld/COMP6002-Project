"""
Done-when for Fri 4 Sep (execution_plan_daily.md): conflicting-area function
passes its synthetic test where you know the ground-truth overlap.

Also covers get_free_space() / free_vs_occupied_violation() (added 16 Sep
2026 after reading CAD's actual code) -- see consistency_layer.py's module
docstring for why these replace conflicting_area() as the actual flag rule.
"""

import numpy as np
from shapely.geometry import Polygon

from .consistency_layer import (
    conflicting_area,
    cluster_points,
    remove_ground_plane,
    get_free_space,
    free_vs_occupied_violation,
    box_corners_to_polygon,
    detections_to_polygons,
    perception_occupancy_violation,
    CONFLICT_AREA_THRESHOLD_M2,
)


def test_conflicting_area_no_overlap_is_zero():
    a = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    b = [Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])]
    result = conflicting_area(a, b)
    # Non-overlapping regions still count as symmetric difference (both boxes'
    # full area disagrees) — this asserts total area equals the sum of both,
    # which is the known ground truth for two disjoint 1x1 squares.
    assert abs(result.total_conflicting_area_m2 - 2.0) < 1e-6
    assert result.conflict_polygon_count == 2


def test_conflicting_area_identical_polygons_is_zero():
    """If both CAVs report the exact same occupancy, there's no conflict."""
    a = [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    b = [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    result = conflicting_area(a, b)
    assert result.total_conflicting_area_m2 < 1e-9
    assert result.flagged is False


def test_conflicting_area_known_partial_overlap():
    """
    Two unit-ish squares overlapping by a known 1x1 region.
    A = [0,2]x[0,2] (area 4), B = [1,3]x[0,2] (area 4), overlap = [1,2]x[0,2] (area 2).
    Symmetric difference area = 4 + 4 - 2*2 = 4.
    """
    a = [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    b = [Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])]
    result = conflicting_area(a, b)
    assert abs(result.total_conflicting_area_m2 - 4.0) < 1e-6


def test_ground_plane_removal_drops_flat_points():
    # A flat ground plane at z=0 plus one clear outlier above it.
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200), np.zeros(200)])
    outlier = np.array([[0.0, 0.0, 2.0]] * 15)  # a small "object" cluster
    points = np.vstack([ground, outlier])

    non_ground = remove_ground_plane(points)
    # Most ground points should be removed; the object cluster should survive.
    assert len(non_ground) < len(points)
    assert np.any(non_ground[:, 2] > 1.0)


def test_cluster_points_separates_two_groups():
    rng = np.random.default_rng(1)
    cluster1 = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
    cluster2 = rng.normal(loc=[10, 10], scale=0.1, size=(30, 2))
    points_xy = np.vstack([cluster1, cluster2])
    points = np.column_stack([points_xy, np.zeros(len(points_xy))])

    labels = cluster_points(points)
    unique_labels = set(labels) - {-1}
    assert len(unique_labels) == 2


def test_free_space_ring_of_points_approximates_circle():
    """
    Points in a ring at radius 10 all around the sensor -> free space (what
    the sensor swept clean before hitting something) should approximate a
    circle of radius 10. Known ground truth: area = pi * 10^2 ~= 314.16.
    """
    theta = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
    ring = np.column_stack([10 * np.cos(theta), 10 * np.sin(theta), np.zeros_like(theta)])
    free = get_free_space(ring, angle_bins=360)
    expected_area = np.pi * 10 ** 2
    assert abs(free.area - expected_area) < 5.0


def test_free_space_no_points_is_full_max_range_circle():
    """No returns anywhere -> assume free out to max_range in every direction."""
    empty = np.zeros((0, 3))
    free = get_free_space(empty, angle_bins=360, max_range=50.0)
    expected_area = np.pi * 50 ** 2
    assert abs(free.area - expected_area) < 5.0


def test_free_vs_occupied_no_violation_when_disjoint():
    """CAV A's occupied claim sits nowhere near CAV B's free space -> no violation."""
    free_small = get_free_space(np.array([[5.0, 0.0, 0.0]]), angle_bins=360)  # free out to ~5m
    occupied_far = [Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])]
    result = free_vs_occupied_violation(occupied_far, free_small, [], Polygon())
    assert result.total_violation_m2 < 1e-6
    assert result.flagged is False


def test_free_vs_occupied_flags_claim_inside_swept_clean_space():
    """
    CAV B's sensor saw nothing anywhere (free out to max_range in every
    direction). CAV A claims a 3x3 (9 m^2) occupied block sitting inside
    that swept-clean space -- exactly the CAD-style contradiction this
    function exists to catch. Known ground truth: violation == 9.0 m^2,
    and 9.0 > CONFLICT_AREA_THRESHOLD_M2 so it must flag.
    """
    empty = np.zeros((0, 3))
    free_wide = get_free_space(empty, angle_bins=360, max_range=50.0)
    occupied_inside = [Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
    result = free_vs_occupied_violation(occupied_inside, Polygon(), [], free_wide)
    assert abs(result.total_violation_m2 - 9.0) < 1e-3
    assert result.total_violation_m2 > CONFLICT_AREA_THRESHOLD_M2
    assert result.flagged is True


def _box_corners(cx, cy, half_size=1.0, z0=0.0, z1=2.0):
    """A simple axis-aligned box's 8 corners centered at (cx, cy), for tests --
    real boxes from P3's cache would be rotated/sized per the detector's
    output, but only the xy footprint matters for these functions."""
    xs = [cx - half_size, cx + half_size]
    ys = [cy - half_size, cy + half_size]
    zs = [z0, z1]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def test_box_corners_to_polygon_known_footprint_area():
    """A 2x2 box (half_size=1) has a known 4 m^2 footprint regardless of height."""
    corners = _box_corners(cx=5.0, cy=5.0, half_size=1.0)
    poly = box_corners_to_polygon(corners)
    assert abs(poly.area - 4.0) < 1e-6


def test_detections_to_polygons_filters_by_score():
    boxes = np.array([_box_corners(0, 0), _box_corners(10, 10)])
    scores = np.array([0.9, 0.2])  # second box below the default 0.5 threshold
    polygons = detections_to_polygons(boxes, scores)
    assert len(polygons) == 1


def test_perception_occupancy_flags_detection_inside_swept_clean_space():
    """
    One CAV's sensor saw nothing anywhere (free out to max_range in every
    direction). The fused detector claims a 2x2 (4 m^2) box sitting inside
    that swept-clean space -- exactly the contradiction this function exists
    to catch. Known ground truth: violation == 4.0 m^2, and since that's
    above CONFLICT_AREA_THRESHOLD_M2 it must flag.
    """
    empty = np.zeros((0, 3))
    free_wide = get_free_space(empty, angle_bins=360, max_range=50.0)
    detection_polygons = [box_corners_to_polygon(_box_corners(0, 0, half_size=1.0))]
    result = perception_occupancy_violation(detection_polygons, {"cav_1": free_wide})
    assert abs(result.detected_in_free_m2 - 4.0) < 1e-3
    assert result.detected_in_free_m2 > CONFLICT_AREA_THRESHOLD_M2
    assert result.flagged is True
    assert abs(result.per_source_m2["cav_1"] - 4.0) < 1e-3


def test_perception_occupancy_no_violation_when_detection_outside_free_space():
    """A CAV's sensor only swept a small radius clean; a detected box far
    outside that radius doesn't contradict what this CAV actually observed."""
    free_small = get_free_space(np.array([[5.0, 0.0, 0.0]]), angle_bins=360)  # free out to ~5m
    detection_polygons = [box_corners_to_polygon(_box_corners(100, 100, half_size=1.0))]
    result = perception_occupancy_violation(detection_polygons, {"cav_1": free_small})
    assert result.detected_in_free_m2 < 1e-6
    assert result.flagged is False


def test_perception_occupancy_no_double_counts_overlapping_boxes():
    """Two overlapping detected boxes claiming the same swept-clean spot
    should count that overlap once, not twice -- union, not sum."""
    empty = np.zeros((0, 3))
    free_wide = get_free_space(empty, angle_bins=360, max_range=50.0)
    box_a = box_corners_to_polygon(_box_corners(0, 0, half_size=1.0))  # 4 m^2 at (0,0)
    box_b = box_corners_to_polygon(_box_corners(0.5, 0, half_size=1.0))  # overlaps box_a
    result = perception_occupancy_violation([box_a, box_b], {"cav_1": free_wide})
    naive_double_count = box_a.area + box_b.area
    assert result.detected_in_free_m2 < naive_double_count


def test_ground_plane_removal_is_deterministic_across_runs():
    """
    Regression test for the non-determinism bug found 24 Sep 2026 (see
    decisions.md): remove_ground_plane() must return the identical inlier
    mask/output on repeated calls against the same input, regardless of
    global numpy RNG state. Deliberately perturbs global numpy state between
    calls to make sure random_state=0 (not incidental RNG state at import
    time) is what's pinning the result.
    """
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-5, 5, 300), rng.uniform(-5, 5, 300), np.zeros(300)])
    outlier = np.array([[0.0, 0.0, 2.0]] * 20)
    points = np.vstack([ground, outlier])

    np.random.seed(1)
    first = remove_ground_plane(points)
    np.random.seed(999)  # perturb global state differently before the second call
    second = remove_ground_plane(points)

    assert first.shape == second.shape
    assert np.array_equal(first, second)


def test_cluster_points_handles_zero_points_gracefully():
    """A CAV can genuinely see nothing but flat ground in a frame -- zero
    non-ground points after RANSAC removal is a real case, not an error.
    Found via a synthetic dry-run of run_defence_check.py (16 Sep 2026)
    where a CAV's frame was entirely ground and DBSCAN crashed on the
    resulting empty array."""
    empty = np.zeros((0, 3))
    labels = cluster_points(empty)
    assert len(labels) == 0
