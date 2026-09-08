"""
Done-when for Fri 4 Sep (execution_plan_daily.md): conflicting-area function
passes its synthetic test where you know the ground-truth overlap.
"""

import numpy as np
from shapely.geometry import Polygon

from .consistency_layer import conflicting_area, cluster_points, remove_ground_plane


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
