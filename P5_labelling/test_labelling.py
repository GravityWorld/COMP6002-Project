"""
P5 tests. Condition-labelling tests implemented 17 Sep 2026 alongside
labelling.py; cluster-correction tests added 25 Sep 2026 alongside the new
oracle-labelling functions (see labelling.py's module docstring).
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from .labelling import (
    label_condition,
    inject_synthetic_attack,
    GHOST_OBJECT_AREA_M2_RANGE,
    label_point_vehicle_membership,
    build_cluster_repair_target,
    pairwise_same_object_affinity,
    BACKGROUND_VEHICLE_ID,
    build_candidate_fragments,
    fragment_majority_vehicle_id,
)


def _box(cx, cy, half_size=1.0):
    """Box footprint in the winding order points_in_rotated_box_xy expects
    (matches create_bbx's corner order 0,1,2,3 -- see labelling.py)."""
    return np.array([
        [cx + half_size, cy - half_size],
        [cx + half_size, cy + half_size],
        [cx - half_size, cy + half_size],
        [cx - half_size, cy - half_size],
    ])


def test_label_condition_clean_arm_a():
    result = label_condition(flag_decision=False, degradation_meta=None, attack_meta=None)
    assert result.arm == "A"
    assert result.y_cause == "none"
    assert result.y_flag is False
    assert result.severity == 0


def test_label_condition_degraded_arm_b():
    result = label_condition(
        flag_decision=True,
        degradation_meta={"type": "fog", "severity": 2},
        attack_meta=None,
    )
    assert result.arm == "B"
    assert result.y_cause == "noise"
    assert result.degradation_type == "fog"
    assert result.severity == 2
    assert result.y_flag is True


def test_label_condition_attack_arm_c():
    result = label_condition(
        flag_decision=True,
        degradation_meta=None,
        attack_meta={"injected": True, "attacker_cav": "1854"},
    )
    assert result.arm == "C"
    assert result.y_cause == "attack"


def test_label_condition_attack_takes_precedence_over_degradation():
    """If both were somehow injected (a caller bug -- arms shouldn't
    overlap), attack must win so it isn't silently miscounted as benign."""
    result = label_condition(
        flag_decision=True,
        degradation_meta={"type": "compression", "severity": 1},
        attack_meta={"injected": True},
    )
    assert result.arm == "C"
    assert result.y_cause == "attack"


def test_label_condition_degradation_meta_with_zero_severity_is_not_degraded():
    result = label_condition(
        flag_decision=False, degradation_meta={"type": "fog", "severity": 0}, attack_meta=None
    )
    assert result.arm == "A"
    assert result.y_cause == "none"


def test_inject_synthetic_attack_places_ghost_inside_free_space():
    # A wide open square of confirmed-free space, 50x50 centered at origin.
    free_space = Polygon([(-25, -25), (25, -25), (25, 25), (-25, 25)])
    ghost = inject_synthetic_attack(free_space, seed=0)
    assert free_space.contains(ghost)
    assert GHOST_OBJECT_AREA_M2_RANGE[0] * 0.99 <= ghost.area <= GHOST_OBJECT_AREA_M2_RANGE[1] * 1.01


def test_inject_synthetic_attack_deterministic_with_seed():
    free_space = Polygon([(-25, -25), (25, -25), (25, 25), (-25, 25)])
    ghost_a = inject_synthetic_attack(free_space, seed=42)
    ghost_b = inject_synthetic_attack(free_space, seed=42)
    assert ghost_a.equals(ghost_b)


def test_inject_synthetic_attack_respects_explicit_area():
    free_space = Polygon([(-25, -25), (25, -25), (25, 25), (-25, 25)])
    ghost = inject_synthetic_attack(free_space, seed=1, area_m2=4.0)
    assert abs(ghost.area - 4.0) < 1e-6


def test_inject_synthetic_attack_empty_free_space_raises():
    with pytest.raises(ValueError):
        inject_synthetic_attack(Polygon(), seed=0)


def test_inject_synthetic_attack_too_small_free_space_raises():
    tiny = Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])  # 0.25 m^2, smaller than any ghost
    with pytest.raises(ValueError):
        inject_synthetic_attack(tiny, seed=0, area_m2=5.0)


# --- Cluster-correction (oracle repair target) tests ---
# Each test builds a small synthetic scene with KNOWN ground truth (which
# vehicle owns which points, and what DBSCAN allegedly did), so the correct
# repaired_cluster_id / affinity is known in advance -- not just "plausible".


def test_label_point_vehicle_membership_assigns_correct_vehicle():
    # Two well-separated 1x1 boxes; 3 points land inside vehicle 100's box,
    # 2 inside vehicle 200's box, 1 point is background (inside neither).
    points = np.array([
        [0.0, 0.0], [0.2, -0.3], [-0.4, 0.1],   # inside vehicle 100's box (center 0,0)
        [10.0, 10.0], [10.3, 9.8],               # inside vehicle 200's box (center 10,10)
        [50.0, 50.0],                             # background
    ])
    gt_boxes = {100: _box(0, 0), 200: _box(10, 10)}
    vehicle_id = label_point_vehicle_membership(points, gt_boxes)
    assert list(vehicle_id) == [100, 100, 100, 200, 200, BACKGROUND_VEHICLE_ID]


def test_label_point_vehicle_membership_background_when_no_gt_boxes():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    vehicle_id = label_point_vehicle_membership(points, gt_boxes={})
    assert np.all(vehicle_id == BACKGROUND_VEHICLE_ID)


def test_build_cluster_repair_target_recovers_ground_swallowed():
    """A vehicle's points that DBSCAN called noise (-1), but that really do
    belong to a GT vehicle, must be recovered into their own repaired
    cluster -- the GROUND_SWALLOWED repair."""
    raw_cluster_id = np.array([-1, -1, -1])       # DBSCAN found nothing -- all noise
    point_vehicle_id = np.array([500, 500, 500])  # but they're all really vehicle 500
    target = build_cluster_repair_target(raw_cluster_id, point_vehicle_id)
    assert np.all(target.repaired_cluster_id == 500)
    # Confirms this WAS a repair, not a no-op: raw said noise, repaired says a real cluster.
    assert np.all(target.raw_cluster_id == -1)


def test_build_cluster_repair_target_splits_merged_cluster():
    """One raw DBSCAN cluster (id 7) actually contains two distinct real
    vehicles -- the MERGED case. The repair target must split them."""
    raw_cluster_id = np.array([7, 7, 7, 7])
    point_vehicle_id = np.array([1, 1, 2, 2])  # first half vehicle 1, second half vehicle 2
    target = build_cluster_repair_target(raw_cluster_id, point_vehicle_id)
    assert list(target.repaired_cluster_id) == [1, 1, 2, 2]
    assert len(set(target.repaired_cluster_id.tolist())) == 2  # was 1 raw cluster, now 2


def test_build_cluster_repair_target_merges_split_clusters():
    """One real vehicle's points were split by DBSCAN into two raw cluster
    ids (3 and 9) -- the SPLIT case. The repair target must merge them."""
    raw_cluster_id = np.array([3, 3, 9, 9])
    point_vehicle_id = np.array([42, 42, 42, 42])  # all really the same vehicle
    target = build_cluster_repair_target(raw_cluster_id, point_vehicle_id)
    assert np.all(target.repaired_cluster_id == 42)
    assert len(set(raw_cluster_id.tolist())) == 2  # confirms this WAS a split to begin with


def test_build_cluster_repair_target_background_points_never_get_a_cluster():
    """Background points (not inside any GT vehicle box) stay unclustered
    in the repaired target regardless of what DBSCAN did with them --
    repairing static structure is out of scope for this labelling task."""
    raw_cluster_id = np.array([5, 5, -1])  # DBSCAN grouped two background points together
    point_vehicle_id = np.array([BACKGROUND_VEHICLE_ID, BACKGROUND_VEHICLE_ID, BACKGROUND_VEHICLE_ID])
    target = build_cluster_repair_target(raw_cluster_id, point_vehicle_id)
    assert np.all(target.repaired_cluster_id == BACKGROUND_VEHICLE_ID)


def test_build_cluster_repair_target_shape_mismatch_raises():
    with pytest.raises(ValueError):
        build_cluster_repair_target(np.array([1, 2, 3]), np.array([1, 2]))


def test_pairwise_same_object_affinity_true_for_same_vehicle():
    point_vehicle_id = np.array([42, 42, 7])
    result = pairwise_same_object_affinity(point_vehicle_id, i=np.array([0]), j=np.array([1]))
    assert result[0] == True  # noqa: E712 -- explicit bool check reads clearer here


def test_pairwise_same_object_affinity_false_for_different_vehicles():
    point_vehicle_id = np.array([42, 7])
    result = pairwise_same_object_affinity(point_vehicle_id, i=np.array([0]), j=np.array([1]))
    assert result[0] == False  # noqa: E712


def test_pairwise_same_object_affinity_false_for_background_pair():
    """Two background points are never asserted to have affinity, even
    though they'd trivially compare equal (-1 == -1) -- see
    pairwise_same_object_affinity's docstring."""
    point_vehicle_id = np.array([BACKGROUND_VEHICLE_ID, BACKGROUND_VEHICLE_ID])
    result = pairwise_same_object_affinity(point_vehicle_id, i=np.array([0]), j=np.array([1]))
    assert result[0] == False  # noqa: E712


def test_pairwise_same_object_affinity_vectorized_over_multiple_pairs():
    point_vehicle_id = np.array([1, 1, 2, BACKGROUND_VEHICLE_ID])
    i = np.array([0, 0, 1, 3])
    j = np.array([1, 2, 2, 0])
    result = pairwise_same_object_affinity(point_vehicle_id, i, j)
    assert list(result) == [True, False, False, False]


# --- Candidate fragments (added 26 Sep 2026) ---
# Each test again builds a small scene with KNOWN ground truth so the
# correct fragment_id / majority vehicle is known in advance.

NOISE_EPS_M = 0.75  # matches consistency_layer.DBSCAN_EPS_M -- see
                     # build_candidate_fragments' docstring on why the real
                     # script passes that value explicitly.


def test_build_candidate_fragments_leaves_raw_clusters_unchanged():
    raw_cluster_id = np.array([0, 0, 0, 1, 1])
    points_xy = np.array([[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5]])
    fragment_id = build_candidate_fragments(points_xy, raw_cluster_id, noise_eps_m=NOISE_EPS_M)
    assert list(fragment_id) == [0, 0, 0, 1, 1]


def test_build_candidate_fragments_recovers_dense_noise_group():
    """A tight group of noise points (e.g. a GROUND_SWALLOWED vehicle's
    points, all DBSCAN noise under the real pipeline's min_samples=10) must
    get a real fragment id under the looser noise-only pass, not stay -1."""
    raw_cluster_id = np.array([-1, -1, -1, -1])
    points_xy = np.array([[50.0, 50.0], [50.1, 50.0], [50.0, 50.1], [50.1, 50.1]])
    fragment_id = build_candidate_fragments(points_xy, raw_cluster_id, noise_eps_m=NOISE_EPS_M, noise_min_samples=3)
    assert np.all(fragment_id != -1)
    assert len(set(fragment_id.tolist())) == 1  # all 4 points recovered into ONE fragment


def test_build_candidate_fragments_leaves_isolated_noise_unfragmented():
    """Noise points nowhere near enough to each other don't manufacture a
    fragment -- a single stray point stays unclustered even under the
    loosened pass."""
    raw_cluster_id = np.array([-1, -1, -1])
    points_xy = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])
    fragment_id = build_candidate_fragments(points_xy, raw_cluster_id, noise_eps_m=NOISE_EPS_M, noise_min_samples=3)
    assert np.all(fragment_id == -1)


def test_build_candidate_fragments_recovered_ids_dont_collide_with_raw_ids():
    """Recovered noise fragments must get ids past raw_cluster_id's own max
    -- otherwise a recovered fragment could be silently confused with a real
    raw DBSCAN cluster downstream."""
    raw_cluster_id = np.array([0, 0, 0, -1, -1, -1, -1])
    points_xy = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0],                 # raw cluster 0 (positions irrelevant)
        [50.0, 50.0], [50.1, 50.0], [50.0, 50.1], [50.1, 50.1],  # tight noise group
    ])
    fragment_id = build_candidate_fragments(points_xy, raw_cluster_id, noise_eps_m=NOISE_EPS_M, noise_min_samples=3)
    assert list(fragment_id[:3]) == [0, 0, 0]
    recovered_ids = set(fragment_id[3:].tolist())
    assert recovered_ids == {1}  # offset past raw max (0) -- not 0 itself
    assert 0 not in fragment_id[3:]


def test_fragment_majority_vehicle_id_pure_fragments():
    fragment_id = np.array([0, 0, 0, 1, 1])
    point_vehicle_id = np.array([7, 7, 7, 9, 9])
    result = fragment_majority_vehicle_id(fragment_id, point_vehicle_id)
    assert result == {0: 7, 1: 9}


def test_fragment_majority_vehicle_id_all_background_fragment():
    fragment_id = np.array([0, 0])
    point_vehicle_id = np.array([BACKGROUND_VEHICLE_ID, BACKGROUND_VEHICLE_ID])
    result = fragment_majority_vehicle_id(fragment_id, point_vehicle_id)
    assert result == {0: BACKGROUND_VEHICLE_ID}


def test_fragment_majority_vehicle_id_plurality_among_non_background():
    """A fragment that isn't perfectly pure is assigned by plurality among
    its non-background points, not thrown out."""
    fragment_id = np.array([0, 0, 0, 0])
    point_vehicle_id = np.array([1, 1, 1, 2])  # 3 votes for vehicle 1, 1 for vehicle 2
    result = fragment_majority_vehicle_id(fragment_id, point_vehicle_id)
    assert result == {0: 1}


def test_fragment_majority_vehicle_id_skips_unfragmented_sentinel():
    fragment_id = np.array([-1, -1, 0])
    point_vehicle_id = np.array([5, 5, 5])
    result = fragment_majority_vehicle_id(fragment_id, point_vehicle_id)
    assert -1 not in result
    assert result == {0: 5}


def test_fragment_level_pairwise_affinity_recovers_split_vehicle():
    """End-to-end contract check: one real vehicle (id 42) SPLIT by DBSCAN
    into two raw clusters (3 and 9) -- fragment-level pairwise affinity
    between those two fragments must come back True, which is exactly the
    signal the correction head is supposed to learn to predict."""
    raw_cluster_id = np.array([3, 3, 9, 9])
    points_xy = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    point_vehicle_id = np.array([42, 42, 42, 42])

    fragment_id = build_candidate_fragments(points_xy, raw_cluster_id, noise_eps_m=NOISE_EPS_M)
    assert list(fragment_id) == [3, 3, 9, 9]  # no noise present, so unchanged

    frag_vehicle = fragment_majority_vehicle_id(fragment_id, point_vehicle_id)
    assert frag_vehicle == {3: 42, 9: 42}

    frag_ids_sorted = sorted(frag_vehicle)
    vehicle_arr = np.array([frag_vehicle[f] for f in frag_ids_sorted])
    result = pairwise_same_object_affinity(vehicle_arr, i=np.array([0]), j=np.array([1]))
    assert result[0] == True  # noqa: E712 -- these two fragments ARE the same real vehicle
