"""
P6 tests. Implemented 17 Sep 2026 alongside features.py.
Extended Thu 1 Oct 2026 for compute_sensing_profile_from_repair_labels() and
the sensing_profile argument to build_vehicle_partition().
"""

import numpy as np
import pandas as pd
import pytest

from .features import (
    build_vehicle_partition,
    build_edge_list_static,
    build_edge_list_traffic,
    build_scenario_split,
    compute_sensing_profile_from_repair_labels,
)


def _toy_manifest():
    return pd.DataFrame(
        [
            {"scenario_id": "sA", "frame_id": "0001", "ego_id": "1", "cav_ids": ["1", "2", "3"], "n_cav": 3},
            {"scenario_id": "sA", "frame_id": "0002", "ego_id": "1", "cav_ids": ["1", "2", "3"], "n_cav": 3},
            {"scenario_id": "sB", "frame_id": "0001", "ego_id": "9", "cav_ids": ["9", "10"], "n_cav": 2},
        ]
    )


def test_build_vehicle_partition_same_key_maps_to_same_vehicle():
    manifest = _toy_manifest()
    part = build_vehicle_partition(manifest, n_vehicles=10)
    # All rows for scenario sA (no degradation_meta -> same partition_key
    # "sA|none|0" for every sA row) must land on the same vehicle_id.
    sa_vehicle_ids = part[part["scenario_id"] == "sA"]["vehicle_id"].unique()
    assert len(sa_vehicle_ids) == 1


def test_build_vehicle_partition_deterministic():
    manifest = _toy_manifest()
    part1 = build_vehicle_partition(manifest, n_vehicles=10)
    part2 = build_vehicle_partition(manifest, n_vehicles=10)
    assert part1["vehicle_id"].tolist() == part2["vehicle_id"].tolist()


def test_build_vehicle_partition_respects_degradation_meta():
    manifest = _toy_manifest()
    # Split sA's own 3 CAVs across two different degradation conditions --
    # they should now NOT all collapse to one vehicle_id, since the
    # partition key now differs.
    degradation_meta = pd.DataFrame(
        [
            {"scenario_id": "sA", "cav_id": "1", "degradation_type": "fog", "severity": 1},
            {"scenario_id": "sA", "cav_id": "2", "degradation_type": "fog", "severity": 2},
            {"scenario_id": "sA", "cav_id": "3", "degradation_type": "compression", "severity": 1},
        ]
    )
    part = build_vehicle_partition(manifest, n_vehicles=50, degradation_meta=degradation_meta)
    sa_vehicle_ids = part[part["scenario_id"] == "sA"]["vehicle_id"].unique()
    # 3 distinct partition keys -> very likely (though not guaranteed with a
    # small bucket count) 3 distinct buckets at n_vehicles=50; check the
    # partition_key column directly instead, which IS guaranteed distinct.
    sa_keys = part[part["scenario_id"] == "sA"]["partition_key"].unique()
    assert len(sa_keys) == 3


def test_build_edge_list_static_is_fully_connected():
    vehicle_ids = [0, 1, 2, 3]
    edges = build_edge_list_static(vehicle_ids)
    # Fully connected undirected graph on 4 nodes has 4*3/2 = 6 edges.
    assert len(edges) == 6
    # Every unordered pair must be present exactly once.
    pairs = set(zip(edges["vehicle_a"], edges["vehicle_b"]))
    expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert pairs == expected


def test_build_edge_list_static_no_self_loops():
    edges = build_edge_list_static([5, 5, 5, 7])  # duplicate ids collapse via set()
    assert not any(a == b for a, b in zip(edges["vehicle_a"], edges["vehicle_b"]))


def test_build_edge_list_traffic_only_connects_co_occurring_vehicles():
    manifest = _toy_manifest()
    # Force each CAV onto its own distinct vehicle_id via degradation_meta,
    # so co-occurrence in a frame is unambiguous.
    degradation_meta = pd.DataFrame(
        [
            {"scenario_id": "sA", "cav_id": "1", "degradation_type": "fog", "severity": 1},
            {"scenario_id": "sA", "cav_id": "2", "degradation_type": "fog", "severity": 2},
            {"scenario_id": "sA", "cav_id": "3", "degradation_type": "compression", "severity": 1},
            {"scenario_id": "sB", "cav_id": "9", "degradation_type": "none", "severity": 0},
            {"scenario_id": "sB", "cav_id": "10", "degradation_type": "fog", "severity": 1},
        ]
    )
    part = build_vehicle_partition(manifest, n_vehicles=50, degradation_meta=degradation_meta)
    traffic = build_edge_list_traffic(part)

    # sA/0001 has 3 co-occurring CAVs -> 3 pairs; sA/0002 the same 3 -> 3
    # pairs; sB/0001 has 2 -> 1 pair. Total rows = 3 + 3 + 1 = 7.
    assert len(traffic) == 7
    # No round should ever connect a vehicle from sA to a vehicle from sB.
    sa_vids = set(part[part["scenario_id"] == "sA"]["vehicle_id"])
    sb_vids = set(part[part["scenario_id"] == "sB"]["vehicle_id"])
    for _, row in traffic.iterrows():
        both_in_sa = row["vehicle_a"] in sa_vids and row["vehicle_b"] in sa_vids
        both_in_sb = row["vehicle_a"] in sb_vids and row["vehicle_b"] in sb_vids
        assert both_in_sa or both_in_sb


def test_build_edge_list_traffic_missing_columns_raises():
    with pytest.raises(ValueError):
        build_edge_list_traffic(pd.DataFrame({"a": [1]}))


def test_build_scenario_split_no_scenario_in_two_splits():
    manifest = pd.DataFrame(
        {"scenario_id": [f"s{i}" for i in range(10)] * 2}
    )  # 10 distinct scenarios, each appearing twice (multiple frames)
    split = build_scenario_split(manifest, val_frac=0.2, test_frac=0.2, seed=0)
    all_assigned = split.train_scenarios | split.val_scenarios | split.test_scenarios
    # Every scenario assigned exactly once across the three sets.
    assert len(split.train_scenarios & split.val_scenarios) == 0
    assert len(split.train_scenarios & split.test_scenarios) == 0
    assert len(split.val_scenarios & split.test_scenarios) == 0
    assert all_assigned == set(manifest["scenario_id"].unique())


def test_build_scenario_split_degenerate_two_scenario_case_documented_behavior():
    """With only 2 scenarios (this project's actual current data), a 20%/20%
    split rounds both val and test to 0 -- everything lands in train. This
    is the degenerate case the module docstring flags; the test locks in
    that it fails predictably (empty splits) rather than silently
    fabricating scenarios or crashing."""
    manifest = pd.DataFrame({"scenario_id": ["2021_09_03_20_08_37", "2021_09_03_22_43_57"]})
    split = build_scenario_split(manifest, val_frac=0.2, test_frac=0.2, seed=0)
    assert len(split.val_scenarios) == 0
    assert len(split.test_scenarios) == 0
    assert len(split.train_scenarios) == 2


def test_build_scenario_split_invalid_fracs_raise():
    manifest = pd.DataFrame({"scenario_id": ["s1", "s2"]})
    with pytest.raises(ValueError):
        build_scenario_split(manifest, val_frac=0.6, test_frac=0.6)


# --- Thu 1 Oct 2026 additions: local range/density partition key ---


def _write_fake_points_npz(labels_dir, scenario_id, frame_id, cav_id, ranges, clustered_flags):
    """Builds a minimal _points.npz matching run_cluster_repair_labelling.py's
    real schema, just enough for compute_sensing_profile_from_repair_labels to
    read (xyz, raw_cluster_id) -- ranges/clustered_flags are parallel lists,
    one point each, placed along +x at the given range so
    norm(xyz[:, :2]) == range exactly."""
    scenario_dir = labels_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    n = len(ranges)
    xyz = np.zeros((n, 3))
    xyz[:, 0] = ranges  # place each point at (range, 0, 0) -> norm(xy) == range
    raw_cluster_id = np.array([0 if flag else -1 for flag in clustered_flags])
    np.savez_compressed(
        scenario_dir / f"{frame_id}_{cav_id}_points.npz",
        xyz=xyz,
        raw_cluster_id=raw_cluster_id,
        point_vehicle_id=np.zeros(n, dtype=int),
        repaired_cluster_id=np.zeros(n, dtype=int),
        fragment_id=raw_cluster_id,
    )


def test_compute_sensing_profile_from_repair_labels_basic(tmp_path):
    labels_dir = tmp_path / "repair_labels"
    # A "near, dense" CAV: short ranges, mostly clustered.
    _write_fake_points_npz(
        labels_dir, "sA", "0001", "1",
        ranges=[10.0, 12.0, 11.0, 9.0], clustered_flags=[True, True, True, True],
    )
    # A "far, sparse" CAV: long ranges, mostly noise -- the GROUND_SWALLOWED-
    # prone profile the audit's own finding describes.
    _write_fake_points_npz(
        labels_dir, "sA", "0001", "2",
        ranges=[60.0, 55.0, 58.0, 62.0], clustered_flags=[False, False, False, True],
    )

    profile = compute_sensing_profile_from_repair_labels(labels_dir, n_buckets=2)

    assert set(profile.columns) >= {
        "scenario_id", "frame_id", "cav_id", "mean_range_m", "clustered_fraction", "profile_bucket",
    }
    assert len(profile) == 2

    near = profile[profile["cav_id"] == "1"].iloc[0]
    far = profile[profile["cav_id"] == "2"].iloc[0]
    assert near["mean_range_m"] < far["mean_range_m"]
    assert near["clustered_fraction"] > far["clustered_fraction"]
    # With 2 rows and 2 buckets, the near/far CAVs must land in different
    # profile_bucket strings -- that's the entire point of the partition key.
    assert near["profile_bucket"] != far["profile_bucket"]


def test_compute_sensing_profile_from_repair_labels_empty_dir_returns_empty_frame(tmp_path):
    profile = compute_sensing_profile_from_repair_labels(tmp_path / "does_not_exist")
    assert len(profile) == 0
    assert "profile_bucket" in profile.columns


def test_build_vehicle_partition_sensing_profile_changes_partition_key():
    manifest = _toy_manifest()
    sensing_profile = pd.DataFrame(
        [
            {"scenario_id": "sA", "frame_id": "0001", "cav_id": "1", "profile_bucket": "range0_density1"},
            {"scenario_id": "sA", "frame_id": "0002", "cav_id": "1", "profile_bucket": "range0_density1"},
            {"scenario_id": "sA", "frame_id": "0001", "cav_id": "2", "profile_bucket": "range1_density0"},
            {"scenario_id": "sA", "frame_id": "0002", "cav_id": "2", "profile_bucket": "range1_density0"},
            # cav "3" deliberately has no profile row -- must fall back to 'unknown'.
        ]
    )
    part = build_vehicle_partition(manifest, n_vehicles=50, sensing_profile=sensing_profile)

    cav1_keys = part[part["cav_id"] == "1"]["partition_key"].unique()
    cav2_keys = part[part["cav_id"] == "2"]["partition_key"].unique()
    cav3 = part[part["cav_id"] == "3"]
    assert len(cav1_keys) == 1 and len(cav2_keys) == 1
    assert cav1_keys[0] != cav2_keys[0]  # different profile_bucket -> different partition_key
    assert (cav3["profile_bucket"] == "unknown").all()


def test_build_vehicle_partition_without_sensing_profile_defaults_to_unknown():
    manifest = _toy_manifest()
    part = build_vehicle_partition(manifest, n_vehicles=10)
    assert (part["profile_bucket"] == "unknown").all()
