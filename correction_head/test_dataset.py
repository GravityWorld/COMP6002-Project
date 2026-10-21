"""Tests for correction_head/dataset.py. Pure numpy/pandas -- no torch
needed (see dataset.py's own try/except torch import), so these run in any
environment, including the cloud sandbox and the device-bridge's Linux VM
neither of which has torch or opencood installed."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from correction_head.dataset import (
    CANDIDATE_PAIR_RADIUS_M,
    FRAGMENT_AUX_DIM,
    PAIR_AUX_DIM,
    build_sample,
    build_samples_for_scenario,
    collate_batches,
    discover_frame_cav_pairs,
    load_degradation_config,
    load_scenario_crops,
    load_violation_lookup,
)

# dataset.py deliberately duplicates these constants from correction_head.model
# rather than importing them (see dataset.py's own comment) so this test file
# stays importable without torch too -- test that the duplication is honest.


def test_dataset_constants_match_model_when_torch_available():
    pytest.importorskip("torch")
    from correction_head import dataset as dataset_module

    dataset_module.assert_matches_model()


@pytest.fixture
def tmp_repo(tmp_path):
    """Builds a minimal fake repo layout: one scenario, one (frame, cav)
    with 4 fragments / some pairs, matching the REAL on-disk schemas
    confirmed by reading the actual files on the connected machine
    (fragments: fragment_id/centroid_x/centroid_y/n_points/vehicle_id/
    is_recovered_noise; pairs: fragment_i/fragment_j/centroid_distance_m/
    affinity_label; crops npz: frame_id/cav_id/fragment_id/crop/valid)."""
    scenario_id = "scenA"
    frame_id = "000100"
    cav_id = "111"

    repair_dir = tmp_path / "repair_labels" / scenario_id
    repair_dir.mkdir(parents=True)

    frags = pd.DataFrame(
        {
            "fragment_id": [0, 1, 2, 3],
            "centroid_x": [10.0, -5.0, 0.0, 20.0],
            "centroid_y": [0.0, 5.0, -3.0, 10.0],
            "n_points": [15, 82, 22, 5],
            "vehicle_id": [-1, 7, -1, 7],
            "is_recovered_noise": [False, False, True, False],
        }
    )
    frags.to_parquet(repair_dir / f"{frame_id}_{cav_id}_fragments.parquet")

    pairs = pd.DataFrame(
        {
            "fragment_i": [0, 1, 2],
            "fragment_j": [1, 3, 3],
            "centroid_distance_m": [3.9, 1.1, 2.0],
            "affinity_label": [False, True, False],
        }
    )
    pairs.to_parquet(repair_dir / f"{frame_id}_{cav_id}_pairs.parquet")

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    rng = np.random.default_rng(0)
    # fragment 3 deliberately invalid -- tests the valid=False drop path
    # and the "pair referencing a dropped fragment is dropped too" rule.
    np.savez(
        crops_dir / f"{scenario_id}_A_crops.npz",
        frame_id=np.array([frame_id] * 4),
        cav_id=np.array([cav_id] * 4),
        fragment_id=np.array([0, 1, 2, 3], dtype=np.int32),
        crop=rng.standard_normal((4, 384, 7, 7)).astype(np.float32),
        valid=np.array([True, True, True, False]),
    )

    return {
        "root": tmp_path,
        "repair_labels_dir": tmp_path / "repair_labels",
        "crops_dir": crops_dir,
        "scenario_id": scenario_id,
        "frame_id": frame_id,
        "cav_id": cav_id,
    }


def test_discover_frame_cav_pairs(tmp_repo):
    pairs = discover_frame_cav_pairs(tmp_repo["repair_labels_dir"], tmp_repo["scenario_id"])
    assert pairs == [(tmp_repo["frame_id"], tmp_repo["cav_id"])]


def test_discover_frame_cav_pairs_missing_scenario(tmp_repo):
    assert discover_frame_cav_pairs(tmp_repo["repair_labels_dir"], "nope") == []


def test_load_scenario_crops_drops_invalid(tmp_repo):
    lookup, n_dropped = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    assert n_dropped == 1
    assert len(lookup) == 3
    key = (tmp_repo["frame_id"], tmp_repo["cav_id"], 0)
    assert key in lookup
    assert lookup[key].shape == (384, 7, 7)
    # the invalid fragment (id 3) must not appear
    assert (tmp_repo["frame_id"], tmp_repo["cav_id"], 3) not in lookup


def test_load_scenario_crops_missing_file_raises(tmp_repo):
    with pytest.raises(FileNotFoundError):
        load_scenario_crops(tmp_repo["crops_dir"], "no_such_scenario", arm="A")


def test_build_sample_shapes_and_values(tmp_repo):
    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
    )
    assert sample is not None
    # fragment 3 had no valid crop -> dropped, leaving 3 fragments
    assert sample.n_fragments == 3
    assert sample.crops.shape == (3, 384, 7, 7)
    assert sample.frag_aux.shape == (3, FRAGMENT_AUX_DIM)

    # pairs referencing fragment 3 (1-3, 2-3) must be dropped; only (0,1) survives
    assert sample.n_pairs == 1
    assert sample.pair_aux.shape == (1, PAIR_AUX_DIM)
    assert sample.labels.shape == (1,)
    assert sample.labels[0] == 0.0  # affinity_label was False for (0,1)

    # pair_index must be valid local indices into the 3 kept fragments
    assert sample.pair_index.max() < sample.n_fragments
    assert sample.pair_index.min() >= 0

    # arm A -> degradation_severity (aux index 5) must be exactly 0 for every fragment
    assert np.all(sample.frag_aux[:, 5] == 0.0)

    # is_recovered_noise (aux index 1) must match the source column for the
    # 3 surviving fragments (ids 0, 1, 2 -> False, False, True)
    np.testing.assert_array_equal(sample.frag_aux[:, 1], [0.0, 0.0, 1.0])

    # log1p_n_points (aux index 0) must match log1p(n_points) exactly
    np.testing.assert_allclose(sample.frag_aux[:, 0], np.log1p([15.0, 82.0, 22.0]))


def test_build_sample_pair_aux_normalisation(tmp_repo):
    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
    )
    # the surviving pair is (fragment_i=0, fragment_j=1), centroid_distance_m=3.9
    expected_dist_norm = 3.9 / CANDIDATE_PAIR_RADIUS_M
    assert sample.pair_aux[0, 0] == pytest.approx(expected_dist_norm, rel=1e-5)
    expected_abs_delta = abs(np.log1p(15.0) - np.log1p(82.0))
    assert sample.pair_aux[0, 1] == pytest.approx(expected_abs_delta, rel=1e-5)
    # no points.npz written in this fixture -> interposed_object_signal
    # (aux index 2) must zero-fill, same convention as a missing
    # violation_lookup entry or missing frag_extent_norm source file.
    assert sample.pair_aux[0, 2] == pytest.approx(0.0)


def test_build_sample_pair_aux_interposed_signal(tmp_repo):
    """Thu 8 Oct 2026 physics feature: a third fragment's real points
    sitting on the straight-line segment between the surviving pair's two
    centroids (fragment 0 at (10, 0), fragment 1 at (-5, 5), per tmp_repo)
    must be counted; points off that segment must not be."""
    points_path = (
        tmp_repo["repair_labels_dir"]
        / tmp_repo["scenario_id"]
        / f"{tmp_repo['frame_id']}_{tmp_repo['cav_id']}_points.npz"
    )
    # 3 points from fragment 99, placed near the midpoint of the (0, 1)
    # segment (exactly on it, and two with a small jitter well inside
    # buffer_radius_m=0.4) -- all three expected to count as "interposed".
    # Fragments 0/1's own points are included too, just to look like a
    # real file; they're excluded by construction (mask_other) regardless
    # of where they sit.
    np.savez(
        points_path,
        xyz=np.array(
            [
                [10.0, 0.0, 0.0],
                [10.1, 0.1, 0.0],
                [-5.0, 5.0, 0.0],
                [-4.9, 4.9, 0.0],
                [2.5, 2.5, 0.0],
                [2.6, 2.4, 0.0],
                [2.4, 2.6, 0.0],
            ]
        ),
        fragment_id=np.array([0, 0, 1, 1, 99, 99, 99], dtype=np.int32),
        point_vehicle_id=np.full(7, -1, dtype=np.int64),
    )

    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
    )
    assert sample.pair_aux[0, 2] == pytest.approx(np.log1p(3), rel=1e-5)


def test_build_sample_degradation_severity_arm_b(tmp_repo):
    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    degradation_config = {(tmp_repo["scenario_id"], tmp_repo["cav_id"]): {"type": "fog", "severity": 2, "seed": 1}}
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="B",
        degradation_config=degradation_config,
    )
    assert np.all(sample.frag_aux[:, 5] == 2.0)


def test_build_sample_no_matching_fragments_file_returns_none(tmp_repo):
    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    sample = build_sample(
        tmp_repo["scenario_id"],
        "999999",
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
    )
    assert sample is None


def test_build_sample_no_valid_crops_returns_none(tmp_repo):
    # all-invalid crop lookup for this (frame, cav) -> every fragment dropped -> None
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        {},  # empty lookup, as if crop caching never ran
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
    )
    assert sample is None


def test_build_samples_for_scenario_warns_when_empty(tmp_repo, recwarn):
    samples = build_samples_for_scenario(
        tmp_repo["scenario_id"],
        crops_dir=tmp_repo["crops_dir"],
        repair_labels_dir=tmp_repo["repair_labels_dir"],
        arm="A",
    )
    assert len(samples) == 1
    assert samples[0].frame_id == tmp_repo["frame_id"]


def test_collate_batches_offsets_pair_index(tmp_repo):
    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
    )
    # duplicate the same sample twice to exercise the offset logic
    crops, frag_aux, pair_index, pair_aux, labels = collate_batches([sample, sample])
    assert crops.shape[0] == 2 * sample.n_fragments
    assert frag_aux.shape[0] == 2 * sample.n_fragments
    assert pair_index.shape[0] == 2 * sample.n_pairs
    # second sample's pair indices must be offset by the first sample's fragment count
    first_pair = pair_index[0]
    second_pair = pair_index[sample.n_pairs]
    np.testing.assert_array_equal(second_pair, first_pair + sample.n_fragments)
    # every index must be in-bounds for the concatenated crops array
    assert pair_index.max() < crops.shape[0]


def test_collate_batches_empty_list():
    crops, frag_aux, pair_index, pair_aux, labels = collate_batches([])
    assert crops.shape == (0, 384, 7, 7)
    assert frag_aux.shape == (0, FRAGMENT_AUX_DIM)
    assert pair_index.shape == (0, 2)
    assert pair_aux.shape == (0, PAIR_AUX_DIM)
    assert labels.shape == (0,)


def test_load_degradation_config_missing_file_returns_empty(tmp_path):
    assert load_degradation_config(tmp_path / "nope.json") == {}


def test_load_degradation_config_parses_real_shape(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "scenA|111": {"type": "fog", "severity": 1, "seed": 1},
                "scenA|222": {"type": "compression", "severity": 2, "seed": 2},
            }
        )
    )
    cfg = load_degradation_config(cfg_path)
    assert cfg[("scenA", "111")]["type"] == "fog"
    assert cfg[("scenA", "222")]["severity"] == 2


def test_load_violation_lookup_missing_returns_none(tmp_path):
    assert load_violation_lookup(tmp_path / "nope.csv") is None


def test_load_violation_lookup_reads_real_shape(tmp_path):
    csv_path = tmp_path / "violations.csv"
    pd.DataFrame(
        {
            "scenario_id": ["scenA"],
            "frame_id": ["000100"],
            "occupancy_violation_m2": [1.23],
            "perception_violation_m2": [4.56],
        }
    ).to_csv(csv_path, index=False)
    lookup = load_violation_lookup(csv_path)
    assert lookup[("scenA", "000100")] == (1.23, 4.56)


def test_build_sample_uses_violation_lookup_and_sensing_profile(tmp_repo):
    crop_lookup, _ = load_scenario_crops(tmp_repo["crops_dir"], tmp_repo["scenario_id"], arm="A")
    violation_lookup = {(tmp_repo["scenario_id"], tmp_repo["frame_id"]): (0.9, 1.8)}
    sensing_profile = pd.DataFrame(
        {
            "scenario_id": [tmp_repo["scenario_id"]],
            "frame_id": [tmp_repo["frame_id"]],
            "cav_id": [tmp_repo["cav_id"]],
            "mean_range_m": [30.0],
            "clustered_fraction": [0.77],
        }
    )
    sample = build_sample(
        tmp_repo["scenario_id"],
        tmp_repo["frame_id"],
        tmp_repo["cav_id"],
        crop_lookup,
        tmp_repo["repair_labels_dir"],
        arm="A",
        degradation_config={},
        sensing_profile=sensing_profile,
        violation_lookup=violation_lookup,
    )
    assert np.all(sample.frag_aux[:, 4] == pytest.approx(0.77))  # range_density_bucket
    # Raw lookup values (0.9, 1.8) are log1p-transformed in build_sample before
    # being stored, to bring them onto a comparable scale to the other aux
    # features (see dataset.py build_sample).
    assert np.all(sample.frag_aux[:, 6] == pytest.approx(np.log1p(0.9)))  # free_occ_conflict_area
    assert np.all(sample.frag_aux[:, 7] == pytest.approx(np.log1p(1.8)))  # tracking_affinity
