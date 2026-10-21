"""
P3 tests. `perception_cache.py` now has a real implementation (17 Sep 2026),
but the parts that call the actual detector (run_frozen_inference_and_cache,
run_frozen_inference_and_cache_with_crops) depend on torch + CUDA + the
actual OpenCOOD checkpoint + the real OPV2V data on disk -- none of which
exist in the cloud sandbox this was written in, and those still cannot be
executed or unit-tested here. The degradation-injection logic they call
(inject_fog/inject_compression) IS independently unit-tested, in
P2_degradation/test_degrade.py.

Thu 1 Oct 2026 addition: the BEV-crop-caching pixel-mapping/cropping helpers
(_pixel_from_xy, crop_bev_feature, _load_fragment_centroids) are PURE
numpy/pandas -- no torch import needed -- so unlike the rest of this file,
these ARE unit-tested for real in the cloud sandbox below, against this
project's actual real model config values (cav_lidar_range, voxel_size, H/W)
read straight from pointpillar_late_fusion/config.yaml, not placeholders.

Run the rest of this file for real in the `opencood` conda env once
perception_cache.py has been run at least once against real data, to catch
integration bugs the sandbox couldn't (idx/frame_id resolution, dataset root
overriding, the model.backbone hook actually firing once per CAV, etc.).
"""

import numpy as np
import pandas as pd
import pytest

from .perception_cache import (
    BEV_CROP_SIZE_PX,
    _pixel_from_xy,
    crop_bev_feature,
    _load_fragment_centroids,
)

# This project's real pointpillar_late_fusion/config.yaml values (Thu 1 Oct
# 2026) -- not placeholders. lidar_range = [xmin, ymin, zmin, xmax, ymax, zmax].
REAL_LIDAR_RANGE = [-70.4, -40, -3, 70.4, 40, 1]
REAL_VOXEL_SIZE = [0.4, 0.4, 4]
REAL_GRID_SHAPE = (200, 352)  # (H, W), matches postprocess.anchor_args in config.yaml


def test_pixel_from_xy_origin_maps_to_grid_centre():
    # (0, 0) in the CAV's own local frame is the sensor's own position, which
    # sits at the centre of a symmetric lidar_range box -- should map to the
    # centre of whatever feature map resolution is passed, regardless of the
    # backbone's actual downsample factor.
    feat_shape = (100, 176)  # this project's real spatial_features_2d shape (2x downsample)
    row, col = _pixel_from_xy(0.0, 0.0, REAL_LIDAR_RANGE, REAL_VOXEL_SIZE, REAL_GRID_SHAPE, feat_shape)
    assert row == pytest.approx(feat_shape[0] / 2, abs=0.5)
    assert col == pytest.approx(feat_shape[1] / 2, abs=0.5)


def test_pixel_from_xy_scales_with_downsample_factor():
    # A point 10m in +x should land at a column proportionally further right
    # at higher resolution (smaller downsample) than at lower resolution --
    # this is what makes the mapping self-correcting against a backbone
    # config change instead of hardcoding "2x".
    row_2x, col_2x = _pixel_from_xy(10.0, 0.0, REAL_LIDAR_RANGE, REAL_VOXEL_SIZE, REAL_GRID_SHAPE, (100, 176))
    row_1x, col_1x = _pixel_from_xy(10.0, 0.0, REAL_LIDAR_RANGE, REAL_VOXEL_SIZE, REAL_GRID_SHAPE, (200, 352))
    # Same physical point should be twice as far (in pixels) from the origin
    # column at the full-resolution (1x) grid vs the 2x-downsampled one.
    origin_col_2x = REAL_GRID_SHAPE[1] / 2 / 2
    origin_col_1x = REAL_GRID_SHAPE[1] / 2
    assert (col_1x - origin_col_1x) == pytest.approx(2 * (col_2x - origin_col_2x), rel=0.05)


def test_crop_bev_feature_centred_window_matches_source():
    feat = np.arange(3 * 20 * 20, dtype=float).reshape(3, 20, 20)
    crop, valid = crop_bev_feature(feat, row=10, col=10, crop_size_px=5)
    assert crop.shape == (3, 5, 5)
    assert valid is True
    # Centre pixel of the crop must equal the source's (10,10) pixel exactly.
    assert np.array_equal(crop[:, 2, 2], feat[:, 10, 10])


def test_crop_bev_feature_near_edge_zero_pads_without_crashing():
    feat = np.ones((2, 10, 10))
    crop, valid = crop_bev_feature(feat, row=0, col=0, crop_size_px=5)
    assert crop.shape == (2, 5, 5)
    assert valid is True  # centre pixel (0,0) is still inside the map
    # Top-left quadrant of the crop (which falls outside the source) must be
    # zero; only the in-bounds portion should carry real content.
    assert np.all(crop[:, 0, 0] == 0)
    assert np.all(crop[:, 2, 2] == 1)  # centre pixel is in-bounds and real


def test_crop_bev_feature_centre_fully_outside_map_is_invalid():
    feat = np.ones((2, 10, 10))
    crop, valid = crop_bev_feature(feat, row=-50, col=-50, crop_size_px=5)
    assert valid is False
    assert np.all(crop == 0)


def test_load_fragment_centroids_reads_real_parquet_schema(tmp_path):
    scenario_dir = tmp_path / "sA"
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {"fragment_id": 5, "centroid_x": 1.5, "centroid_y": -2.0, "n_points": 12, "vehicle_id": 1845, "is_recovered_noise": False},
            {"fragment_id": 9, "centroid_x": 3.0, "centroid_y": 4.0, "n_points": 6, "vehicle_id": 1924, "is_recovered_noise": True},
        ]
    ).to_parquet(scenario_dir / "0001_1845_fragments.parquet")

    centroids = _load_fragment_centroids(str(tmp_path), "sA", "0001", "1845")
    assert len(centroids) == 2
    assert (5, 1.5, -2.0) in centroids


def test_load_fragment_centroids_missing_file_returns_empty_list(tmp_path):
    assert _load_fragment_centroids(str(tmp_path), "sA", "9999", "1845") == []
