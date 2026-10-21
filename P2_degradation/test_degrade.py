"""
P2 tests -- fog + compression injection. Implemented 17 Sep 2026 alongside
degrade.py (see that file's docstring for parameter provenance -- Hahner et
al. arXiv:2108.05249's Koschmieder-law model, substituted for Adver-City's
own unreachable numeric tables, documented as a TODO to verify).
"""

import numpy as np
import pytest

from .degrade import inject_fog, inject_compression, FOG_VISIBILITY_M, COMPRESSION_BITS


def _make_ring_cloud(rng, n_per_ring=200):
    """Points at several known ranges from the sensor, for testing range-
    dependent fog behaviour with a known ground truth per range band."""
    ranges = [5.0, 30.0, 60.0, 100.0]
    pts = []
    for r in ranges:
        theta = rng.uniform(-np.pi, np.pi, n_per_ring)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(n_per_ring)
        intensity = np.full(n_per_ring, 0.8)
        pts.append(np.column_stack([x, y, z, intensity]))
    return np.vstack(pts), ranges


def test_inject_fog_invalid_severity_raises():
    points = np.zeros((10, 4))
    with pytest.raises(ValueError):
        inject_fog(points, severity=3, seed=0)


def test_inject_fog_deterministic_with_seed():
    rng = np.random.default_rng(0)
    points, _ = _make_ring_cloud(rng)
    out_a = inject_fog(points, severity=2, seed=42)
    out_b = inject_fog(points, severity=2, seed=42)
    assert out_a.shape == out_b.shape
    assert np.allclose(out_a, out_b)


def test_inject_fog_dense_drops_more_far_range_points_than_moderate():
    """
    Known ground truth, worked out from the formula directly: at r=40m,
    moderate fog (V=100m, alpha=-ln(0.05)/100~=0.02996) attenuates to
    exp(-2*0.02996*40)~=0.091, comfortably above the 0.03 noise floor -- most
    points should survive as hard targets. Dense fog (V=30m, alpha~=0.0998)
    attenuates to exp(-2*0.0998*40)~=0.00034, far below the floor -- points
    should be lost entirely as hard targets (relocated to backscatter or
    dropped instead).
    """
    rng = np.random.default_rng(1)
    theta = rng.uniform(-np.pi, np.pi, 500)
    far_points = np.column_stack(
        [40.0 * np.cos(theta), 40.0 * np.sin(theta), np.zeros(500), np.full(500, 0.8)]
    )

    out_moderate = inject_fog(far_points, severity=1, seed=7)
    out_dense = inject_fog(far_points, severity=2, seed=7)

    # Hard targets: identify by range, excluding any near-sensor backscatter
    # points (range < 15m per FOG_BACKSCATTER_MAX_RANGE_M).
    n_hard_moderate = np.sum(np.linalg.norm(out_moderate[:, :2], axis=1) > 20)
    n_hard_dense = np.sum(np.linalg.norm(out_dense[:, :2], axis=1) > 20)

    assert n_hard_dense < n_hard_moderate
    assert n_hard_dense == 0
    assert n_hard_moderate > 400  # >80% survive under moderate fog at this range


def test_inject_fog_produces_near_sensor_backscatter_when_points_are_lost():
    """Known ground truth: under dense fog, far points that don't survive as
    hard targets partially reappear as near-sensor (<15m) backscatter."""
    rng = np.random.default_rng(2)
    theta = rng.uniform(-np.pi, np.pi, 500)
    far_points = np.column_stack(
        [100.0 * np.cos(theta), 100.0 * np.sin(theta), np.zeros(500), np.full(500, 0.8)]
    )
    out = inject_fog(far_points, severity=2, seed=3)
    ranges_out = np.linalg.norm(out[:, :2], axis=1)
    assert len(out) > 0
    assert np.all(ranges_out < 15.0)  # every surviving point is backscatter here (0 hard targets)
    # backscatter fraction should be close to FOG_BACKSCATTER_REAPPEAR_FRAC of the originals
    expected = int(round(500 * 0.35))
    assert abs(len(out) - expected) <= 5


def test_inject_fog_near_points_mostly_survive_as_hard_targets():
    rng = np.random.default_rng(4)
    theta = rng.uniform(-np.pi, np.pi, 500)
    near_points = np.column_stack(
        [5.0 * np.cos(theta), 5.0 * np.sin(theta), np.zeros(500), np.full(500, 0.8)]
    )
    out = inject_fog(near_points, severity=2, seed=5)  # even dense fog
    ranges_out = np.linalg.norm(out[:, :2], axis=1)
    # Most survivors should still be near the original 5m range (hard targets,
    # not relocated backscatter), since attenuation at 5m is mild.
    n_near_hard = np.sum(ranges_out > 3.0)
    assert n_near_hard > 450  # >90% survive essentially intact


def test_inject_compression_invalid_severity_raises():
    tensor = np.random.default_rng(0).uniform(0, 1, (4, 4))
    with pytest.raises(ValueError):
        inject_compression(tensor, severity=5, seed=0)


def test_inject_compression_reduces_unique_value_count():
    rng = np.random.default_rng(6)
    tensor = rng.uniform(-1, 1, (64, 64))
    original_unique = len(np.unique(tensor))
    out_8bit = inject_compression(tensor, severity=1, seed=0)
    out_4bit = inject_compression(tensor, severity=2, seed=0)

    assert len(np.unique(out_8bit)) <= 256
    assert len(np.unique(out_4bit)) <= 16
    assert len(np.unique(out_4bit)) <= len(np.unique(out_8bit))
    assert len(np.unique(out_8bit)) < original_unique


def test_inject_compression_dense_severity_loses_more_information():
    """Known ground truth: coarser quantisation (severity 2, 4-bit) should
    have strictly higher mean squared reconstruction error than the finer
    8-bit quantisation, for the same input tensor."""
    rng = np.random.default_rng(7)
    tensor = rng.uniform(-1, 1, (32, 32))
    out_8bit = inject_compression(tensor, severity=1, seed=0)
    out_4bit = inject_compression(tensor, severity=2, seed=0)

    mse_8bit = np.mean((tensor - out_8bit) ** 2)
    mse_4bit = np.mean((tensor - out_4bit) ** 2)
    assert mse_4bit > mse_8bit


def test_inject_compression_constant_tensor_is_a_noop():
    tensor = np.full((8, 8), 3.5)
    out = inject_compression(tensor, severity=1, seed=0)
    assert np.allclose(out, tensor)
