import numpy as np
import pytest

from .align import add_pose_noise, align_to_ego


def test_pose_noise_off_by_default():
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    assert np.array_equal(add_pose_noise(pose, std=0.0), pose)


def test_pose_noise_perturbs_translation_only_when_enabled():
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    noisy = add_pose_noise(pose, std=0.5)
    assert not np.array_equal(noisy, pose)
    # rotation channel (indices 3:) untouched by this simple version
    assert np.array_equal(noisy[3:], pose[3:])


def test_align_to_ego_identity_pose_is_unchanged():
    """CAV and ego at the exact same pose -> points pass through unchanged."""
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    points = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]])
    aligned = align_to_ego(points, cav_pose=pose, ego_pose=pose)
    assert np.allclose(aligned[:, :3], points[:, :3], atol=1e-5)
    assert np.allclose(aligned[:, 3], points[:, 3])  # intensity passed through


def test_align_to_ego_known_translation():
    """
    CAV is 10m ahead of ego in world x, same orientation (yaw=0). A point at
    the CAV's own local origin (its own LiDAR position) must land at
    (10, 0, 0) in the ego frame -- known ground truth, hand-derived from
    OpenCOOD's x_to_world/x1_to_x2 convention and verified independently
    before this test was written.
    """
    cav_pose = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ego_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    point_at_cav_origin = np.array([[0.0, 0.0, 0.0, 1.0]])
    aligned = align_to_ego(point_at_cav_origin, cav_pose=cav_pose, ego_pose=ego_pose)
    assert np.allclose(aligned[0, :3], [10.0, 0.0, 0.0], atol=1e-5)
    assert np.isclose(aligned[0, 3], 1.0)  # intensity untouched


def test_align_to_ego_known_yaw_rotation():
    """
    CAV at the same location as ego but yawed +90 degrees. A point 5m ahead
    of the CAV in the CAV's own local +x axis must land at (0, 5, 0) in the
    ego frame under OpenCOOD's yaw convention -- hand-derived and verified
    independently (see decisions.md) before this test was written, so a
    future accidental sign flip in the rotation math gets caught here rather
    than only showing up as a subtle misalignment in the eyeball check.
    """
    cav_pose = np.array([0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    ego_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    point_ahead_of_cav = np.array([[5.0, 0.0, 0.0, 1.0]])
    aligned = align_to_ego(point_ahead_of_cav, cav_pose=cav_pose, ego_pose=ego_pose)
    assert np.allclose(aligned[0, :3], [0.0, 5.0, 0.0], atol=1e-3)


def test_align_to_ego_with_pose_noise_perturbs_result():
    """pose_noise_std > 0 should move the aligned point away from the
    noiseless result -- confirms the noise channel is actually wired into
    align_to_ego, not just add_pose_noise in isolation."""
    cav_pose = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ego_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    point = np.array([[0.0, 0.0, 0.0, 1.0]])

    clean = align_to_ego(point, cav_pose=cav_pose, ego_pose=ego_pose, pose_noise_std=0.0)
    noisy = align_to_ego(point, cav_pose=cav_pose, ego_pose=ego_pose, pose_noise_std=2.0)
    assert not np.allclose(clean[0, :3], noisy[0, :3])


def test_align_overlay_visual_check():
    pytest.skip(
        "Manual done-when: plot aligned multi-CAV clouds for one real scenario "
        "and eyeball the overlay now that align_to_ego() is implemented. "
        "Misalignment is obvious visually and invisible in metrics -- don't "
        "skip the eyeball check even though this test exists."
    )
