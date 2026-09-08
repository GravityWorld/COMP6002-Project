import numpy as np
import pytest

from .align import add_pose_noise


def test_pose_noise_off_by_default():
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    assert np.array_equal(add_pose_noise(pose, std=0.0), pose)


def test_pose_noise_perturbs_translation_only_when_enabled():
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    noisy = add_pose_noise(pose, std=0.5)
    assert not np.array_equal(noisy, pose)
    # rotation channel (indices 3:) untouched by this simple version
    assert np.array_equal(noisy[3:], pose[3:])


def test_align_overlay_visual_check():
    pytest.skip(
        "Manual done-when: plot aligned multi-CAV clouds for one real scenario "
        "and eyeball the overlay once align_to_ego() is implemented. "
        "Misalignment is obvious visually and invisible in metrics — don't "
        "skip the eyeball check even once this test exists."
    )
