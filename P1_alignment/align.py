"""
P1 — Alignment.

Transform all CAVs into the ego frame using ground-truth pose.

Critical constraint (scope_and_timeline_oct19.md, execution_plan_daily.md Sat 5):
pose noise is a SEPARATE, SWITCHABLE channel — never mixed into the weather /
degradation path (P2). If you ablate one, the other must stay off.

Done when: aligned multi-CAV point clouds render correctly overlaid for one
scenario. Check this by eye — misalignment is obvious visually and invisible
in aggregate metrics.

Implementation note: this reuses OpenCOOD's own pose convention and transform
utilities rather than re-deriving the rotation math, exactly per this file's
original instruction:
  - pose format is [x, y, z, roll, yaw, pitch] — confirmed against
    opencood/data_utils/datasets/basedataset.py:385 (add_loc_noise's own
    docstring) and transformation_utils.x_to_world's unpacking order. This
    is read straight out of each frame's yaml 'lidar_pose' field.
  - opencood.utils.transformation_utils.x1_to_x2(cav_pose, ego_pose) returns
    the 4x4 matrix mapping points FROM the cav's frame TO the ego frame —
    confirmed by reading how early_fusion_dataset.py itself calls it on a
    CAV's own lidar points (early_fusion_dataset.py:168-185).
  - opencood.utils.box_utils.project_points_by_matrix_torch applies that
    matrix to an (N, 3) point array (numpy in, numpy out) — the same
    function OpenCOOD's own fusion datasets use for this exact step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_OPENCOOD_ROOT = Path(__file__).resolve().parent.parent / "OpenCOOD"
if str(_OPENCOOD_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENCOOD_ROOT))

from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402
from opencood.utils.box_utils import project_points_by_matrix_torch  # noqa: E402


def align_to_ego(cav_points: np.ndarray, cav_pose: np.ndarray, ego_pose: np.ndarray,
                  pose_noise_std: float = 0.0) -> np.ndarray:
    """
    Transform `cav_points` (N, 3+) from the CAV's frame into the ego frame.

    cav_pose, ego_pose : array-like, shape (6,)
        [x, y, z, roll, yaw, pitch] — OpenCOOD's own pose convention, read
        straight from each frame's yaml 'lidar_pose' field. No re-parsing
        needed beyond what the Sat 5 Sep detector run already established.

    pose_noise_std : float
        If > 0, perturb `cav_pose`'s translation BEFORE computing the
        transform (not after — noise must propagate through the rotation).
        Defaults to 0.0 (off): the switchable channel required by the plan.
        Never couple this implicitly to a P2 degradation config.

    Returns
    -------
    aligned_points : np.ndarray
        Same shape as cav_points. xyz columns (first 3) are replaced with
        ego-frame coordinates; any extra columns (e.g. intensity) pass
        through unchanged.
    """
    cav_pose = np.asarray(cav_pose, dtype=float)
    ego_pose = np.asarray(ego_pose, dtype=float)

    if pose_noise_std > 0.0:
        cav_pose = add_pose_noise(cav_pose, pose_noise_std)

    ego_T_cav = x1_to_x2(list(cav_pose), list(ego_pose))

    aligned = np.array(cav_points, dtype=float, copy=True)
    aligned[:, :3] = project_points_by_matrix_torch(cav_points[:, :3], ego_T_cav)
    return aligned


def add_pose_noise(pose: np.ndarray, std: float) -> np.ndarray:
    """Additive Gaussian noise on translation; keep rotation noise separate
    and smaller in scale if you add it — don't conflate the two magnitudes."""
    if std <= 0.0:
        return pose
    noisy = pose.copy()
    noisy[:3] += np.random.normal(0.0, std, size=3)
    return noisy
