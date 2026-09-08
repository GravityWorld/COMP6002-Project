"""
P1 — Alignment.

Transform all CAVs into the ego frame using ground-truth pose.

Critical constraint (scope_and_timeline_oct19.md, execution_plan_daily.md Sat 5):
pose noise is a SEPARATE, SWITCHABLE channel — never mixed into the weather /
degradation path (P2). If you ablate one, the other must stay off.

Done when: aligned multi-CAV point clouds render correctly overlaid for one
scenario. Check this by eye — misalignment is obvious visually and invisible
in aggregate metrics.
"""

import numpy as np


def align_to_ego(cav_points: np.ndarray, cav_pose: np.ndarray, ego_pose: np.ndarray,
                  pose_noise_std: float = 0.0) -> np.ndarray:
    """
    Transform `cav_points` (N, 3+) from the CAV's frame into the ego frame.

    cav_pose, ego_pose: (x, y, z, roll, pitch, yaw) or a 4x4 transform —
        pick one convention and match whatever OpenCOOD's loader already
        uses (you have this working from Sat 5 Sep's detector run — reuse
        the same pose parsing rather than re-deriving it).

    pose_noise_std: if > 0, perturb `cav_pose` before transforming. This is
        the switchable noise channel — it must default to 0.0 (off) and be
        toggled explicitly, never implicitly coupled to a degradation config.
    """
    raise NotImplementedError(
        "Implement using the same pose-parsing convention as the OpenCOOD "
        "loader from the Sat 5 Sep detector run. Compute ego_T_cav = "
        "inv(ego_pose) @ cav_pose (or the (x,y,z,yaw) equivalent), apply to "
        "cav_points, and optionally perturb cav_pose with pose_noise_std "
        "BEFORE computing the transform (not after, or noise won't propagate "
        "correctly through rotation)."
    )


def add_pose_noise(pose: np.ndarray, std: float) -> np.ndarray:
    """Additive Gaussian noise on translation; keep rotation noise separate
    and smaller in scale if you add it — don't conflate the two magnitudes."""
    if std <= 0.0:
        return pose
    noisy = pose.copy()
    noisy[:3] += np.random.normal(0.0, std, size=3)
    return noisy
