"""
P2 -- Degradation injection (fog + compression, 2 severities each).

Implemented 17 Sep 2026 (Week 2, compressed into one sitting -- originally
Wed 9 / Thu 10 Sep). Per scope_and_timeline_oct19.md: two degradation types
(fog, compression), applied to one benign CAV's data at a time, BEFORE the
perception model. Severities cut from 3 -> 2. `(type, severity)` metadata is
load-bearing for Part 3's fleet partition (P6) and topology construction --
not just Part 1's variable anymore.

--- Parameter provenance (read this before citing in the methods section) ---

Adver-City's own paper (arXiv:2410.06380) was checked this session via web
search/fetch. Its abstract and body text confirm "fog" and "foggy heavy rain"
exist as two of its six weather conditions, but do NOT expose numeric LiDAR
degradation parameters (attenuation coefficients, dropout rates, visibility
values) in any form the fetch could extract -- those would live in a
supplementary/appendix section or the dataset's generation code, neither of
which was accessible here. So "calibrate against Adver-City's published
statistics" as originally scoped could not be done directly this session.

Instead, `inject_fog` below implements the standard physical LiDAR fog model
from Hahner et al., "Fog Simulation on Real LiDAR Point Clouds for 3D Object
Detection in Adverse Weather" (ICCV 2021, arXiv:2108.05249) -- the model this
literature typically uses (and plausibly what a CARLA-derived dataset like
Adver-City approximates for LiDAR, since CARLA has no built-in physical LiDAR
fog attenuation of its own). This is a REAL, citable, physically-grounded
source -- just not Adver-City's own numbers. **Open TODO before the methods
section is written:** check Adver-City's GitHub (github.com/QUARRG/Adver-City)
or supplementary material directly for their actual LiDAR fog parameters, and
either confirm this model matches or replace these constants with theirs.
Document this substitution explicitly either way -- do not silently present
Hahner et al.'s numbers as Adver-City's own.

The physical model (Hahner et al., Koschmieder's law):
  - Meteorological visibility V (metres) sets the extinction coefficient
    alpha = -ln(0.05) / V  (~3/V; 0.05 = the standard 5% contrast threshold
    used to define "visibility" in atmospheric optics).
  - A hard-target return's power attenuates as exp(-2*alpha*R) with range R.
  - If the attenuated return falls below a noise floor, the beam is more
    likely to reflect off fog droplets themselves (a "soft target") instead
    of reaching the real object -- producing spurious near-sensor returns
    (backscatter) rather than the true point.

Two severities, chosen as the plan's "separated" pair -- one the defence
mostly survives, one that clearly stresses it (execution_plan_daily.md,
Thu 10 Sep note):
  1 = moderate fog,  V = 100 m  (still well within OPV2V's 120 m clip range)
  2 = dense fog,      V = 30 m   (WHO/aviation-standard "dense fog" bound)
"""

from __future__ import annotations

import numpy as np

# --- Fog: Hahner et al. (arXiv:2108.05249) Koschmieder-law parameters ---
FOG_VISIBILITY_M = {1: 100.0, 2: 30.0}          # moderate / dense
FOG_CONTRAST_THRESHOLD = 0.05                    # standard 5% definition of "visibility"
FOG_NOISE_FLOOR = 0.03                           # attenuated-power fraction below which a
                                                  # return is treated as lost to backscatter
                                                  # rather than a valid hard-target hit
FOG_BACKSCATTER_REAPPEAR_FRAC = 0.35             # fraction of "lost" points that reappear as
                                                  # spurious near-sensor fog returns rather than
                                                  # vanishing outright (soft-target reflections)
FOG_BACKSCATTER_MAX_RANGE_M = 15.0               # fog backscatter is a near-sensor phenomenon

# --- Compression: voxel/BEV-tensor quantisation ---
# Bit-depth chosen as a deliberately simple, symmetric pair either side of a
# typical fp16-safe range -- 8-bit is common lossy-int8 quantisation (mirrors
# Part 3's own planned int8 comms-budget quantisation, axis 5), 4-bit is a
# stress case with visible precision loss.
COMPRESSION_BITS = {1: 8, 2: 4}


def inject_fog(points: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """
    Range-dependent dropout + attenuation + near-sensor backscatter.

    points: (N, 4+) array, columns x, y, z, intensity[, ...]. Assumed to
        already be in the sensor's own local frame (this runs on one CAV's
        raw point cloud, before P1 alignment / P3 detection).
    severity: 1 (moderate, V=100m) or 2 (dense, V=30m).
    seed: RNG seed, keyed by caller on (scenario, type, severity) per the
        plan's reproducibility requirement -- same seed always gives the
        same degraded cloud for a given input.

    Returns a new (M, 4+) array: intensity-attenuated hard-target points
    that survived, plus synthetic near-sensor backscatter points, in place
    of the points that fog absorbed.
    """
    if severity not in FOG_VISIBILITY_M:
        raise ValueError(f"severity must be one of {sorted(FOG_VISIBILITY_M)}, got {severity}")
    rng = np.random.default_rng(seed)

    if len(points) == 0:
        return points.copy()

    visibility = FOG_VISIBILITY_M[severity]
    alpha = -np.log(FOG_CONTRAST_THRESHOLD) / visibility

    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)
    attenuation = np.exp(-2.0 * alpha * ranges)  # fraction of original return power surviving

    survives_as_hard_target = attenuation >= FOG_NOISE_FLOOR
    hard_targets = points[survives_as_hard_target].copy()
    if hard_targets.shape[1] >= 4:
        # Intensity column attenuates with the same fraction -- a real return
        # that made it back weaker than it left.
        hard_targets[:, 3] = hard_targets[:, 3] * attenuation[survives_as_hard_target]

    lost = points[~survives_as_hard_target]
    n_lost = len(lost)
    n_backscatter = int(round(n_lost * FOG_BACKSCATTER_REAPPEAR_FRAC))

    if n_backscatter == 0:
        return hard_targets

    # Backscatter points: reuse the lost points' own azimuth/elevation
    # directions (so the spurious returns still look like they came from
    # roughly where the beam was pointed) but pull them in to a short,
    # near-sensor range -- this is what "confirmed empty" free-space in
    # P4's get_free_space() would otherwise wrongly claim, since fog
    # particles give the sensor *something* to bounce off before it would
    # have reached the real (now-lost) target.
    chosen = rng.choice(n_lost, size=n_backscatter, replace=False)
    lost_dirs = lost[chosen, :3]
    lost_norms = np.linalg.norm(lost_dirs, axis=1, keepdims=True)
    lost_norms[lost_norms == 0] = 1.0
    unit_dirs = lost_dirs / lost_norms

    backscatter_ranges = rng.uniform(0.3, FOG_BACKSCATTER_MAX_RANGE_M, size=n_backscatter)
    backscatter_xyz = unit_dirs * backscatter_ranges[:, None]

    backscatter_pts = np.zeros((n_backscatter, points.shape[1]), dtype=points.dtype)
    backscatter_pts[:, :3] = backscatter_xyz
    if points.shape[1] >= 4:
        # Backscatter returns are characteristically low-intensity, diffuse
        # reflections off droplets, not a solid surface.
        backscatter_pts[:, 3] = rng.uniform(0.01, 0.15, size=n_backscatter)

    return np.vstack([hard_targets, backscatter_pts])


def inject_compression(bev_tensor: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """
    Voxel / BEV-tensor quantisation -- simulates lossy compression of the
    intermediate representation (uniform min-max quantisation to N bits per
    channel, the standard lossy-quantisation model and the same mechanism
    Part 3's own int8 communication-budget axis will use later, axis 5).

    bev_tensor: any float array (voxel grid or BEV feature map).
    severity: 1 (8-bit) or 2 (4-bit, more aggressive).
    seed: kept as a parameter for schema consistency with inject_fog and in
        case dithering noise is added later; quantisation itself is
        deterministic given the tensor, so it's unused for now.
    """
    if severity not in COMPRESSION_BITS:
        raise ValueError(f"severity must be one of {sorted(COMPRESSION_BITS)}, got {severity}")
    del seed  # deterministic given the tensor; kept for interface symmetry with inject_fog

    tensor = np.asarray(bev_tensor, dtype=np.float64)
    bits = COMPRESSION_BITS[severity]
    n_levels = 2 ** bits

    vmin = tensor.min()
    vmax = tensor.max()
    if vmax <= vmin:
        # Degenerate constant tensor -- nothing to quantise, return as-is
        # (upcast back to the caller's dtype).
        return tensor.astype(bev_tensor.dtype)

    step = (vmax - vmin) / (n_levels - 1)
    quantised = np.round((tensor - vmin) / step) * step + vmin
    return quantised.astype(bev_tensor.dtype)
