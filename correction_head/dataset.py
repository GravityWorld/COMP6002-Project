"""Dataset loader joining P3's cached BEV feature crops with P5's
repair-labels fragment/pairs tables, for training
correction_head.model.CorrectionHead.

WHAT THIS JOINS
----------------
- P3_perception/cache/<scenario>_<arm>_crops.npz (perception_cache.py
  --cache-bev-crops): parallel arrays (frame_id, cav_id, fragment_id, crop,
  valid), one row per candidate fragment that had a BEV feature crop
  cached. `valid=False` means the fragment's centroid pixel fell outside
  the feature map -- those rows are dropped here, same convention as
  crop_bev_feature()'s own docstring.
- P4_defence/cache/repair_labels/<scenario>/<frame>_<cav>_fragments.parquet
  and _pairs.parquet (run_cluster_repair_labelling.py): per (scenario,
  frame, cav), the real candidate-fragment table and the real pairwise
  affinity_label supervision target.

`fragment_id` is only locally unique within one (scenario, frame, cav) --
both the crops npz and the repair-labels parquet files were written using
the exact same local numbering (perception_cache.py's
_load_fragment_centroids() reads centroids straight out of the same
fragments.parquet this loader reads), so joining on
(frame_id, cav_id, fragment_id) is safe and exact, not approximate.

ONE TRAINING UNIT = ONE (scenario, frame, cav)
------------------------------------------------
CorrectionHead.forward() takes one flat batch of fragments + requested
pairs (see model.py) -- there's no notion of "this pair's two fragments
must come from the same (frame, cav)" baked into the model itself, but in
practice every real pair in pairs.parquet already is confined to one
(frame, cav) (run_cluster_repair_labelling.py only pairs fragments
observed by the same CAV in the same frame -- a cross-CAV or cross-frame
pair has no meaning for "is this the same raw cluster fragment or not").
So CorrectionHeadSample keeps that unit intact: one sample = one
(scenario, frame, cav)'s full fragment set + its full candidate-pair set.
`collate_batches()` below concatenates several samples into one larger
flat batch (offsetting pair indices per sample) when a bigger mini-batch
than a single (frame, cav) is wanted -- this is what lets training batch
size be decoupled from how many fragments one frame/cav happens to have.

AUXILIARY FEATURES -- SOURCES AND TWO KNOWN GAPS, STATED EXPLICITLY
-----------------------------------------------------------------------
Seven of the nine AUX_FEATURE_NAMES (model.py) are wired to real data here:
log1p_n_points, is_recovered_noise, range_from_ego_norm, bearing_norm (all
from fragments.parquet), range_density_bucket (from P6's
compute_sensing_profile_from_repair_labels -- clustered_fraction, the
density half of that function's profile, used as the scalar directly
rather than its quantile-bucket label, since a continuous feature carries
more signal than decoding the bucket string back to a number), and
degradation_severity (0.0 for arm A; looked up from
degradation_config_arm_b.json for arm B, keyed the same way P2/P3 key it),
and frag_extent_norm (added Wed 7 Oct 2026 -- log1p of each fragment's RMS
radius from its own centroid, computed straight from the real per-point
xyz in <frame>_<cav>_points.npz grouped by fragment_id; see
_fragment_extent_norm() below). This was added specifically because
log1p_n_points alone conflates "few points because it's a small real
object" with "few points because it's a sliver accidentally split off a
bigger one" -- extent gives the model an independent spatial-spread signal
a point count alone can't.

The remaining two -- free_occ_conflict_area and tracking_affinity -- are
GAP #1, ZERO-FILLED FOR NOW. model.py's own comment says both come from
"P4's free/occupied violation scalars, frame-level, broadcast": in
practice that's `run_defence_check.evaluate_frame()`'s two returned
magnitudes, occupancy_violation (`max_violation_m2`) and
perception_violation (`perception_violation_m2`) -- "tracking_affinity"
is a misleading name inherited from the architecture docstring written
the same day; there is no real object tracker anywhere in this project
(P0-P6 are all single-frame), so that name should eventually be corrected
to something like `perception_occupancy_violation_norm`. Computing either
scalar for real requires running evaluate_frame() per (scenario, frame),
which needs the real opencood conda env (pcd_to_np/mask_ego_points import
torch indirectly) -- not available in the cloud sandbox or the
device-bridge's own Linux VM this file was written and tested against.
`load_violation_lookup()` below reads a precomputed cache if one exists
(see compute_violation_scalars.py, not yet written) and otherwise returns
None, in which case both features are 0.0 for every fragment. This is a
real, open gap, not an oversight -- flagged in decisions.md and
handover.md, next up after this loader lands.

GAP #2, A REAL DATA-COVERAGE FINDING, NOT A LOADER BUG: as of Thu 1 Oct
2026, P3_perception/cache/*_crops.npz only contains crops for ONE
(frame, cav) per scenario (003464/1872 and 003507/... respectively) --
not all 6 repair-labelled frames x 3 CAVs each, despite the
--cache-bev-crops run's own printed "146"/"126 BEV crops" looking
plausible at a glance. Root cause not yet established (see
decisions.md) -- candidates include frame-list mismatch (P3's own
--max-frames-per-scenario selection vs. run_cluster_repair_labelling.py's
6-evenly-spaced-frames selection only overlapping at one frame) and/or a
real bug in run_frozen_inference_and_cache_with_crops()'s per-CAV loop
only capturing one CAV's hook output. This loader handles it gracefully
(a (frame, cav) with no cached crops simply contributes zero samples --
see build_sample() returning None), but it means only ~2 real training
units exist right now, nowhere near enough to train on. Re-running crop
caching with full frame/CAV coverage is the next real blocker before
training the head for real -- this loader is ready for that data the
moment it exists; it does not need to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # keeps this file's pure-numpy/pandas logic importable
    # and unit-testable without torch (cloud sandbox / device-bridge VM,
    # neither of which has torch -- same convention as model.py's own
    # module docstring).
    torch = None
    Dataset = object

# Deliberately NOT importing these from correction_head.model: model.py
# imports torch unconditionally at module level (it's an nn.Module file,
# that's correct for it), so importing it here would make this whole
# module -- including its pure-numpy/pandas pieces -- require torch too.
# Same reasoning P3_perception/perception_cache.py's own module docstring
# already gives for duplicating a constant instead of importing from a
# torch-requiring module. Values below MUST stay in sync with model.py;
# `assert_matches_model()` checks that for real whenever torch IS
# available (call it once at the start of a real training run).
AUX_FEATURE_NAMES = (
    "log1p_n_points",
    "is_recovered_noise",
    "range_from_ego_norm",
    "bearing_norm",
    "range_density_bucket",
    "degradation_severity",
    "free_occ_conflict_area",
    "tracking_affinity",
    "frag_extent_norm",
)
FRAGMENT_AUX_DIM = len(AUX_FEATURE_NAMES)
PAIR_AUX_DIM = 3
LIDAR_RANGE = (-70.4, -40.0, -3.0, 70.4, 40.0, 1.0)  # config.yaml model.args.lidar_range


def assert_matches_model() -> None:
    """Call this once at the start of a real training run (after torch is
    importable) to confirm this module's duplicated constants haven't
    drifted from correction_head.model's real ones. Not called at import
    time here on purpose -- that would re-impose the torch dependency this
    duplication exists to avoid."""
    from correction_head import model as _model

    assert AUX_FEATURE_NAMES == _model.AUX_FEATURE_NAMES, (AUX_FEATURE_NAMES, _model.AUX_FEATURE_NAMES)
    assert FRAGMENT_AUX_DIM == _model.FRAGMENT_AUX_DIM
    assert PAIR_AUX_DIM == _model.PAIR_AUX_DIM
    assert LIDAR_RANGE == _model.LIDAR_RANGE


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CROPS_DIR = REPO_ROOT / "P3_perception" / "cache"
DEFAULT_REPAIR_LABELS_DIR = REPO_ROOT / "P4_defence" / "cache" / "repair_labels"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"
DEFAULT_DEGRADATION_CONFIG = REPO_ROOT / "P3_perception" / "degradation_config_arm_b.json"

# Must match P4_defence/run_cluster_repair_labelling.py's own constant --
# pairs.parquet's centroid_distance_m was generated against this radius,
# so this is what normalises it back into roughly [0, 1] for the model.
CANDIDATE_PAIR_RADIUS_M = 4.0

# LIDAR_RANGE = (x_min, y_min, z_min, x_max, y_max, z_max) from model.py /
# config.yaml -- used only to normalise a fragment's range from the
# sensor origin into roughly [0, 1].
_LIDAR_MAX_RANGE_M = max(
    abs(LIDAR_RANGE[0]), abs(LIDAR_RANGE[1]), abs(LIDAR_RANGE[3]), abs(LIDAR_RANGE[4])
)


@dataclass
class CorrectionHeadSample:
    """One (scenario, frame, cav)'s fragments + candidate pairs, as plain
    numpy arrays -- shapes match CorrectionHead.forward()'s per-call
    arguments exactly (see model.py)."""

    scenario_id: str
    frame_id: str
    cav_id: str
    crops: np.ndarray  # (N, 384, 7, 7) float32
    frag_aux: np.ndarray  # (N, FRAGMENT_AUX_DIM) float32
    pair_index: np.ndarray  # (P, 2) int64, local indices 0..N-1
    pair_aux: np.ndarray  # (P, PAIR_AUX_DIM) float32
    labels: np.ndarray  # (P,) float32, 0.0/1.0

    @property
    def n_fragments(self) -> int:
        return self.crops.shape[0]

    @property
    def n_pairs(self) -> int:
        return self.pair_index.shape[0]


def load_degradation_config(path=DEFAULT_DEGRADATION_CONFIG) -> dict:
    """Parses P3's `{"<scenario>|<cav>": {"type", "severity", "seed"}}`
    JSON (same file P3_perception/perception_cache.py and
    run_defence_check.py's apply_degradation_if_assigned() already key the
    same way) into a {(scenario_id, cav_id): meta} dict. Returns {} if the
    file doesn't exist -- arm A (clean) never needs it."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key, meta in raw.items():
        scenario_id, cav_id = key.split("|", 1)
        out[(scenario_id, cav_id)] = meta
    return out


def _degradation_severity(arm: str, scenario_id: str, cav_id: str, degradation_config: dict) -> float:
    if arm != "B":
        return 0.0
    meta = degradation_config.get((scenario_id, str(cav_id)))
    if meta is None:
        return 0.0
    return float(meta["severity"])


def load_violation_lookup(path=None):
    """Reads a precomputed {(scenario_id, frame_id): (occupancy_violation_m2,
    perception_violation_m2)} table, if `compute_violation_scalars.py` (not
    yet written -- see module docstring GAP #1) has been run for real and
    cached one. Returns None if no such cache exists, in which case the
    caller zero-fills those two aux features. Deliberately permissive
    about the on-disk format (csv with those exact column names) so a
    future precompute script has one obvious contract to match."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path, dtype={"scenario_id": str, "frame_id": str})
    return {
        (row.scenario_id, row.frame_id): (row.occupancy_violation_m2, row.perception_violation_m2)
        for row in df.itertuples()
    }


def load_scenario_crops(crops_dir, scenario_id: str, arm: str = "A"):
    """Loads P3's `<scenario>_<arm>_crops.npz`. Returns
    ({(frame_id, cav_id, fragment_id): crop array (384,7,7)}, n_dropped)
    -- rows with valid=False are dropped (nothing to feed the encoder;
    see crop_bev_feature()'s own docstring for what `valid` means) and
    counted so a silently-all-invalid file doesn't pass unnoticed."""
    path = Path(crops_dir) / f"{scenario_id}_{arm}_crops.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached BEV crops at {path}. Run "
            f"`perception_cache.py --cache-bev-crops --arm {arm}` for scenario "
            f"{scenario_id} on the real machine first (see decisions.md, "
            f"Thu 1 Oct 2026 -- as of this loader being written, only arm A "
            f"has been cache-crop-run at all)."
        )
    d = np.load(path, allow_pickle=True)
    out = {}
    n_dropped = 0
    for frame_id, cav_id, fragment_id, crop, valid in zip(
        d["frame_id"], d["cav_id"], d["fragment_id"], d["crop"], d["valid"]
    ):
        if not bool(valid):
            n_dropped += 1
            continue
        out[(str(frame_id), str(cav_id), int(fragment_id))] = crop
    return out, n_dropped


def discover_frame_cav_pairs(repair_labels_dir, scenario_id: str) -> list[tuple[str, str]]:
    """Lists every (frame_id, cav_id) run_cluster_repair_labelling.py
    produced output for in this scenario, by scanning
    `<repair_labels_dir>/<scenario_id>/*_fragments.parquet` filenames
    (stem convention "{frame_id}_{cav_id}", same as
    perception_cache._load_fragment_centroids() relies on)."""
    scen_dir = Path(repair_labels_dir) / scenario_id
    if not scen_dir.exists():
        return []
    pairs = []
    for f in sorted(scen_dir.glob("*_fragments.parquet")):
        stem = f.name[: -len("_fragments.parquet")]
        frame_id, cav_id = stem.split("_", 1)
        pairs.append((frame_id, cav_id))
    return pairs


def _load_and_filter_frags_pairs(scenario_id: str, frame_id: str, cav_id: str, crop_lookup: dict, repair_labels_dir):
    """Shared by build_sample() and build_pair_type_weights() -- the exact
    same frags/pairs filtering (crop-coverage drop, orphaned-pair drop),
    so a pair-type weight array built from this function's `pairs` return
    is GUARANTEED to line up row-for-row with the pair_index build_sample()
    derives from the same filtered `pairs` -- no separate traversal to
    keep in sync, no alignment risk between the two. Returns
    (frags, pairs, local_index) or None if there's nothing usable."""
    frag_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_fragments.parquet"
    pair_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_pairs.parquet"
    if not frag_path.exists() or not pair_path.exists():
        return None

    frags = pd.read_parquet(frag_path)
    pairs = pd.read_parquet(pair_path)

    has_crop = frags["fragment_id"].apply(
        lambda fid: (str(frame_id), str(cav_id), int(fid)) in crop_lookup
    )
    frags = frags[has_crop].reset_index(drop=True)
    if frags.empty:
        return None

    # A pair referencing a fragment that was dropped (no valid crop) can't
    # be learned from -- its affinity label is real, but the model has no
    # input for that fragment, so the pair is dropped with it.
    kept_ids = set(frags["fragment_id"].tolist())
    pairs = pairs[pairs["fragment_i"].isin(kept_ids) & pairs["fragment_j"].isin(kept_ids)].reset_index(
        drop=True
    )

    local_index = {fid: i for i, fid in enumerate(frags["fragment_id"].tolist())}
    return frags, pairs, local_index


def _fragment_extent_norm(scenario_id: str, frame_id: str, cav_id: str, repair_labels_dir, fragment_ids) -> np.ndarray:
    """log1p(RMS radius from centroid, in meters) for each id in
    `fragment_ids`, in that exact order -- the frag_extent_norm aux
    feature (added Wed 7 Oct 2026, see module docstring). Computed
    straight from the real per-point xyz in <frame>_<cav>_points.npz
    (the same file fragment_vehicle_ids_for_unit() already reads below),
    grouped by its real per-point fragment_id column -- no crop-caching,
    no opencood-env dependency. A fragment with 0 or 1 points gets extent
    0.0 (nothing to measure a spread over), not NaN. Returns an
    all-zeros array (same convention as a missing violation_lookup entry)
    if <frame>_<cav>_points.npz doesn't exist for this unit -- callers
    that got this far already have frags/pairs from the sibling
    *_fragments.parquet, so this should only be missing if the two files
    ever go out of sync on disk."""
    points_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_points.npz"
    if not points_path.exists():
        return np.zeros(len(fragment_ids), dtype=np.float32)

    d = np.load(points_path)
    point_fragment_id = d["fragment_id"]
    xy = d["xyz"][:, :2]

    extent_by_frag = {}
    for fid in np.unique(point_fragment_id).tolist():
        pts = xy[point_fragment_id == fid]
        if len(pts) < 2:
            extent_by_frag[fid] = 0.0
            continue
        centroid = pts.mean(axis=0)
        extent_by_frag[fid] = float(np.sqrt(np.mean(np.sum((pts - centroid) ** 2, axis=1))))

    raw = np.array([extent_by_frag.get(fid, 0.0) for fid in fragment_ids], dtype=np.float32)
    return np.log1p(raw).astype(np.float32)


def _pairwise_interposed_signal(
    scenario_id: str,
    frame_id: str,
    cav_id,
    repair_labels_dir,
    frag_i_ids,
    frag_j_ids,
    centroid_by_frag: dict,
    buffer_radius_m: float = 0.4,
    margin: float = 0.15,
) -> np.ndarray:
    """Physics-based PAIRWISE feature (Thu 8 Oct 2026, GAP #1 follow-up):
    for each requested (frag_i_ids[k], frag_j_ids[k]) pair, counts real
    LiDAR points belonging to some OTHER fragment that sit on the
    straight-line segment between the two centroids. A solid rigid body
    can't be pierced by another solid object -- if a third fragment's
    points lie physically between two candidate pieces, that's direct
    evidence AGAINST merging them into one vehicle.

    This is deliberately computed per-pair from the exact two fragments in
    question, unlike free_occ_conflict_area/tracking_affinity (aux indices
    6/7, P4's occupancy-consistency scalars) which are frame-level
    aggregates broadcast identically to every fragment in the frame no
    matter which pair is being scored -- a real run (Thu 8 Oct 2026, this
    same day) showed that frame-level broadcast regressing different-
    vehicle FPR (7.69% -> 15.38% raw-scale, 10.44% after log1p rescaling,
    both worse than the 7.69% standing best). This feature is the direct
    fix for that weakness: pair-specific, not frame-wide.

    Reads the same <frame>_<cav>_points.npz that _fragment_extent_norm()
    above reads (real per-point xyz + fragment_id) -- no new on-disk
    dependency. Returns log1p(blocking point count) per pair, zero-filled
    (not NaN) wherever points.npz is missing or either fragment's centroid
    is unknown, same convention as every other real-data aux feature in
    this file.

    buffer_radius_m: lateral distance (metres) from the segment for a
    point to count as "on" it -- a feature-computation parameter (same
    status as get_free_space's angle_bins/max_range in
    P4_defence/consistency_layer.py), not a hardcoded merge-decision
    threshold; the model still learns how much to weigh the resulting
    count. margin: fraction of the segment length excluded at each end
    (projection parameter t in (margin, 1-margin)) so points right at
    either fragment's own edge don't trivially count as "interposed".
    """
    n_pairs = len(frag_i_ids)
    out = np.zeros(n_pairs, dtype=np.float32)
    if n_pairs == 0:
        return out

    points_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_points.npz"
    if not points_path.exists():
        return out

    d = np.load(points_path)
    point_fragment_id = d["fragment_id"]
    xy = d["xyz"][:, :2].astype(np.float64)

    for k in range(n_pairs):
        fid_i, fid_j = frag_i_ids[k], frag_j_ids[k]
        ci = centroid_by_frag.get(fid_i)
        cj = centroid_by_frag.get(fid_j)
        if ci is None or cj is None:
            continue
        ci = np.asarray(ci, dtype=np.float64)
        cj = np.asarray(cj, dtype=np.float64)
        seg = cj - ci
        seg_len_sq = float(seg @ seg)
        if seg_len_sq < 1e-6:
            continue  # degenerate (near-identical centroids) -- nothing meaningful to raycast

        mask_other = (point_fragment_id != fid_i) & (point_fragment_id != fid_j)
        if not np.any(mask_other):
            continue
        pts = xy[mask_other]

        rel = pts - ci
        t = (rel @ seg) / seg_len_sq
        in_range = (t > margin) & (t < (1.0 - margin))
        if not np.any(in_range):
            continue
        proj = ci[None, :] + t[:, None] * seg[None, :]
        perp_dist = np.linalg.norm(pts - proj, axis=1)
        blocked_count = int(np.sum(in_range & (perp_dist < buffer_radius_m)))
        out[k] = np.log1p(blocked_count)

    return out


# Ground-truth real-vehicle id a point belongs to; -1 means the point is
# background (not inside any real vehicle's GT box). Same convention as
# P5_labelling/labelling.py and evaluate_pair_types.py -- this project's
# one oracle definition, not a separate guess.
BACKGROUND_VEHICLE_ID = -1


def fragment_vehicle_ids_for_unit(scenario_id: str, frame_id: str, cav_id: str, repair_labels_dir) -> dict:
    """{fragment_id: None (background) | int vehicle_id | "mixed"} from the
    real per-point ground truth in <frame>_<cav>_points.npz. TRAIN-TIME
    USE ONLY -- for classifying a training pair's bucket so the loss can
    be reweighted (see build_pair_type_weights() below); this is never
    fed to the model as a feature (same status as pos_weight: computed
    from labels/ground truth, used only to shape the loss). Reimplemented
    here rather than imported from evaluate_pair_types.py/
    visualize_corrections.py, which pull in matplotlib/shapely -- dataset.py
    stays a light numpy/pandas-only module (see its own module docstring)."""
    points_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_points.npz"
    if not points_path.exists():
        return {}
    d = np.load(points_path)
    point_fragment_id = d["fragment_id"]
    point_vehicle_id = d["point_vehicle_id"]
    out = {}
    for fid in np.unique(point_fragment_id).tolist():
        vids = point_vehicle_id[point_fragment_id == fid]
        real = vids[vids != BACKGROUND_VEHICLE_ID]
        if len(real) == 0:
            out[fid] = None
        else:
            distinct = set(real.tolist())
            out[fid] = distinct.pop() if len(distinct) == 1 else "mixed"
    return out


def build_pair_type_weights(
    scenario_id: str,
    frame_id: str,
    cav_id: str,
    crop_lookup: dict,
    repair_labels_dir,
    different_vehicle_weight: float = 1.0,
    other_weight: float = 1.0,
) -> np.ndarray | None:
    """Per-pair loss weight for this (frame, cav)'s pairs, aligned row-for-
    row with build_sample()'s pair_index (both are built from the same
    _load_and_filter_frags_pairs() call, same `pairs` dataframe, same row
    order -- see that function's own docstring). Every pair gets weight
    1.0 EXCEPT a NEGATIVE (label=0) pair where both fragments are real,
    ground-truth-DIFFERENT vehicles -- the specific failure bucket found
    Mon 5 Oct 2026 (decisions.md): the head was absorbing small sliver
    fragments into a nearby, differently-sized, spatially-FAR fragment
    from a different real vehicle, a pattern this bucket is only ~1.2% of
    all pairs, easily swamped in the gradient by the much larger
    background-background bucket. `different_vehicle_weight` multiplies
    that bucket's loss; everything else (background-background,
    background-vehicle, same-vehicle, and any "mixed"/ambiguous fragment
    pair) stays at weight 1.0. Returns None under the same conditions
    build_sample() returns None (nothing usable for this unit)."""
    loaded = _load_and_filter_frags_pairs(scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir)
    if loaded is None:
        return None
    frags, pairs, local_index = loaded
    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.zeros((0,), dtype=np.float32)

    frag_vid = fragment_vehicle_ids_for_unit(scenario_id, frame_id, cav_id, repair_labels_dir)
    weights = np.full(n_pairs, other_weight, dtype=np.float32)  # Fri 9 Oct 2026: separate other_weight knob for every non-different-vehicle bucket
    # Fri 9 Oct 2026 fix: pandas Series.map(dict) silently coerces a
    # dict value of None to NaN when mapping onto an int64 column --
    # confirmed on real data: float('nan') != float('nan') is True in
    # Python, so every background-background pair (both fragments
    # None -> NaN) was passing the "a != b" check below and getting
    # incorrectly flagged as a different-vehicle negative. On the
    # 34-scenario val set this inflated the bucket from a true 483
    # pairs to 5736 -- ~91% of what the loss was upweighting 8x was
    # actually ordinary background-background pairs, drowning out the
    # real signal. Plain Python dict.get() over a list preserves None
    # correctly and sidesteps pandas' dtype coercion entirely.
    vid_i = [frag_vid.get(fid) for fid in pairs["fragment_i"]]
    vid_j = [frag_vid.get(fid) for fid in pairs["fragment_j"]]
    labels = pairs["affinity_label"].to_numpy()
    for k in range(n_pairs):
        a, b = vid_i[k], vid_j[k]
        if (
            labels[k] == 0
            and a is not None and b is not None
            and a != "mixed" and b != "mixed"
            and a != b
        ):
            weights[k] = different_vehicle_weight
    return weights


def build_pair_type_weights_hard_negative(
    scenario_id: str,
    frame_id: str,
    cav_id: str,
    crop_lookup: dict,
    repair_labels_dir,
    different_vehicle_weight: float = 8.0,
    hard_negative_weight: float = 20.0,
    hard_dist_norm_threshold: float = 0.68,
    hard_size_gap_threshold: float = 1.0,
    other_weight: float = 1.0,
) -> np.ndarray | None:
    """Hard-negative-mining variant of build_pair_type_weights(), added
    Tue 7 Oct 2026 (decisions.md) as "option 1" of two candidate fixes
    for the different-vehicle FPR gap left over after the flat
    different_vehicle_weight=8.0 run (43.41%% -> 12.09%% FPR, still ~3x
    denoise's 3.85%%). The flat version upweights EVERY different-vehicle
    negative pair equally, spreading the correction across easy and hard
    cases alike. diagnose_false_merges.py (Mon 5 Oct 2026) found the
    head's actual false merges concentrate on one specific pattern: a
    small sliver fragment absorbed into a large, SPATIALLY-FAR fragment
    from a different real vehicle (10/15 sampled culprit pairs had
    dist_norm 0.72-0.99, i.e. well past the denoise threshold, with a
    large point-count gap, abs_delta_log_n often 1.3-5.5). This function
    gives that specific pattern extra weight on top of the flat
    different-vehicle bump, instead of (or as well as) spreading the
    correction evenly -- a learned-signal fix, not a hardcoded distance
    cutoff: the model still decides what to do with a far, mismatched
    pair, it's just shown many more of exactly those during training.

    Same alignment contract as build_pair_type_weights(): one weight per
    row of build_sample()'s pair_index, built from the identical
    _load_and_filter_frags_pairs() call.

    different_vehicle_weight: baseline multiplier on ALL different-vehicle
        negative pairs (same meaning as build_pair_type_weights()'s own
        argument -- kept here so the hard subset is additionally boosted
        ON TOP of the established 8x baseline, not instead of it).
    hard_negative_weight: multiplier applied instead of
        different_vehicle_weight for the "hard" subset -- a different-
        vehicle negative pair that is BOTH spatially far
        (dist_norm > hard_dist_norm_threshold) AND size-mismatched
        (abs_delta_log_n > hard_size_gap_threshold). Should be set
        >= different_vehicle_weight for this to actually concentrate
        more correction on the hard cases rather than less.
    hard_dist_norm_threshold: default 0.68 matches denoise's own chosen
        threshold (evaluate_pair_types.py / decisions.md, Fri 2 Oct 2026)
        -- reused here only to IDENTIFY which training pairs are "far",
        not applied as an inference-time cutoff on the model's own
        predictions (that would just be denoise's rule hardcoded back in,
        rejected Tue 7 Oct 2026 as not a learned fix and config-dependent).
    hard_size_gap_threshold: default 1.0 sits below the observed 1.3-5.5
        range of real culprit pairs' abs_delta_log_n, so it includes the
        full observed pattern plus some margin.
    """
    loaded = _load_and_filter_frags_pairs(scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir)
    if loaded is None:
        return None
    frags, pairs, local_index = loaded
    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.zeros((0,), dtype=np.float32)

    frag_vid = fragment_vehicle_ids_for_unit(scenario_id, frame_id, cav_id, repair_labels_dir)
    weights = np.full(n_pairs, other_weight, dtype=np.float32)  # Fri 9 Oct 2026: separate other_weight knob for every non-different-vehicle bucket
    # Fri 9 Oct 2026 fix: pandas Series.map(dict) silently coerces a
    # dict value of None to NaN when mapping onto an int64 column --
    # confirmed on real data: float('nan') != float('nan') is True in
    # Python, so every background-background pair (both fragments
    # None -> NaN) was passing the "a != b" check below and getting
    # incorrectly flagged as a different-vehicle negative. On the
    # 34-scenario val set this inflated the bucket from a true 483
    # pairs to 5736 -- ~91% of what the loss was upweighting 8x was
    # actually ordinary background-background pairs, drowning out the
    # real signal. Plain Python dict.get() over a list preserves None
    # correctly and sidesteps pandas' dtype coercion entirely.
    vid_i = [frag_vid.get(fid) for fid in pairs["fragment_i"]]
    vid_j = [frag_vid.get(fid) for fid in pairs["fragment_j"]]
    labels = pairs["affinity_label"].to_numpy()

    n_points_by_frag = dict(zip(frags["fragment_id"], frags["n_points"]))
    dist_norm = pairs["centroid_distance_m"].to_numpy(dtype=np.float32) / np.float32(CANDIDATE_PAIR_RADIUS_M)
    log_n_i = np.log1p(pairs["fragment_i"].map(n_points_by_frag).to_numpy(dtype=np.float32))
    log_n_j = np.log1p(pairs["fragment_j"].map(n_points_by_frag).to_numpy(dtype=np.float32))
    abs_delta_log_n = np.abs(log_n_i - log_n_j)

    for k in range(n_pairs):
        a, b = vid_i[k], vid_j[k]
        if (
            labels[k] == 0
            and a is not None and b is not None
            and a != "mixed" and b != "mixed"
            and a != b
        ):
            is_hard = (dist_norm[k] > hard_dist_norm_threshold) and (abs_delta_log_n[k] > hard_size_gap_threshold)
            weights[k] = hard_negative_weight if is_hard else different_vehicle_weight
    return weights


def build_sample(
    scenario_id: str,
    frame_id: str,
    cav_id: str,
    crop_lookup: dict,
    repair_labels_dir,
    arm: str,
    degradation_config: dict,
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
) -> CorrectionHeadSample | None:
    """Builds one training unit for (scenario_id, frame_id, cav_id), or
    returns None if none of this (frame, cav)'s fragments have a cached,
    valid BEV crop (no crops means nothing CorrectionHead.encoder can run
    on for this unit -- see module docstring GAP #2 for why this currently
    happens for almost every real (frame, cav))."""
    loaded = _load_and_filter_frags_pairs(scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir)
    if loaded is None:
        return None
    frags, pairs, local_index = loaded
    n = len(frags)

    crops = np.stack(
        [crop_lookup[(str(frame_id), str(cav_id), int(fid))] for fid in frags["fragment_id"]], axis=0
    ).astype(np.float32)

    density = 0.5  # fallback when no sensing_profile row matches -- see build_vehicle_partition's
    # own identical "unknown" fallback convention in P6_features/features.py.
    if sensing_profile is not None and not sensing_profile.empty:
        match = sensing_profile[
            (sensing_profile["scenario_id"] == scenario_id)
            & (sensing_profile["frame_id"] == frame_id)
            & (sensing_profile["cav_id"] == str(cav_id))
        ]
        if not match.empty:
            density = float(match.iloc[0]["clustered_fraction"])

    deg_severity = _degradation_severity(arm, scenario_id, cav_id, degradation_config)

    occ_violation = 0.0
    perc_violation = 0.0
    if violation_lookup is not None:
        v = violation_lookup.get((scenario_id, frame_id))
        if v is not None:
            raw_occ_violation, raw_perc_violation = v
            # Raw values are unbounded m^2 areas (mean ~12-15, max ~85) -- vastly
            # larger scale than every other aux feature. log1p brings them onto a
            # comparable scale, matching the convention already used for
            # n_points and frag_extent_norm. Values are always >= 0 (areas), so
            # log1p is safe without abs()/clipping.
            occ_violation = float(np.log1p(raw_occ_violation))
            perc_violation = float(np.log1p(raw_perc_violation))

    centroids = frags[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    range_from_ego_norm = np.clip(np.linalg.norm(centroids, axis=1) / _LIDAR_MAX_RANGE_M, 0.0, 1.0)
    bearing_norm = np.arctan2(centroids[:, 1], centroids[:, 0]) / np.pi

    frag_extent_norm = _fragment_extent_norm(
        scenario_id, frame_id, cav_id, repair_labels_dir, frags["fragment_id"].tolist()
    )

    frag_aux = np.stack(
        [
            np.log1p(frags["n_points"].to_numpy(dtype=np.float32)),
            frags["is_recovered_noise"].to_numpy(dtype=np.float32),
            range_from_ego_norm.astype(np.float32),
            bearing_norm.astype(np.float32),
            np.full(n, density, dtype=np.float32),
            np.full(n, deg_severity, dtype=np.float32),
            np.full(n, occ_violation, dtype=np.float32),
            np.full(n, perc_violation, dtype=np.float32),
            frag_extent_norm,
        ],
        axis=1,
    ).astype(np.float32)
    assert frag_aux.shape == (n, FRAGMENT_AUX_DIM), (frag_aux.shape, FRAGMENT_AUX_DIM, AUX_FEATURE_NAMES)

    if pairs.empty:
        pair_index = np.zeros((0, 2), dtype=np.int64)
        pair_aux = np.zeros((0, PAIR_AUX_DIM), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.float32)
    else:
        n_points_by_frag = dict(zip(frags["fragment_id"], frags["n_points"]))
        i_local = pairs["fragment_i"].map(local_index).to_numpy(dtype=np.int64)
        j_local = pairs["fragment_j"].map(local_index).to_numpy(dtype=np.int64)
        pair_index = np.stack([i_local, j_local], axis=1)

        dist_norm = pairs["centroid_distance_m"].to_numpy(dtype=np.float32) / np.float32(
            CANDIDATE_PAIR_RADIUS_M
        )
        log_n_i = np.log1p(pairs["fragment_i"].map(n_points_by_frag).to_numpy(dtype=np.float32))
        log_n_j = np.log1p(pairs["fragment_j"].map(n_points_by_frag).to_numpy(dtype=np.float32))
        abs_delta_log_n = np.abs(log_n_i - log_n_j)

        centroid_by_frag = dict(zip(frags["fragment_id"].tolist(), map(tuple, centroids)))
        interposed_signal = _pairwise_interposed_signal(
            scenario_id,
            frame_id,
            cav_id,
            repair_labels_dir,
            pairs["fragment_i"].tolist(),
            pairs["fragment_j"].tolist(),
            centroid_by_frag,
        )

        pair_aux = np.stack([dist_norm, abs_delta_log_n, interposed_signal], axis=1).astype(np.float32)
        labels = pairs["affinity_label"].to_numpy(dtype=np.float32)

    return CorrectionHeadSample(scenario_id, frame_id, str(cav_id), crops, frag_aux, pair_index, pair_aux, labels)


def build_samples_for_scenario(
    scenario_id: str,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    degradation_config: dict | None = None,
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
) -> list[CorrectionHeadSample]:
    """All of one scenario's (frame, cav) training units that have at
    least one valid cached crop. Skips (frame, cav) combos that
    discover_frame_cav_pairs() lists but have zero crop coverage -- see
    module docstring GAP #2."""
    if degradation_config is None:
        degradation_config = load_degradation_config()

    crop_lookup, n_dropped_invalid = load_scenario_crops(crops_dir, scenario_id, arm)

    samples = []
    n_skipped_no_crops = 0
    for frame_id, cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
        sample = build_sample(
            scenario_id,
            frame_id,
            cav_id,
            crop_lookup,
            repair_labels_dir,
            arm,
            degradation_config,
            sensing_profile,
            violation_lookup,
        )
        if sample is None:
            n_skipped_no_crops += 1
            continue
        samples.append(sample)

    if not samples:
        import warnings

        warnings.warn(
            f"build_samples_for_scenario({scenario_id!r}, arm={arm!r}): 0 usable "
            f"training samples out of {n_skipped_no_crops} (frame, cav) combos with "
            f"repair labels ({n_dropped_invalid} crop rows dropped as valid=False). "
            f"Crop caching likely hasn't covered this scenario/arm's frames yet -- "
            f"see dataset.py's module docstring, GAP #2."
        )

    return samples


def build_samples_for_scenario_with_pair_weights(
    scenario_id: str,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    degradation_config: dict | None = None,
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
    different_vehicle_weight: float = 1.0,
    other_weight: float = 1.0,
) -> tuple[list[CorrectionHeadSample], list[np.ndarray]]:
    """Same traversal as build_samples_for_scenario() (same crop_lookup,
    same discover_frame_cav_pairs() order, same per-unit skip rule), but
    also returns a per-sample pair-weight array (build_pair_type_weights())
    built from the identical inputs -- so `weights[i]` always has exactly
    `len(samples[i].labels)` entries and lines up positionally with it.
    TRAIN-TIME USE ONLY (see build_pair_type_weights()'s own docstring) --
    used by train_v1.py to reweight the different-vehicle-negative bucket
    in the loss; `different_vehicle_weight=1.0` (the default) makes this
    function behave identically to build_samples_for_scenario() plus an
    all-ones weight per sample, so it's safe to call unconditionally."""
    if degradation_config is None:
        degradation_config = load_degradation_config()

    crop_lookup, n_dropped_invalid = load_scenario_crops(crops_dir, scenario_id, arm)

    samples = []
    weights = []
    n_skipped_no_crops = 0
    for frame_id, cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
        sample = build_sample(
            scenario_id,
            frame_id,
            cav_id,
            crop_lookup,
            repair_labels_dir,
            arm,
            degradation_config,
            sensing_profile,
            violation_lookup,
        )
        if sample is None:
            n_skipped_no_crops += 1
            continue
        w = build_pair_type_weights(
            scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, different_vehicle_weight, other_weight
        )
        assert w is not None and len(w) == len(sample.labels), (
            f"pair-weight/sample misalignment for {scenario_id}/{frame_id}/{cav_id}: "
            f"{None if w is None else len(w)} weights vs {len(sample.labels)} pairs -- "
            f"build_sample() and build_pair_type_weights() should never disagree, both "
            f"filter via the same _load_and_filter_frags_pairs()."
        )
        samples.append(sample)
        weights.append(w)

    if not samples:
        import warnings

        warnings.warn(
            f"build_samples_for_scenario_with_pair_weights({scenario_id!r}, arm={arm!r}): 0 usable "
            f"training samples out of {n_skipped_no_crops} (frame, cav) combos with "
            f"repair labels ({n_dropped_invalid} crop rows dropped as valid=False)."
        )

    return samples, weights


def build_samples_for_scenario_with_hard_negative_weights(
    scenario_id: str,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    degradation_config: dict | None = None,
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
    different_vehicle_weight: float = 8.0,
    hard_negative_weight: float = 20.0,
    hard_dist_norm_threshold: float = 0.68,
    hard_size_gap_threshold: float = 1.0,
    other_weight: float = 1.0,
) -> tuple[list[CorrectionHeadSample], list[np.ndarray]]:
    """Hard-negative-mining counterpart to
    build_samples_for_scenario_with_pair_weights() -- same traversal and
    alignment guarantees, but calls build_pair_type_weights_hard_negative()
    instead of build_pair_type_weights(). See that function's docstring
    for the "option 1" rationale (decisions.md, Tue 7 Oct 2026)."""
    if degradation_config is None:
        degradation_config = load_degradation_config()

    crop_lookup, n_dropped_invalid = load_scenario_crops(crops_dir, scenario_id, arm)

    samples = []
    weights = []
    n_skipped_no_crops = 0
    for frame_id, cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
        sample = build_sample(
            scenario_id,
            frame_id,
            cav_id,
            crop_lookup,
            repair_labels_dir,
            arm,
            degradation_config,
            sensing_profile,
            violation_lookup,
        )
        if sample is None:
            n_skipped_no_crops += 1
            continue
        w = build_pair_type_weights_hard_negative(
            scenario_id,
            frame_id,
            cav_id,
            crop_lookup,
            repair_labels_dir,
            different_vehicle_weight=different_vehicle_weight,
            hard_negative_weight=hard_negative_weight,
            hard_dist_norm_threshold=hard_dist_norm_threshold,
            hard_size_gap_threshold=hard_size_gap_threshold,
            other_weight=other_weight,
        )
        assert w is not None and len(w) == len(sample.labels), (
            f"pair-weight/sample misalignment for {scenario_id}/{frame_id}/{cav_id}: "
            f"{None if w is None else len(w)} weights vs {len(sample.labels)} pairs."
        )
        samples.append(sample)
        weights.append(w)

    if not samples:
        import warnings

        warnings.warn(
            f"build_samples_for_scenario_with_hard_negative_weights({scenario_id!r}, arm={arm!r}): "
            f"0 usable training samples out of {n_skipped_no_crops} (frame, cav) combos "
            f"({n_dropped_invalid} crop rows dropped as valid=False)."
        )

    return samples, weights


class CorrectionHeadDataset(Dataset):
    """torch Dataset over CorrectionHeadSample units, built from one or
    more scenarios. Returns raw numpy-backed samples; use collate_batches()
    (directly, or as a DataLoader `collate_fn`) to get torch tensors."""

    def __init__(self, samples: list[CorrectionHeadSample]):
        if torch is None:
            raise ImportError(
                "CorrectionHeadDataset needs torch (for torch.utils.data.Dataset). "
                "The pure-numpy builder functions in this module (build_sample, "
                "build_samples_for_scenario) work without torch -- use those directly "
                "if you only need to inspect/test the data, e.g. in an environment "
                "without the opencood conda env."
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CorrectionHeadSample:
        return self.samples[idx]


def collate_batches(samples: list[CorrectionHeadSample]):
    """Concatenates several CorrectionHeadSample units into one flat batch
    matching CorrectionHead.forward()'s arguments: crops (N,384,7,7),
    frag_aux (N,8), pair_index (P,2), pair_aux (P,2), labels (P,). Each
    sample's own pair_index is offset by the running fragment count so
    indices stay valid into the concatenated crops/frag_aux. Returns numpy
    arrays if torch isn't available, else torch tensors (float32 for
    crops/frag_aux/pair_aux/labels, int64 for pair_index) -- matching
    verify_on_device.py's own synthetic-batch dtypes exactly. An empty
    `samples` list returns all-empty arrays with the right shapes/dtypes
    rather than raising (np.concatenate itself refuses an empty list of
    arrays -- caught by this module's own test suite)."""
    if not samples:
        empty = (
            np.zeros((0, 384, 7, 7), dtype=np.float32),
            np.zeros((0, FRAGMENT_AUX_DIM), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, PAIR_AUX_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
        if torch is None:
            return empty
        crops, frag_aux, pair_index, pair_aux, labels = empty
        return (
            torch.from_numpy(crops).float(),
            torch.from_numpy(frag_aux).float(),
            torch.from_numpy(pair_index).long(),
            torch.from_numpy(pair_aux).float(),
            torch.from_numpy(labels).float(),
        )

    crops = np.concatenate([s.crops for s in samples], axis=0)
    frag_aux = np.concatenate([s.frag_aux for s in samples], axis=0)

    pair_index_parts = []
    offset = 0
    for s in samples:
        pair_index_parts.append(s.pair_index + offset)
        offset += s.n_fragments
    pair_index = (
        np.concatenate(pair_index_parts, axis=0) if pair_index_parts else np.zeros((0, 2), dtype=np.int64)
    )

    pair_aux = np.concatenate([s.pair_aux for s in samples], axis=0) if samples else np.zeros((0, PAIR_AUX_DIM), dtype=np.float32)
    labels = np.concatenate([s.labels for s in samples], axis=0) if samples else np.zeros((0,), dtype=np.float32)

    if torch is None:
        return crops, frag_aux, pair_index, pair_aux, labels

    return (
        torch.from_numpy(crops).float(),
        torch.from_numpy(frag_aux).float(),
        torch.from_numpy(pair_index).long(),
        torch.from_numpy(pair_aux).float(),
        torch.from_numpy(labels).float(),
    )


def make_scenario_splits(manifest_path=DEFAULT_MANIFEST_PATH, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0):
    """Thin wrapper over P6_features.features.build_scenario_split, so
    callers of this module don't need a separate P6 import for the one
    thing they need from it. Splits by scenario, never frame (see that
    function's own docstring) -- with only 2 qualifying scenarios right
    now, this is known to produce an empty val/test split, which is the
    documented, expected degenerate case, not a bug to work around here."""
    import pandas as pd

    from P6_features.features import build_scenario_split

    manifest = pd.read_parquet(manifest_path)
    return build_scenario_split(manifest, val_frac=val_frac, test_frac=test_frac, seed=seed)
