"""
P3 -- Frozen perception pass + caching. Implemented 17 Sep 2026 (Week 2,
compressed into one sitting -- originally Fri 11 Sep, Week 2 pilot).

Reuses the exact PointPillars checkpoint + inference path validated on Sat 5
Sep 2026 (epoch 30, late-fusion, OpenCOOD model zoo) and its environment
fix-ups (num_workers=0, the extract_timestamps() Windows-path patch already
applied in basedataset.py -- see decisions.md, Sat 5 Sep entry). **This
script cannot be run or tested in this session** -- it needs the `opencood`
conda env, CUDA, and the actual dataset on the Windows machine, none of
which are available here. It's built by reading OpenCOOD's own
basedataset.py / late_fusion_dataset.py / inference.py source directly
(17 Sep 2026) so the integration points below are grounded in the real API,
not guessed -- but it has NOT been execution-verified. Run it, and if
anything breaks, paste the traceback back for a fix.

--- Design: how this differs from OpenCOOD's own inference.py ---

OpenCOOD's inference.py (opencood/tools/inference.py) iterates its ENTIRE
configured validate_dir via a DataLoader with shuffle=False -- every
scenario, every frame. We only want the ~20-scenario P0 manifest's
qualifying frames (n_cav>=3), capped at --max-frames-per-scenario, and we
need to inject P2 degradation into a specific (scenario, cav) before that
frame's point cloud reaches the detector. Neither is something the stock
DataLoader iteration supports, so this file:

  1. Subclasses LateFusionDataset (`ManifestFilteredLateFusionDataset`) and
     overrides `retrieve_base_data()` to inject fog/compression (via P2's
     inject_fog/inject_compression) into a non-ego CAV's raw `lidar_np`
     array, keyed by a degradation_assignment dict the caller builds from
     the manifest + P6's vehicle partition. Ego's own point cloud is never
     degraded -- P2 degrades "one benign CAV at a time", i.e. a specific
     non-ego reporter, which is also the CAV P4/P5 will attribute the
     resulting flag/label to.
  2. **Simplification, flagged explicitly:** `inject_compression` is applied
     to the raw `lidar_np` array (intensity column only, see the bug-fix
     note inside `retrieve_base_data` below), NOT to the internal voxel/BEV
     tensor `pre_processor.preprocess()` produces internally. Hooking after
     that step would require depending on `SpVoxelPreprocessor`'s internal
     return-dict layout (voxel features / coords / num-points keys), which
     isn't documented anywhere read this session and couldn't be verified
     without actually running it. Quantising the recorded intensity value
     still captures the essential effect P2 is after (precision loss in the
     representation reaching the detector) without coupling to internals
     nobody has confirmed, and -- as of the 17 Sep bug fix below -- without
     risking destroying the point cloud's actual geometry. Revisit this once
     the pipeline runs a full pilot, if the effect looks too weak/strong
     relative to fog's effect on FPR.
  3. Builds an explicit (scenario_id, frame_id) -> dataset `idx` lookup by
     replicating the same len_record/timestamp-key resolution
     `BaseDataset.retrieve_base_data` uses internally (that mapping isn't
     exposed publicly), so only the manifest's chosen frames are ever
     touched -- no full-dataset pass.
  4. Runs inference frame-by-frame (batch size 1, matching
     `collate_batch_test`'s own `assert len(batch) <= 1` requirement) using
     the same `inference_utils.inference_late_fusion` call inference.py
     itself uses, so the actual model call is identical to the already
     GPU-validated path.
  5. Caches, per scenario, an .npz with per-frame predicted boxes/scores,
     GT boxes, and wall-clock timing -- one file per scenario under
     `out_dir`, plus a small JSON manifest of what was cached (arm, config
     used, disk size, per-scenario timing) so P4/P5 can consume it and so
     the Fri 11 Sep done-when ("know the per-scenario cost in minutes and
     GB") has a concrete answer.
  6. **BEV feature crop caching (Thu 1 Oct 2026 addition -- Week 4' of the
     rewritten execution_plan_daily.md).** Was deliberately deferred on 17
     Sep pending reading PointPillars' actual backbone module structure;
     that's now been read (opencood/models/point_pillar.py,
     opencood/models/sub_modules/base_bev_backbone.py, and this project's
     real pointpillar_late_fusion/config.yaml) and grounds the
     implementation below precisely rather than guessing:
       - `model.backbone` (a BaseBEVBackbone) is a real nn.Module whose
         forward() returns a dict containing `spatial_features_2d`, a
         (1, C, feat_h, feat_w) BEV feature map already in the SAME local
         lidar frame each CAV's own points/DBSCAN clusters live in (late
         fusion never transforms a CAV's points out of its own frame
         before the detector runs).
       - `inference_utils.inference_late_fusion` calls
         `model(cav_content)` once per cav_id in a plain loop
         (`for cav_id, cav_content in batch_data.items()`) -- but doesn't
         expose the backbone's intermediate output, so it can't be reused
         as-is for cropping. `run_frozen_inference_and_cache_with_crops`
         below replicates that same loop itself (still calling
         `dataset.post_process` afterward exactly as inference_utils does,
         so detection output is identical either way) and additionally
         reads each per-CAV `spatial_features_2d` via a forward hook
         registered once on `model.backbone`.
       - Converting a candidate fragment's real-world (x, y) centroid (from
         `run_cluster_repair_labelling.py`'s `*_fragments.parquet`, already
         in this exact same local frame) into a (row, col) pixel index in
         `spatial_features_2d` is done via `_pixel_from_xy()`, using this
         project's real `cav_lidar_range`/`voxel_size` config values PLUS
         the actual captured feature map's shape at runtime -- not a
         hardcoded "2x downsample" constant -- so a later backbone/config
         change can't silently misalign every cached crop without the
         shapes to notice.
     **Not execution-verified** (same standing caveat as this whole file --
     needs the real `opencood` env + GPU + checkpoint; run it and paste
     back any traceback for a fix). In particular the exact key `model`
     stores its backbone under (`model.backbone`) and the exact dict key
     `spatial_features_2d` are read directly from point_pillar.py/
     base_bev_backbone.py's real source as of Thu 1 Oct 2026 -- if the
     checkpoint's actual state differs (e.g. a config using a different
     `core_method` than plain `point_pillar`), the hook registration will
     raise `AttributeError` immediately and loudly rather than silently
     capturing nothing, which is the fail-safe this was written to prefer.

Usage (from the MastersProj repo root, opencood conda env active):
    python P3_perception/perception_cache.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data" ^
        --model-dir "C:\\path\\to\\pointpillar_late_fusion\\checkpoint_dir" ^
        --config "pointpillar_late_fusion/config.yaml" ^
        --out-dir P3_perception/cache ^
        --max-scenarios 20 --max-frames-per-scenario 10 ^
        --arm A
Then repeat with --arm B --degradation-config <path to a small JSON/YAML
mapping (scenario_id, cav_id) -> {type, severity}, e.g. from P6's vehicle
partition> to cache the degraded-benign arm.

To additionally cache BEV feature crops for the correction head (Thu 1 Oct
2026), add --cache-bev-crops and --repair-labels-dir (defaults to
P4_defence/cache/repair_labels, run_cluster_repair_labelling.py's real
output dir):
    python P3_perception/perception_cache.py ^
        --opv2v-root "..." --model-dir "..." --out-dir P3_perception/cache ^
        --arm A --cache-bev-crops --repair-labels-dir P4_defence/cache/repair_labels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
sys.path.insert(0, str(REPO_ROOT / "P2_degradation"))

import torch  # noqa: E402
import pandas as pd  # noqa: E402

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils, inference_utils  # noqa: E402
from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset  # noqa: E402
from opencood.utils.common_utils import torch_tensor_to_numpy  # noqa: E402

from degrade import inject_fog, inject_compression  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"


class ManifestFilteredLateFusionDataset(LateFusionDataset):
    """
    LateFusionDataset, but with an (scenario_id, frame_id) -> idx lookup
    exposed, and optional per-(scenario_id, cav_id) degradation injection
    hooked into retrieve_base_data(). See module docstring for why.
    """

    def __init__(self, params, opv2v_root, degradation_assignment=None, train=False):
        # BaseDataset reads params['validate_dir'] when train=False -- point
        # it at the actual data root the caller passed, overriding whatever
        # the yaml config file itself said (keeps this script usable against
        # any data root without editing the shared config.yaml).
        params = dict(params)
        params["validate_dir"] = opv2v_root
        params["root_dir"] = opv2v_root
        super().__init__(params, visualize=True, train=train)

        # BaseDataset.__init__ scans scenario folders internally but doesn't
        # keep the folder-name list around -- recompute it the same way
        # (sorted dir listing) so idx -> scenario_id resolution is possible.
        self.scenario_ids = sorted(
            x for x in os.listdir(opv2v_root) if os.path.isdir(os.path.join(opv2v_root, x))
        )
        self.degradation_assignment = degradation_assignment or {}
        self._build_frame_index()

    def _build_frame_index(self):
        """(scenario_id, frame_id) -> global dataset idx, by replicating the
        same len_record-based resolution retrieve_base_data() uses."""
        self.frame_index = {}
        prev_len = 0
        for scenario_index, cumulative_len in enumerate(self.len_record):
            scenario_id = self.scenario_ids[scenario_index]
            scenario_database = self.scenario_database[scenario_index]
            n_frames_this_scenario = cumulative_len - prev_len
            for timestamp_index in range(n_frames_this_scenario):
                timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
                global_idx = prev_len + timestamp_index
                self.frame_index[(scenario_id, timestamp_key)] = global_idx
            prev_len = cumulative_len

    def idx_for(self, scenario_id: str, frame_id: str) -> int:
        key = (scenario_id, frame_id)
        if key not in self.frame_index:
            raise KeyError(
                f"({scenario_id}, {frame_id}) not found in this dataset root -- "
                f"check --opv2v-root points at the right data and the frame_id "
                f"matches the manifest exactly (zero-padding etc.)"
            )
        return self.frame_index[key]

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        data = super().retrieve_base_data(idx, cur_ego_pose_flag)
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_id = self.scenario_ids[scenario_index]

        for cav_id, cav_content in data.items():
            if cav_content["ego"]:
                continue  # never degrade ego's own report -- see module docstring
            assignment = self.degradation_assignment.get((scenario_id, cav_id))
            if assignment is None:
                continue
            deg_type = assignment["type"]
            severity = assignment["severity"]
            seed = assignment.get("seed", 0)
            points = cav_content["lidar_np"]
            if deg_type == "fog":
                cav_content["lidar_np"] = inject_fog(points, severity, seed)
            elif deg_type == "compression":
                # BUG FIX (17 Sep 2026, found via real run on scenario
                # 2021_09_03_22_43_57): quantising the WHOLE (x,y,z,intensity)
                # array with one shared vmin/vmax collapses x/y/z geometry
                # into a handful of coarse spatial buckets at severity 2
                # (4-bit -> 16 levels across a ~140m span) -- coarse enough
                # that mask_points_by_range/mask_ego_points filtered out
                # EVERY point for the degraded CAV, leaving PointPillars'
                # voxel scatter with zero voxels and crashing with
                # "RuntimeError: max(): Expected reduction dim to be
                # specified for input.numel() == 0." Compression should only
                # ever degrade the recorded return's precision (intensity),
                # never geometry -- quantise column 3 alone, leave x/y/z
                # untouched.
                degraded = points.copy()
                if degraded.shape[1] >= 4:
                    degraded[:, 3] = inject_compression(degraded[:, 3], severity, seed)
                cav_content["lidar_np"] = degraded
            else:
                raise ValueError(f"unknown degradation type {deg_type!r}")
        return data


# --- Thu 1 Oct 2026 addition: BEV feature crop caching for the correction head ---

# Crop window side length in feature-map pixels (odd, so "centred on the
# fragment centroid" is exact). At this project's real config (0.4m voxels,
# 2x backbone downsample -> 0.8m/pixel), 7px ~= 5.6m across -- comfortably
# larger than a single vehicle footprint (~4.5m x 2m) without being so large
# it always includes a neighbouring cluster. Tunable; revisit once the
# correction head's own architecture (Thu 1 Oct 2026 gate) is pinned down --
# this should match whatever receptive field that head actually expects.
BEV_CROP_SIZE_PX = 7  # ~5.6m at this config's 0.8m/px feature-map resolution. Tried 11px (~8.8m) on Mon 5 Oct 2026 to test whether more context would reduce different-vehicle FPR; it regressed instead (see decisions.md), so reverted back to 7.


def _pixel_from_xy(x: float, y: float, lidar_range, voxel_size, grid_shape, feat_shape):
    """
    Converts a point (x, y) in a CAV's own local lidar frame into a
    (row, col) pixel index in that CAV's spatial_features_2d BEV map.

    lidar_range: [xmin, ymin, zmin, xmax, ymax, zmax], this project's real
        model config value (hypes['postprocess']['anchor_args']['cav_lidar_range']).
    voxel_size: [vx, vy, vz], this project's real config value
        (hypes['preprocess']['args']['voxel_size']).
    grid_shape: (grid_h, grid_w) -- the raw voxel grid BEFORE the backbone's
        internal downsampling (hypes['postprocess']['anchor_args']['H'],
        ['W'] -- 200, 352 in this project's real config).
    feat_shape: (feat_h, feat_w) -- spatial_features_2d.shape[2:], read from
        the ACTUAL captured tensor at runtime. Comparing this to grid_shape
        gives the backbone's real downsample factor empirically (currently
        2x in both dims for this project's config -- 3 stride-2 conv blocks
        with upsample_strides [1,2,4] all reconverging at the first block's
        resolution, per base_bev_backbone.py's forward()) rather than
        trusting a hardcoded constant that a later config change could
        silently invalidate.
    """
    xmin, ymin = lidar_range[0], lidar_range[1]
    vx, vy = voxel_size[0], voxel_size[1]
    grid_h, grid_w = grid_shape
    feat_h, feat_w = feat_shape
    m_per_px_x = vx * (grid_w / feat_w)
    m_per_px_y = vy * (grid_h / feat_h)
    col = (x - xmin) / m_per_px_x
    row = (y - ymin) / m_per_px_y
    return row, col


def crop_bev_feature(spatial_features_2d: np.ndarray, row: float, col: float, crop_size_px: int = BEV_CROP_SIZE_PX):
    """
    Crops a (crop_size_px x crop_size_px) window centred on (row, col) out of
    a (C, H, W) BEV feature map (numpy, already detached/moved off-GPU),
    zero-padding wherever the window falls outside the map -- a candidate
    fragment near the sensor's max range legitimately sits close to the
    grid edge, and this should degrade gracefully rather than crash or clip
    asymmetrically.

    Returns (crop, valid): `crop` is always exactly
    (C, crop_size_px, crop_size_px); `valid` is False iff the window's
    centre pixel itself falls outside the map bounds (e.g. a fragment
    centroid beyond the model's own lidar_range, which the detector never
    saw at all) -- the correction head's dataset loader should treat an
    invalid crop as "no signal available", not as legitimate all-zero
    content.
    """
    c, h, w = spatial_features_2d.shape
    half = crop_size_px // 2
    r_center, c_center = int(round(row)), int(round(col))
    valid = 0 <= r_center < h and 0 <= c_center < w

    r0, r1 = r_center - half, r_center - half + crop_size_px
    c0, c1 = c_center - half, c_center - half + crop_size_px

    out = np.zeros((c, crop_size_px, crop_size_px), dtype=spatial_features_2d.dtype)
    src_r0, src_r1 = max(r0, 0), min(r1, h)
    src_c0, src_c1 = max(c0, 0), min(c1, w)
    if src_r0 < src_r1 and src_c0 < src_c1:
        dst_r0, dst_c0 = src_r0 - r0, src_c0 - c0
        out[:, dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = spatial_features_2d[
            :, src_r0:src_r1, src_c0:src_c1
        ]
    return out, valid


def _load_fragment_centroids(repair_labels_dir: str, scenario_id: str, frame_id: str, cav_id: str):
    """
    Reads run_cluster_repair_labelling.py's real *_fragments.parquet for this
    exact (scenario, frame, cav) -- same stem convention that script uses
    (f"{frame_id}_{cav_id}") -- so crops key-join directly back onto that
    table by (scenario_id, frame_id, cav_id, fragment_id). Returns an empty
    list if the file doesn't exist (e.g. this frame/cav wasn't one of the
    ones run_cluster_repair_labelling.py processed, or --repair-labels-dir
    doesn't point at real output yet) -- caching detections should never be
    blocked on repair-labelling having been run first.
    """
    path = os.path.join(repair_labels_dir, scenario_id, f"{frame_id}_{cav_id}_fragments.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    return list(zip(df["fragment_id"], df["centroid_x"], df["centroid_y"]))


def discover_repair_label_frames(repair_labels_dir: str, scenario_id: str) -> list[str]:
    """
    Lists the frame_ids run_cluster_repair_labelling.py actually produced
    fragments for, by scanning <repair_labels_dir>/<scenario_id>/*_fragments.parquet
    filenames (stem convention "<frame_id>_<cav_id>_fragments.parquet" -- the
    same one _load_fragment_centroids() reads). Used by --cache-bev-crops so
    the frames this script runs inference on are EXACTLY the frames
    repair-labelling covers, instead of relying on two independently
    configured frame-selection rules staying in sync by coincidence: this
    script's own --max-frames-per-scenario .head() selection on the P0
    manifest, vs cluster_failure_audit.load_qualifying_scenarios()'s
    evenly-spaced N_FRAMES_PER_SCENARIO pick, which run_cluster_repair_
    labelling.py actually uses. Those two only coincided on 1 of 18
    (frame, cav) combos per scenario in the first --cache-bev-crops run
    (1 Oct 2026) -- see decisions.md's GAP #2 entry, the real coverage gap
    this function exists to close. Returns [] if the scenario's
    repair-labels directory doesn't exist yet (run
    run_cluster_repair_labelling.py for it first).
    """
    scenario_dir = os.path.join(repair_labels_dir, scenario_id)
    if not os.path.isdir(scenario_dir):
        return []
    frame_ids = set()
    for name in os.listdir(scenario_dir):
        if name.endswith("_fragments.parquet"):
            # "<frame_id>_<cav_id>_fragments.parquet" -- frame_id is
            # everything before the first underscore (this dataset's
            # frame_ids are fixed-width numeric strings with no underscore
            # of their own, e.g. "003464").
            frame_ids.add(name.split("_", 1)[0])
    return sorted(frame_ids)


def run_frozen_inference_and_cache_with_crops(
    scenario_ids: list[str],
    frame_ids_by_scenario: dict[str, list[str]],
    opv2v_root: str,
    model_dir: str,
    config_path: str,
    out_dir: str,
    arm: str,
    repair_labels_dir: str,
    degradation_assignment: dict | None = None,
) -> dict:
    """
    Same detections + GT caching as run_frozen_inference_and_cache (byte-
    identical predicted/GT boxes for the same inputs -- this reimplements
    inference_utils.inference_late_fusion's own per-CAV loop rather than
    calling it, specifically so the backbone's intermediate
    spatial_features_2d can be captured per CAV via a forward hook, which
    that function doesn't expose), PLUS one BEV feature crop per candidate
    fragment (from run_cluster_repair_labelling.py's real fragments table)
    per (scenario, frame, cav), written to
    `<out_dir>/<scenario_id>_<arm>_crops.npz` as parallel arrays keyed by
    (frame_id, cav_id, fragment_id) so they join directly onto the
    fragments/pairs parquet tables already on disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(config_path, None)
    dataset = ManifestFilteredLateFusionDataset(
        hypes, opv2v_root, degradation_assignment=degradation_assignment, train=False
    )

    print("Creating model and loading checkpoint...")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()

    lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]
    voxel_size = hypes["preprocess"]["args"]["voxel_size"]
    grid_shape = (hypes["postprocess"]["anchor_args"]["H"], hypes["postprocess"]["anchor_args"]["W"])

    _captured = {}

    def _capture_hook(module, inputs, output):
        _captured["spatial_features_2d"] = output["spatial_features_2d"].detach().cpu().numpy()

    handle = model.backbone.register_forward_hook(_capture_hook)

    summary = {"arm": arm, "scenarios": {}}

    try:
        for scenario_id in scenario_ids:
            frame_ids = frame_ids_by_scenario.get(scenario_id, [])
            if not frame_ids:
                continue

            # BUG FIX (1 Oct 2026, found by comparing dataset.py's real
            # verify_dataset_on_device.py sample count (15/36 real combos)
            # against the actual crops.npz contents): OpenCOOD's own
            # LateFusionDataset.get_item_test() (late_fusion_dataset.py line
            # ~158) renames the ego CAV's dict key to the literal string
            # "ego" -- `update_cav = "ego" if cav_id == ego_id else cav_id`
            # -- not its real numeric cav_id. That string then flowed
            # straight into _load_fragment_centroids(..., str(cav_id)) below,
            # which looked for a file literally named
            # "<frame_id>_ego_fragments.parquet" -- never exists, since
            # run_cluster_repair_labelling.py always wrote the real numeric
            # id -- so ego's crops were silently never produced, every
            # frame, every scenario. Resolved here by reading the real ego
            # cav_id straight from the dataset's own per-scenario raw
            # cav table (the same 'ego' boolean flag basedataset.py sets,
            # not a reapplication of the separate "lowest id" convention),
            # so "ego" can be mapped back before the fragment lookup.
            scenario_index = dataset.scenario_ids.index(scenario_id)
            ego_cav_id = next(
                cid
                for cid, c in dataset.scenario_database[scenario_index].items()
                if c["ego"]
            )

            t0 = time.time()
            per_frame_records = []
            crop_frame_ids, crop_cav_ids, crop_fragment_ids, crop_arrays, crop_valid = [], [], [], [], []

            for frame_id in frame_ids:
                idx = dataset.idx_for(scenario_id, frame_id)
                sample = dataset[idx]
                batch_data = dataset.collate_batch_test([sample])
                batch_data = train_utils.to_device(batch_data, device)

                from collections import OrderedDict

                output_dict = OrderedDict()
                for cav_id, cav_content in batch_data.items():
                    real_cav_id = ego_cav_id if cav_id == "ego" else cav_id
                    _captured.clear()
                    with torch.no_grad():
                        output_dict[cav_id] = model(cav_content)
                    # _captured now holds exactly this cav_id's spatial_features_2d
                    # (batch size 1 per collate_batch_test's own requirement, so
                    # index 0 is always this CAV, never a different one).
                    spatial_features_2d = _captured.get("spatial_features_2d")
                    if spatial_features_2d is None:
                        continue
                    feat_map = spatial_features_2d[0]  # (C, feat_h, feat_w)
                    feat_shape = feat_map.shape[1:]

                    for fragment_id, cx, cy in _load_fragment_centroids(
                        repair_labels_dir, scenario_id, frame_id, str(real_cav_id)
                    ):
                        row, col = _pixel_from_xy(float(cx), float(cy), lidar_range, voxel_size, grid_shape, feat_shape)
                        crop, valid = crop_bev_feature(feat_map, row, col)
                        crop_frame_ids.append(frame_id)
                        crop_cav_ids.append(str(real_cav_id))
                        crop_fragment_ids.append(int(fragment_id))
                        crop_arrays.append(crop)
                        crop_valid.append(valid)

                with torch.no_grad():
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

                per_frame_records.append(
                    {
                        "frame_id": frame_id,
                        "pred_boxes": torch_tensor_to_numpy(pred_box_tensor)
                        if pred_box_tensor is not None
                        else np.zeros((0, 8, 3)),
                        "pred_scores": torch_tensor_to_numpy(pred_score)
                        if pred_score is not None
                        else np.zeros((0,)),
                        "gt_boxes": torch_tensor_to_numpy(gt_box_tensor)
                        if gt_box_tensor is not None
                        else np.zeros((0, 8, 3)),
                    }
                )

            elapsed = time.time() - t0

            out_path = os.path.join(out_dir, f"{scenario_id}_{arm}.npz")
            np.savez_compressed(
                out_path,
                frame_ids=np.array([r["frame_id"] for r in per_frame_records]),
                pred_boxes=np.array([r["pred_boxes"] for r in per_frame_records], dtype=object),
                pred_scores=np.array([r["pred_scores"] for r in per_frame_records], dtype=object),
                gt_boxes=np.array([r["gt_boxes"] for r in per_frame_records], dtype=object),
            )
            disk_bytes = os.path.getsize(out_path)

            crops_path = os.path.join(out_dir, f"{scenario_id}_{arm}_crops.npz")
            np.savez_compressed(
                crops_path,
                frame_id=np.array(crop_frame_ids),
                cav_id=np.array(crop_cav_ids),
                fragment_id=np.array(crop_fragment_ids, dtype=int),
                crop=np.array(crop_arrays) if crop_arrays else np.zeros((0, 1, BEV_CROP_SIZE_PX, BEV_CROP_SIZE_PX)),
                valid=np.array(crop_valid, dtype=bool),
            )
            crops_disk_bytes = os.path.getsize(crops_path)

            summary["scenarios"][scenario_id] = {
                "n_frames": len(per_frame_records),
                "wall_clock_s": elapsed,
                "s_per_frame": elapsed / len(per_frame_records) if per_frame_records else float("nan"),
                "disk_bytes": disk_bytes,
                "cache_path": out_path,
                "n_crops": len(crop_arrays),
                "crops_disk_bytes": crops_disk_bytes,
                "crops_cache_path": crops_path,
            }
            print(
                f"  {scenario_id} [{arm}]: {len(per_frame_records)} frames, "
                f"{elapsed:.1f}s ({elapsed / max(len(per_frame_records),1):.2f}s/frame), "
                f"{disk_bytes / 1e6:.1f} MB detections, "
                f"{len(crop_arrays)} BEV crops ({crops_disk_bytes / 1e6:.1f} MB)"
            )
    finally:
        handle.remove()

    total_frames = sum(s["n_frames"] for s in summary["scenarios"].values())
    total_time = sum(s["wall_clock_s"] for s in summary["scenarios"].values())
    total_disk = sum(s["disk_bytes"] for s in summary["scenarios"].values())
    total_crops = sum(s["n_crops"] for s in summary["scenarios"].values())
    summary["totals"] = {
        "n_scenarios": len(summary["scenarios"]),
        "n_frames": total_frames,
        "wall_clock_s": total_time,
        "disk_bytes": total_disk,
        "n_crops": total_crops,
        "extrapolated_gb_per_342_frames": (total_disk / max(total_frames, 1)) * 342 / 1e9,
    }
    print(
        f"\n[{arm}] TOTAL: {summary['totals']['n_scenarios']} scenarios, "
        f"{total_frames} frames, {total_crops} BEV crops, {total_time:.1f}s, {total_disk / 1e6:.1f} MB detections."
    )

    summary_path = os.path.join(out_dir, f"cache_manifest_{arm}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to {summary_path}")

    return summary


def run_frozen_inference_and_cache(
    scenario_ids: list[str],
    frame_ids_by_scenario: dict[str, list[str]],
    opv2v_root: str,
    model_dir: str,
    config_path: str,
    out_dir: str,
    arm: str,
    degradation_assignment: dict | None = None,
) -> dict:
    """
    Runs the frozen (never fine-tuned) detector on the given scenario/frame
    selection and caches predicted + GT boxes to `out_dir`, one .npz per
    scenario. Returns a summary dict (also written to
    `out_dir/cache_manifest_{arm}.json`) with per-scenario timing and disk
    cost -- this is Fri 11 Sep's actual done-when: know the cost per
    scenario before committing to the full run.
    """
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(config_path, None)
    dataset = ManifestFilteredLateFusionDataset(
        hypes, opv2v_root, degradation_assignment=degradation_assignment, train=False
    )

    print("Creating model and loading checkpoint...")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()

    summary = {"arm": arm, "scenarios": {}}

    for scenario_id in scenario_ids:
        frame_ids = frame_ids_by_scenario.get(scenario_id, [])
        if not frame_ids:
            continue

        t0 = time.time()
        per_frame_records = []

        for frame_id in frame_ids:
            idx = dataset.idx_for(scenario_id, frame_id)
            sample = dataset[idx]
            batch_data = dataset.collate_batch_test([sample])
            batch_data = train_utils.to_device(batch_data, device)

            with torch.no_grad():
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(
                    batch_data, model, dataset
                )

            per_frame_records.append(
                {
                    "frame_id": frame_id,
                    "pred_boxes": torch_tensor_to_numpy(pred_box_tensor)
                    if pred_box_tensor is not None
                    else np.zeros((0, 8, 3)),
                    "pred_scores": torch_tensor_to_numpy(pred_score)
                    if pred_score is not None
                    else np.zeros((0,)),
                    "gt_boxes": torch_tensor_to_numpy(gt_box_tensor)
                    if gt_box_tensor is not None
                    else np.zeros((0, 8, 3)),
                }
            )

        elapsed = time.time() - t0

        out_path = os.path.join(out_dir, f"{scenario_id}_{arm}.npz")
        np.savez_compressed(
            out_path,
            frame_ids=np.array([r["frame_id"] for r in per_frame_records]),
            pred_boxes=np.array([r["pred_boxes"] for r in per_frame_records], dtype=object),
            pred_scores=np.array([r["pred_scores"] for r in per_frame_records], dtype=object),
            gt_boxes=np.array([r["gt_boxes"] for r in per_frame_records], dtype=object),
        )
        disk_bytes = os.path.getsize(out_path)

        summary["scenarios"][scenario_id] = {
            "n_frames": len(per_frame_records),
            "wall_clock_s": elapsed,
            "s_per_frame": elapsed / len(per_frame_records) if per_frame_records else float("nan"),
            "disk_bytes": disk_bytes,
            "cache_path": out_path,
        }
        print(
            f"  {scenario_id} [{arm}]: {len(per_frame_records)} frames, "
            f"{elapsed:.1f}s ({elapsed / max(len(per_frame_records),1):.2f}s/frame), "
            f"{disk_bytes / 1e6:.1f} MB"
        )

    total_frames = sum(s["n_frames"] for s in summary["scenarios"].values())
    total_time = sum(s["wall_clock_s"] for s in summary["scenarios"].values())
    total_disk = sum(s["disk_bytes"] for s in summary["scenarios"].values())
    summary["totals"] = {
        "n_scenarios": len(summary["scenarios"]),
        "n_frames": total_frames,
        "wall_clock_s": total_time,
        "disk_bytes": total_disk,
        "extrapolated_gb_per_342_frames": (total_disk / max(total_frames, 1)) * 342 / 1e9,
    }
    print(
        f"\n[{arm}] TOTAL: {summary['totals']['n_scenarios']} scenarios, "
        f"{total_frames} frames, {total_time:.1f}s, {total_disk / 1e6:.1f} MB. "
        f"Extrapolated to all 342 qualifying frames: "
        f"~{summary['totals']['extrapolated_gb_per_342_frames']:.2f} GB."
    )

    summary_path = os.path.join(out_dir, f"cache_manifest_{arm}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--model-dir", required=True, help="Directory containing the epoch_30 checkpoint")
    parser.add_argument("--config", default=str(REPO_ROOT / "pointpillar_late_fusion" / "config.yaml"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "cache"))
    parser.add_argument("--max-scenarios", type=int, default=20)
    parser.add_argument("--max-frames-per-scenario", type=int, default=10)
    parser.add_argument("--arm", choices=["A", "B"], default="A")
    parser.add_argument(
        "--degradation-config",
        default=None,
        help="JSON file mapping 'scenario_id|cav_id' -> {type, severity, seed}. "
        "Required for --arm B, ignored for --arm A.",
    )
    parser.add_argument(
        "--cache-bev-crops",
        action="store_true",
        help="Also cache a BEV feature crop per candidate fragment (Thu 1 Oct 2026 "
        "addition, for the correction head). Requires --repair-labels-dir to "
        "point at run_cluster_repair_labelling.py's real output.",
    )
    parser.add_argument(
        "--repair-labels-dir",
        default=str(REPO_ROOT / "P4_defence" / "cache" / "repair_labels"),
        help="Only used with --cache-bev-crops.",
    )
    parser.add_argument(
        "--crop-frames-from-manifest",
        action="store_true",
        help="Debug escape hatch: with --cache-bev-crops, use this script's own "
        "--max-frames-per-scenario .head() frame selection instead of "
        "discovering frames from --repair-labels-dir's actual output. Normally "
        "left off -- this is what caused GAP #2 (only 1 of 18 (frame, cav) "
        "combos per scenario got a cached crop, 1 Oct 2026) in the first place; "
        "see discover_repair_label_frames()'s docstring.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run P0's build_manifest() first.")
    df = pd.read_parquet(MANIFEST_PATH)
    scenario_ids = sorted(df["scenario_id"].unique())[: args.max_scenarios]
    frame_ids_by_scenario = {
        sid: df[df["scenario_id"] == sid]["frame_id"].head(args.max_frames_per_scenario).tolist()
        for sid in scenario_ids
    }

    degradation_assignment = None
    if args.arm == "B":
        if not args.degradation_config:
            raise ValueError("--degradation-config is required for --arm B")
        with open(args.degradation_config) as f:
            raw = json.load(f)
        degradation_assignment = {tuple(k.split("|")): v for k, v in raw.items()}

    if args.cache_bev_crops:
        if args.crop_frames_from_manifest:
            crop_frame_ids_by_scenario = frame_ids_by_scenario
        else:
            # Default, correct behaviour (fixes GAP #2, decisions.md Thu 1 Oct
            # 2026 continued): crop coverage must match repair-labelling's
            # actual frame coverage, not this script's own
            # --max-frames-per-scenario selection -- the two were
            # independently configured and only coincided on 1 of 18
            # (frame, cav) combos per scenario in the first --cache-bev-crops
            # run. Discovering frames straight from --repair-labels-dir's own
            # output makes this correct by construction instead of by two
            # separately-tuned constants staying in sync.
            crop_frame_ids_by_scenario = {
                sid: discover_repair_label_frames(args.repair_labels_dir, sid)
                for sid in scenario_ids
            }
            missing = [sid for sid, frames in crop_frame_ids_by_scenario.items() if not frames]
            if missing:
                print(
                    f"WARNING: no repair-labels frames found for {missing} under "
                    f"{args.repair_labels_dir} -- run run_cluster_repair_labelling.py "
                    "for these scenarios first, or crops for them will be empty."
                )
        run_frozen_inference_and_cache_with_crops(
            scenario_ids=scenario_ids,
            frame_ids_by_scenario=crop_frame_ids_by_scenario,
            opv2v_root=args.opv2v_root,
            model_dir=args.model_dir,
            config_path=args.config,
            out_dir=args.out_dir,
            arm=args.arm,
            repair_labels_dir=args.repair_labels_dir,
            degradation_assignment=degradation_assignment,
        )
    else:
        run_frozen_inference_and_cache(
            scenario_ids=scenario_ids,
            frame_ids_by_scenario=frame_ids_by_scenario,
            opv2v_root=args.opv2v_root,
            model_dir=args.model_dir,
            config_path=args.config,
            out_dir=args.out_dir,
            arm=args.arm,
            degradation_assignment=degradation_assignment,
        )


if __name__ == "__main__":
    main()
