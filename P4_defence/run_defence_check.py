"""
P4 — Wire the full (simplified) CAD-style defence together and reproduce a
clean-condition FPR number. Per execution_plan_daily.md's original Mon 7 Sep
task: "Consistency layer + detections -> the complete CAD-style flag
decision. Run on 5 clean scenarios... Done when: you have a clean-condition
FPR number written down with the scenario count it came from."

NOTE on scope (see consistency_layer.py's module docstring for the full
comparison against CAD's actual released code, read 16 Sep 2026):
  - This uses free_vs_occupied_violation(), not conflicting_area() -- the
    free-space / occupied-space cross-check is closer to CAD's real logic
    than a raw blob-vs-blob diff.
  - The detector (P3) is NOT wired in yet -- P3 is still an unimplemented
    stub (P3_perception/perception_cache.py), scheduled for Week 2. So this
    FPR number reflects the occupancy-consistency check ALONE, not CAD's
    full two-part defence (occupancy-consistency + perception-occupancy).
    That's a real, deliberate scope cut for today -- document it as such.
  - No lane-area filtering (CAD's code filters occupied/free areas to
    drivable lanes using pre-computed CARLA town maps this project doesn't
    have). Also a deliberate simplification.

Because every scenario is run WITHOUT any injected attack or degradation
(P2 doesn't exist yet either), every frame the defence flags here is, by
definition, a FALSE positive -- that's what makes the flagged-frame rate
below a genuine clean-condition FPR number.

Usage (from the MastersProj repo root, opencood conda env active):
    python P4_defence/run_defence_check.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data" ^
        --max-scenarios 5 --max-frames-per-scenario 10

Added 17 Sep 2026: optional --degradation-config (same JSON format as
P3_perception/perception_cache.py's --degradation-config) applies P2's
fog/compression injection to the assigned non-ego CAV's point cloud before
the occupancy-consistency check runs -- this is what makes an arm-B
(degraded-benign) FPR run possible, for comparison against arm A (clean,
no --degradation-config) -- see P4_defence/compute_fp_gap.py, which runs
both arms back-to-back and reports the gap between them.

Added 17 Sep 2026 continued: the detector half is now wired in too, via
--detection-cache-dir (P3's cached .npz output, P3_perception/cache by
default) -- see consistency_layer.py's "Perception-occupancy check" section
for the actual logic and, importantly, a real scope note: P3's cache holds
ONE fused detection set per frame (OpenCOOD's late-fusion output), not a
per-CAV detection, so this checks the fused detector's claimed boxes
against each individual CAV's own free space, not CAV-vs-CAV detector
claims the way CAD's own per-CAV design works. A frame is now flagged if
EITHER the occupancy-consistency check OR the perception-occupancy check
fires (OR-combination) -- CAD's own combination rule wasn't confirmed
against their released code this session, so this is a reasonable default,
not a verified match; flagged as such rather than assumed correct.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# See P0_scenario_indexing/index.py's module comment: OPV2V's official
# train/validate/test splits dump numpy scalars into frame yaml, which
# yaml.safe_load() refuses -- reusing OpenCOOD's own loader (same fix).
from opencood.hypes_yaml.yaml_utils import load_yaml as _opencood_load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenCOOD"))
sys.path.insert(0, str(REPO_ROOT / "P1_alignment"))
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points  # noqa: E402

from consistency_layer import (  # noqa: E402
    remove_ground_plane,
    cluster_points,
    clusters_to_polygons,
    get_free_space,
    free_vs_occupied_violation,
    detections_to_polygons,
    perception_occupancy_violation,
    CONFLICT_AREA_THRESHOLD_M2,
)
from align import align_to_ego  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "P2_degradation"))
from degrade import inject_fog, inject_compression  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"
OUT_DIR = Path(__file__).resolve().parent / "cache"


import functools as _functools


@_functools.lru_cache(maxsize=None)
def resolve_scenario_dir(opv2v_root: str, scenario_id: str) -> Path:
    """
    OPV2V extracts sometimes put a scenario directly under opv2v_root (e.g.
    culver_data), sometimes one level deeper under a split subfolder
    (train/validate/test) -- mirrors P0's own _discover_scenarios()
    two-level search (P0_scenario_indexing/index.py) so a manifest
    scenario_id resolves correctly regardless of which OPV2V extract it
    came from. Found by running this script against the combined
    train/validate/test+culver_data root (Thu 2 Oct 2026) and getting 0
    audited instances -- this function previously assumed every scenario
    sat directly under opv2v_root, which broke once scenarios came from
    more than one subfolder.
    """
    root = Path(opv2v_root)
    direct = root / scenario_id
    if direct.is_dir():
        return direct
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidate = child / scenario_id
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(
        f"scenario {scenario_id!r} not found under {opv2v_root} "
        "(checked directly under opv2v_root and one level deeper)"
    )


def frame_yaml_path(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> Path:
    return resolve_scenario_dir(opv2v_root, scenario_id) / cav_id / f"{frame_id}.yaml"


def frame_pcd_path(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> Path:
    return resolve_scenario_dir(opv2v_root, scenario_id) / cav_id / f"{frame_id}.pcd"


def load_lidar_pose(opv2v_root: str, scenario_id: str, cav_id: str, frame_id: str) -> np.ndarray:
    yaml_path = frame_yaml_path(opv2v_root, scenario_id, cav_id, frame_id)
    data = _opencood_load_yaml(str(yaml_path))
    return np.asarray(data["lidar_pose"], dtype=float)


def align_polygon_to_ego(polygon, cav_pose: np.ndarray, ego_pose: np.ndarray):
    """A polygon's vertices are just points -- reuse the same validated
    align_to_ego() transform used for point clouds rather than re-deriving
    a separate polygon-transform path."""
    if polygon.is_empty:
        return polygon
    coords = np.array(polygon.exterior.coords)
    pts = np.zeros((len(coords), 4))
    pts[:, :2] = coords
    aligned = align_to_ego(pts, cav_pose=cav_pose, ego_pose=ego_pose)
    from shapely.geometry import Polygon as ShapelyPolygon
    return ShapelyPolygon(aligned[:, :2])


def apply_degradation_if_assigned(points: np.ndarray, scenario_id: str, cav_id: str,
                                   degradation_assignment: dict | None) -> np.ndarray:
    """
    Applies P2's inject_fog/inject_compression to `points` if this
    (scenario_id, cav_id) has an assignment in `degradation_assignment` --
    same lookup shape as P3's ManifestFilteredLateFusionDataset
    (`{(scenario_id, cav_id): {"type", "severity", "seed"}}`), and the same
    fix applied there on 17 Sep 2026: compression quantises the intensity
    column ONLY, never x/y/z geometry -- quantising geometry with a shared
    vmin/vmax across all four columns was found (via a real crash in P3) to
    collapse a CAV's point cloud enough to get filtered out entirely by
    downstream range/ego masking. That masking doesn't apply here (this
    file never calls OpenCOOD's mask_points_by_range), but the fix is kept
    identical for consistency between what P3's detector sees and what
    this occupancy-consistency check sees for the same degraded CAV.
    """
    if not degradation_assignment:
        return points
    assignment = degradation_assignment.get((scenario_id, cav_id))
    if assignment is None:
        return points
    deg_type = assignment["type"]
    severity = assignment["severity"]
    seed = assignment.get("seed", 0)
    if deg_type == "fog":
        return inject_fog(points, severity, seed)
    elif deg_type == "compression":
        degraded = points.copy()
        if degraded.shape[1] >= 4:
            degraded[:, 3] = inject_compression(degraded[:, 3], severity, seed)
        return degraded
    else:
        raise ValueError(f"unknown degradation type {deg_type!r}")


def load_detection_cache(cache_dir: str, scenario_id: str, arm: str) -> dict:
    """
    Loads P3's cached .npz for one scenario/arm (P3_perception/perception_cache.py's
    output: frame_ids, pred_boxes, pred_scores, gt_boxes, each an object
    array in the same frame order). Returns {frame_id: (pred_boxes,
    pred_scores)}. Returns {} (not an error) if the cache file doesn't exist
    -- callers fall back to occupancy-consistency-only for that scenario's
    frames rather than crashing, since not every scenario/arm combination is
    guaranteed to have been cached yet.
    """
    path = Path(cache_dir) / f"{scenario_id}_{arm}.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {str(fid): (data["pred_boxes"][i], data["pred_scores"][i])
            for i, fid in enumerate(data["frame_ids"])}


def evaluate_frame(opv2v_root: str, scenario_id: str, frame_id: str, cav_ids: list[str], ego_id: str,
                    degradation_assignment: dict | None = None,
                    detections: tuple | None = None):
    """
    Returns (flagged: bool, detail: dict) for one frame -- flagged is True if
    EITHER free_vs_occupied_violation() fires for any pair of CAVs
    (occupancy-consistency half) OR perception_occupancy_violation() fires
    against the frame's cached detections (perception-occupancy half, added
    17 Sep 2026 -- see this file's + consistency_layer.py's module
    docstrings for the scope note on what this half actually checks).

    degradation_assignment: optional {(scenario_id, cav_id): {"type",
        "severity", "seed"}} -- same shape P3's --degradation-config JSON
        produces once parsed. Ego's own point cloud is never degraded (same
        convention as P3), matching P2's "one benign CAV at a time" design.

    detections: optional (pred_boxes, pred_scores) for this exact frame, as
        load_detection_cache() returns per frame_id -- pass None to run the
        occupancy-consistency check alone (e.g. no P3 cache available yet).
    """
    ego_pose = load_lidar_pose(opv2v_root, scenario_id, ego_id, frame_id)

    per_cav = {}
    for cav_id in cav_ids:
        pcd_file = frame_pcd_path(opv2v_root, scenario_id, cav_id, frame_id)
        points = pcd_to_np(str(pcd_file))
        points = mask_ego_points(points)
        if cav_id != ego_id:
            points = apply_degradation_if_assigned(points, scenario_id, cav_id, degradation_assignment)

        non_ground = remove_ground_plane(points)
        labels = cluster_points(non_ground)
        occupied_local = clusters_to_polygons(non_ground, labels)
        free_local = get_free_space(points)  # full points, ground included -- see module docstring

        cav_pose = load_lidar_pose(opv2v_root, scenario_id, cav_id, frame_id)
        occupied_ego = [align_polygon_to_ego(p, cav_pose, ego_pose) for p in occupied_local]
        free_ego = align_polygon_to_ego(free_local, cav_pose, ego_pose)

        per_cav[cav_id] = (occupied_ego, free_ego)

    occupancy_flagged = False
    max_violation = 0.0
    flagged_pairs = []
    for cav_a, cav_b in itertools.combinations(cav_ids, 2):
        occ_a, free_a = per_cav[cav_a]
        occ_b, free_b = per_cav[cav_b]
        result = free_vs_occupied_violation(occ_a, free_a, occ_b, free_b)
        max_violation = max(max_violation, result.total_violation_m2)
        if result.flagged:
            occupancy_flagged = True
            flagged_pairs.append((cav_a, cav_b, result.total_violation_m2))

    perception_violation_m2 = 0.0
    perception_flagged = False
    if detections is not None:
        pred_boxes, pred_scores = detections
        detection_polygons = detections_to_polygons(pred_boxes, pred_scores)
        free_space_by_cav = {cav_id: per_cav[cav_id][1] for cav_id in cav_ids}
        perception_result = perception_occupancy_violation(detection_polygons, free_space_by_cav)
        perception_violation_m2 = perception_result.detected_in_free_m2
        perception_flagged = perception_result.flagged

    frame_flagged = occupancy_flagged or perception_flagged

    return frame_flagged, {
        "max_violation_m2": max_violation,
        "flagged_pairs": flagged_pairs,
        "occupancy_flagged": occupancy_flagged,
        "perception_violation_m2": perception_violation_m2,
        "perception_flagged": perception_flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--max-frames-per-scenario", type=int, default=10)
    parser.add_argument(
        "--degradation-config",
        default=None,
        help="Optional JSON file mapping 'scenario_id|cav_id' -> {type, severity, seed} -- "
        "same format as P3_perception/perception_cache.py's --degradation-config. Omit for "
        "a clean-condition (arm A) run.",
    )
    parser.add_argument(
        "--detection-cache-dir",
        default=str(REPO_ROOT / "P3_perception" / "cache"),
        help="Directory holding P3's cached {scenario_id}_{arm}.npz files. Pass "
        "--no-detector to skip the perception-occupancy half entirely (occupancy-"
        "consistency only, the pre-17-Sep behaviour).",
    )
    parser.add_argument(
        "--detection-arm",
        choices=["A", "B"],
        default=None,
        help="Which arm's P3 cache to load (must match what P3_perception/perception_cache.py "
        "was run with). Defaults to 'B' if --degradation-config is given, else 'A' -- "
        "override only if you're intentionally mixing arms.",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Skip the perception-occupancy half entirely and run occupancy-consistency "
        "alone, regardless of what's cached (useful for an apples-to-apples comparison "
        "against pre-17-Sep results).",
    )
    args = parser.parse_args()

    degradation_assignment = None
    if args.degradation_config:
        import json
        with open(args.degradation_config) as f:
            raw = json.load(f)
        degradation_assignment = {tuple(k.split("|")): v for k, v in raw.items()}

    detection_arm = args.detection_arm or ("B" if args.degradation_config else "A")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run P0's build_manifest() first.")
    df = pd.read_parquet(MANIFEST_PATH)
    scenario_ids = sorted(df["scenario_id"].unique())
    print(f"{len(scenario_ids)} qualifying scenarios in manifest; evaluating up to {args.max_scenarios}.")

    total_frames = 0
    flagged_frames = 0
    per_scenario_results = []
    all_violations = []  # max_violation_m2 for every evaluated frame, flagged or not --
    # this is the diagnostic added 16 Sep 2026 after an exact 0.0% FPR result raised
    # the question "is this a genuine result or a bug hiding behind a boolean flag?"

    all_perception_violations = []
    n_perception_only_flags = 0  # frames flagged by perception-occupancy but NOT occupancy-consistency --
    # this is the number that shows whether wiring the detector in actually catches anything new

    for scenario_id in scenario_ids[: args.max_scenarios]:
        detection_cache = {}
        if not args.no_detector:
            detection_cache = load_detection_cache(args.detection_cache_dir, scenario_id, detection_arm)
            if not detection_cache:
                print(f"  [{scenario_id}] no P3 cache found for arm {detection_arm} in "
                      f"{args.detection_cache_dir} -- running occupancy-consistency only for this scenario.")

        rows = df[df["scenario_id"] == scenario_id].head(args.max_frames_per_scenario)
        scenario_flagged = 0
        scenario_total = 0
        for _, row in rows.iterrows():
            frame_id = row["frame_id"]
            cav_ids = list(row["cav_ids"])
            ego_id = min(cav_ids, key=int)  # P0's confirmed ego convention (17 Sep)
            detections = detection_cache.get(frame_id)
            try:
                flagged, detail = evaluate_frame(
                    args.opv2v_root, scenario_id, frame_id, cav_ids, ego_id,
                    degradation_assignment=degradation_assignment,
                    detections=detections,
                )
            except FileNotFoundError as e:
                print(f"  [skip] {scenario_id}/{frame_id}: {e}")
                continue
            scenario_total += 1
            total_frames += 1
            all_violations.append(detail["max_violation_m2"])
            all_perception_violations.append(detail["perception_violation_m2"])
            if detail["perception_flagged"] and not detail["occupancy_flagged"]:
                n_perception_only_flags += 1
            if flagged:
                scenario_flagged += 1
                flagged_frames += 1
                pairs_str = ", ".join(f"{a}-{b}={v:.1f}m2" for a, b, v in detail["flagged_pairs"])
                which = []
                if detail["occupancy_flagged"]:
                    which.append(f"occupancy({pairs_str})")
                if detail["perception_flagged"]:
                    which.append(f"perception({detail['perception_violation_m2']:.1f}m2)")
                print(f"  [FLAG] {scenario_id}/{frame_id}: {' + '.join(which)}")
            else:
                # Always print the non-flagged violation magnitude too -- a boolean
                # alone can't distinguish "always exactly 0.00 (probably a bug)" from
                # "small but nonzero, consistently under threshold (a plausible result)".
                print(f"  [ok]   {scenario_id}/{frame_id}: occupancy={detail['max_violation_m2']:.4f}m2 "
                      f"perception={detail['perception_violation_m2']:.4f}m2")
        scenario_fpr = scenario_flagged / scenario_total if scenario_total else float("nan")
        per_scenario_results.append((scenario_id, scenario_flagged, scenario_total, scenario_fpr))
        print(f"Scenario {scenario_id}: {scenario_flagged}/{scenario_total} frames flagged (FPR={scenario_fpr:.1%})")

    print("\n=== Clean-condition FPR summary ===")
    for scenario_id, flagged, total, fpr in per_scenario_results:
        print(f"  {scenario_id}: {flagged}/{total} ({fpr:.1%})")
    overall_fpr = flagged_frames / total_frames if total_frames else float("nan")
    print(f"\nOVERALL: {flagged_frames}/{total_frames} frames flagged across "
          f"{len(per_scenario_results)} scenarios -- clean-condition FPR = {overall_fpr:.1%}")

    if all_violations:
        arr = np.array(all_violations)
        n_exact_zero = int(np.sum(arr < 1e-9))
        print(
            f"\n=== Occupancy-consistency violation-magnitude diagnostic (max_violation_m2 "
            f"per frame, threshold={CONFLICT_AREA_THRESHOLD_M2} m^2) ===\n"
            f"  min={arr.min():.4f}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
            f"max={arr.max():.4f}\n"
            f"  frames with essentially exact 0.0 m^2 (<1e-9): {n_exact_zero}/{len(arr)}\n"
            f"  Interpretation: if n_exact_zero is close to len(arr), the occupied/free "
            f"polygons are likely never actually overlapping at all (check alignment / "
            f"CAV separation before trusting the FPR) -- if most values are small but "
            f"nonzero and clustered well under the threshold, that's a plausible real "
            f"result given how far apart these CAVs' sensors are (see P1 alignment "
            f"check, 16 Sep 2026: CAVs ~50-80m apart in this scenario)."
        )

    if all_perception_violations:
        parr = np.array(all_perception_violations)
        print(
            f"\n=== Perception-occupancy diagnostic (detected_in_free_m2 per frame, "
            f"threshold={CONFLICT_AREA_THRESHOLD_M2} m^2, arm={detection_arm}) ===\n"
            f"  min={parr.min():.4f}  mean={parr.mean():.4f}  median={np.median(parr):.4f}  "
            f"max={parr.max():.4f}\n"
            f"  frames flagged by perception-occupancy but NOT by occupancy-consistency: "
            f"{n_perception_only_flags}/{len(parr)}\n"
            f"  Interpretation: a nonzero n_perception_only_flags count is the actual "
            f"answer to 'does wiring the detector in catch anything the free/occupied "
            f"geometry check alone misses' -- if it's 0, the detector isn't adding "
            f"anything new in this sample (still worth reporting, not just omitting)."
        )

    print(
        f"\nCaveats (see this file's + consistency_layer.py's module docstrings): "
        f"no lane-area filtering, threshold={CONFLICT_AREA_THRESHOLD_M2} m^2 borrowed "
        f"from a secondary paper's analysis of CAD, not CAD's own source. The "
        f"perception-occupancy half (added 17 Sep 2026) checks the FUSED detector's "
        f"claimed boxes against each CAV's own free space, not CAV-vs-CAV detector "
        f"claims the way CAD's own per-CAV design works -- P3's cache only ever holds "
        f"one fused detection set per frame. The occupancy-consistency + perception-"
        f"occupancy combination rule (OR) is a reasonable default, not verified "
        f"against CAD's own source. This is a deliberately simplified reproduction -- "
        f"not a claim of matching CAD's published number exactly."
    )


if __name__ == "__main__":
    main()
