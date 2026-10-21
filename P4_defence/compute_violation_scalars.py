"""GAP #1 close-out (decisions.md / correction_head/dataset.py's
load_violation_lookup() docstring) -- computes the real
free_occ_conflict_area / tracking_affinity values that correction_head's
model has been training with zero-filled until now.

WHAT THIS ACTUALLY COMPUTES
-----------------------------
For every (scenario_id, frame_id) that actually appears in
P4_defence/cache/repair_labels/ (i.e. the frames correction_head's dataset
loader will ever ask for -- no point computing this for frames that were
never repair-labelled), runs run_defence_check.evaluate_frame() once and
keeps its two violation magnitudes:
  - occupancy_violation_m2  <- detail["max_violation_m2"]   (occupancy-
    consistency half: CAV-vs-CAV free/occupied cross-check)
  - perception_violation_m2 <- detail["perception_violation_m2"] (detector-
    vs-free-space half; 0.0 if no P3 detection cache is available for that
    scenario/arm, same convention evaluate_frame() already uses)

These are FRAME-level (not per-fragment, not per-CAV in the ego frame
sense -- see evaluate_frame()'s own per-CAV loop, which already aligns
everything to the ego frame before computing the violation), which is
exactly why dataset.py's model.py comment calls them "frame-level,
broadcast to its fragments": every fragment belonging to the same
(scenario_id, frame_id) gets the same two numbers, same as
range_density_bucket's per-(frame,cav) broadcast.

WHY THIS COULD HELP THE DIFFERENT-VEHICLE FPR SPECIFICALLY
--------------------------------------------------------------
Not guaranteed, but the mechanism is real: occupancy_violation_m2 is
highest exactly when CAVs disagree about what's actually out there --
one CAV's sensor swept a patch of ground clean while another CAV claims
something occupies it. That is the same kind of multi-CAV disagreement
signature that shows up around genuinely ambiguous, cluttered regions
where two separate vehicles sit close together -- precisely the regime
diagnose_false_merges.py found the false merges concentrated in. Feeding
the model a real measurement of "how much do the CAVs disagree about
this scene" gives it information it never had access to before (it was
training on a literal 0.0 for both features, every single fragment,
every single frame -- pure noise, not just a weak feature).

OUTPUT
------
Writes {out_path} (default P4_defence/cache/violation_scalars.csv), with
EXACTLY the columns correction_head/dataset.py's load_violation_lookup()
expects: scenario_id, frame_id, occupancy_violation_m2,
perception_violation_m2 -- one row per (scenario_id, frame_id).

USAGE (opencood conda env active -- this needs torch via pcd_to_np/
mask_ego_points, same as run_defence_check.py; NOT runnable in the cloud
sandbox or the device-bridge's own Linux VM):
    python P4_defence/compute_violation_scalars.py --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data"

Then point training at the result:
    python -m correction_head.train_v1 --violation-lookup P4_defence/cache/violation_scalars.csv [...other args as before...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from P4_defence.run_defence_check import (  # noqa: E402
    MANIFEST_PATH, evaluate_frame, load_detection_cache,
)
from correction_head.dataset import (  # noqa: E402
    DEFAULT_REPAIR_LABELS_DIR, discover_frame_cav_pairs,
)


def discover_scenario_frame_pairs(repair_labels_dir) -> list[tuple[str, str]]:
    """Every (scenario_id, frame_id) with at least one repair-labelled
    (frame, cav) unit -- i.e. every frame correction_head's dataset loader
    could ever ask load_violation_lookup() about. One frame can have
    several CAVs' worth of (frame, cav) files; we only need to evaluate
    that frame once (the violation check is frame-level, not per-CAV)."""
    pairs = set()
    for scen_dir in sorted(Path(repair_labels_dir).glob("*")):
        if not scen_dir.is_dir():
            continue
        scenario_id = scen_dir.name
        for frame_id, _cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
            pairs.add((scenario_id, frame_id))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--repair-labels-dir", default=str(DEFAULT_REPAIR_LABELS_DIR))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "P4_defence" / "cache" / "violation_scalars.csv")
    )
    parser.add_argument(
        "--detection-cache-dir",
        default=str(REPO_ROOT / "P3_perception" / "cache"),
        help="Same convention as run_defence_check.py -- pass --no-detector to skip the "
        "perception-occupancy half (occupancy-consistency only).",
    )
    parser.add_argument("--detection-arm", choices=["A", "B"], default="A")
    parser.add_argument("--no-detector", action="store_true")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run P0's build_manifest() first.")
    manifest = pd.read_parquet(MANIFEST_PATH)

    scenario_frame_pairs = discover_scenario_frame_pairs(args.repair_labels_dir)
    if not scenario_frame_pairs:
        print(f"No (scenario, frame) pairs found under {args.repair_labels_dir} -- "
              "nothing to compute. Run run_cluster_repair_labelling.py first.")
        return
    print(f"{len(scenario_frame_pairs)} (scenario, frame) pairs to evaluate "
          f"(from {args.repair_labels_dir}).")

    detection_cache_by_scenario = {}
    rows = []
    n_missing_manifest_row = 0
    for scenario_id, frame_id in scenario_frame_pairs:
        manifest_rows = manifest[
            (manifest["scenario_id"] == scenario_id) & (manifest["frame_id"] == frame_id)
        ]
        if manifest_rows.empty:
            # A repair-labelled frame that P0's manifest doesn't know about --
            # can happen if the manifest was rebuilt/filtered after labelling
            # ran. Skip it (load_violation_lookup()'s caller already handles
            # a missing lookup entry by zero-filling, same as today), but
            # count it so this isn't silently lossy.
            n_missing_manifest_row += 1
            continue
        cav_ids = list(manifest_rows.iloc[0]["cav_ids"])
        ego_id = min(cav_ids, key=int)  # P0's confirmed ego convention (17 Sep 2026)

        if not args.no_detector and scenario_id not in detection_cache_by_scenario:
            detection_cache_by_scenario[scenario_id] = load_detection_cache(
                args.detection_cache_dir, scenario_id, args.detection_arm
            )
        detections = None
        if not args.no_detector:
            detections = detection_cache_by_scenario[scenario_id].get(frame_id)

        try:
            _flagged, detail = evaluate_frame(
                args.opv2v_root, scenario_id, frame_id, cav_ids, ego_id,
                degradation_assignment=None,  # arm A convention, same as run_defence_check.py's default
                detections=detections,
            )
        except Exception as exc:  # noqa: BLE001 -- real-data pass, log and keep going per-frame
            print(f"  [{scenario_id}/{frame_id}] evaluate_frame() failed: {exc!r} -- skipping, "
                  "this frame will fall back to zero-filled violation features.")
            continue

        rows.append({
            "scenario_id": scenario_id,
            "frame_id": frame_id,
            "occupancy_violation_m2": detail["max_violation_m2"],
            "perception_violation_m2": detail["perception_violation_m2"],
        })

    if n_missing_manifest_row:
        print(f"{n_missing_manifest_row} repair-labelled (scenario, frame) pairs had no matching "
              "manifest row -- skipped, will fall back to zero-filled features for those frames.")

    if not rows:
        print("No frames evaluated successfully -- nothing written.")
        return

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df[["occupancy_violation_m2", "perception_violation_m2"]].describe())


if __name__ == "__main__":
    main()
