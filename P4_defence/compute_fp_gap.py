"""
P4 -- The first FP-gap number. Per execution_plan_daily.md's original Mon 14
Sep task: "Compute FPR on arm B (degraded benign) vs arm A (clean). This
number is the entire premise of the thesis. Write it down prominently."

Implemented 17 Sep 2026, once P3's real run confirmed the compression-vs-
geometry bug and its fix (see decisions.md, Thu 17 Sep entry) -- that fix is
mirrored here too (`run_defence_check.py`'s `apply_degradation_if_assigned`),
so arm B's degraded point clouds are produced the same way in both the
detector path (P3) and this occupancy-consistency-only path (P4).

**Scope note, same as run_defence_check.py's own:** this now runs the FULL
(simplified) two-part CAD-style flag decision -- occupancy-consistency OR
perception-occupancy -- not just the occupancy-consistency half. The
perception-occupancy half was wired in on 17 Sep 2026 once it became clear
(via a real, unexpected result: fog degradation LOWERED violation magnitude
instead of raising it) that reporting an FP-gap from half the defence
wasn't a meaningful comparison. See run_defence_check.py's + consistency_layer.py's
module docstrings for the real scope note that still applies: P3's cache
holds one FUSED detection set per frame, not a per-CAV detection, so the
perception-occupancy half checks the fused detector's claims against each
CAV's own free space, not CAV-vs-CAV detector claims the way CAD's own
per-CAV design works, and the OR-combination rule is a reasonable default,
not verified against CAD's own source.

Usage (from the MastersProj repo root, opencood conda env active) -- run P3
for both arms FIRST if you haven't already (perception_cache.py --arm A and
--arm B), since this reads their cached output:
    python P4_defence/compute_fp_gap.py ^
        --opv2v-root "C:\\Arnav\\University Work\\Masters\\Sem 2\\MastersProj\\data\\culver_data" ^
        --degradation-config P3_perception\\degradation_config_arm_b.json ^
        --detection-cache-dir P3_perception\\cache ^
        --max-scenarios 2 --max-frames-per-scenario 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_defence_check import evaluate_frame, load_detection_cache, MANIFEST_PATH


def run_arm(opv2v_root: str, degradation_assignment: dict | None, scenario_ids: list[str],
            max_frames_per_scenario: int, df: pd.DataFrame, arm_label: str,
            detection_cache_dir: str | None, detection_arm: str) -> dict:
    """Runs evaluate_frame() over the given scenarios/frames for one arm,
    returns a summary dict with per-scenario and overall FPR plus the raw
    per-frame occupancy- and perception-violation magnitudes (for the same
    diagnostic transparency run_defence_check.py already gives)."""
    total_frames = 0
    flagged_frames = 0
    per_scenario = {}
    all_violations = []
    all_perception_violations = []
    n_perception_only_flags = 0

    for scenario_id in scenario_ids:
        detection_cache = {}
        if detection_cache_dir:
            detection_cache = load_detection_cache(detection_cache_dir, scenario_id, detection_arm)
            if not detection_cache:
                print(f"  [{arm_label}] [{scenario_id}] no P3 cache found for arm "
                      f"{detection_arm} -- occupancy-consistency only for this scenario.")

        rows = df[df["scenario_id"] == scenario_id].head(max_frames_per_scenario)
        scenario_flagged = 0
        scenario_total = 0
        for _, row in rows.iterrows():
            frame_id = row["frame_id"]
            cav_ids = list(row["cav_ids"])
            ego_id = min(cav_ids, key=int)
            detections = detection_cache.get(frame_id)
            try:
                flagged, detail = evaluate_frame(
                    opv2v_root, scenario_id, frame_id, cav_ids, ego_id,
                    degradation_assignment=degradation_assignment,
                    detections=detections,
                )
            except FileNotFoundError as e:
                print(f"  [{arm_label}] [skip] {scenario_id}/{frame_id}: {e}")
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
            print(
                f"  [{arm_label}] {scenario_id}/{frame_id}: "
                f"{'FLAG' if flagged else 'ok  '} occupancy={detail['max_violation_m2']:.4f}m2 "
                f"perception={detail['perception_violation_m2']:.4f}m2"
            )
        scenario_fpr = scenario_flagged / scenario_total if scenario_total else float("nan")
        per_scenario[scenario_id] = {
            "flagged": scenario_flagged, "total": scenario_total, "fpr": scenario_fpr,
        }
        print(f"  [{arm_label}] scenario {scenario_id}: {scenario_flagged}/{scenario_total} "
              f"flagged (FPR={scenario_fpr:.1%})")

    overall_fpr = flagged_frames / total_frames if total_frames else float("nan")
    return {
        "arm": arm_label,
        "per_scenario": per_scenario,
        "total_frames": total_frames,
        "flagged_frames": flagged_frames,
        "overall_fpr": overall_fpr,
        "violations": all_violations,
        "perception_violations": all_perception_violations,
        "n_perception_only_flags": n_perception_only_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", required=True)
    parser.add_argument("--degradation-config", required=True,
                        help="JSON file defining arm B's degradation assignment "
                             "(same format as P3's --degradation-config).")
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--max-frames-per-scenario", type=int, default=10)
    parser.add_argument(
        "--detection-cache-dir",
        default=str(Path(__file__).resolve().parent.parent / "P3_perception" / "cache"),
        help="Directory holding P3's cached {scenario_id}_{arm}.npz files. Pass "
        "--no-detector to run occupancy-consistency only (pre-17-Sep behaviour).",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Skip the perception-occupancy half entirely, regardless of what's cached.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run P0's build_manifest() first.")
    df = pd.read_parquet(MANIFEST_PATH)
    scenario_ids = sorted(df["scenario_id"].unique())[: args.max_scenarios]

    with open(args.degradation_config) as f:
        raw = json.load(f)
    degradation_assignment = {tuple(k.split("|")): v for k, v in raw.items()}

    detection_cache_dir = None if args.no_detector else args.detection_cache_dir

    print(f"=== Arm A (clean) -- {len(scenario_ids)} scenarios ===")
    arm_a = run_arm(args.opv2v_root, None, scenario_ids, args.max_frames_per_scenario, df, "A",
                     detection_cache_dir=detection_cache_dir, detection_arm="A")

    print(f"\n=== Arm B (degraded benign) -- {len(scenario_ids)} scenarios ===")
    arm_b = run_arm(args.opv2v_root, degradation_assignment, scenario_ids,
                    args.max_frames_per_scenario, df, "B",
                    detection_cache_dir=detection_cache_dir, detection_arm="B")

    print("\n=== FP-GAP: arm B (degraded benign) vs arm A (clean) ===")
    print(f"  Arm A: {arm_a['flagged_frames']}/{arm_a['total_frames']} flagged, "
          f"FPR = {arm_a['overall_fpr']:.1%}")
    print(f"  Arm B: {arm_b['flagged_frames']}/{arm_b['total_frames']} flagged, "
          f"FPR = {arm_b['overall_fpr']:.1%}")
    gap = arm_b["overall_fpr"] - arm_a["overall_fpr"]
    print(f"  GAP (B - A): {gap:+.1%}")

    if arm_a["violations"] and arm_b["violations"]:
        va, vb = np.array(arm_a["violations"]), np.array(arm_b["violations"])
        print(f"\n  Occupancy violation magnitude, arm A: mean={va.mean():.4f}  "
              f"median={np.median(va):.4f}  max={va.max():.4f}")
        print(f"  Occupancy violation magnitude, arm B: mean={vb.mean():.4f}  "
              f"median={np.median(vb):.4f}  max={vb.max():.4f}")
        print(
            "  Interpretation: even without any frame crossing the flag threshold, a higher "
            "mean/median violation magnitude under degradation is itself evidence the defence "
            "is being pushed CLOSER to a false positive by benign noise -- worth reporting "
            "alongside the FPR gap itself, not just the binary flagged-or-not count, especially "
            "with a frame count this small (per-scenario FPR swings by one flagged frame here)."
        )

    if arm_a["perception_violations"] and arm_b["perception_violations"]:
        pa, pb = np.array(arm_a["perception_violations"]), np.array(arm_b["perception_violations"])
        print(f"\n  Perception violation magnitude, arm A: mean={pa.mean():.4f}  "
              f"median={np.median(pa):.4f}  max={pa.max():.4f}")
        print(f"  Perception violation magnitude, arm B: mean={pb.mean():.4f}  "
              f"median={np.median(pb):.4f}  max={pb.max():.4f}")
        print(
            f"  Frames flagged by perception-occupancy but NOT occupancy-consistency: "
            f"arm A={arm_a['n_perception_only_flags']}/{arm_a['total_frames']}, "
            f"arm B={arm_b['n_perception_only_flags']}/{arm_b['total_frames']}"
        )
        print(
            "  Interpretation: this is the direct answer to whether the detector half catches "
            "anything the free/occupied geometry check alone misses, and specifically whether "
            "it shares the occupancy-consistency check's fog-driven free-space-collapse "
            "vulnerability (documented 17 Sep 2026) or not -- a detector-based check isn't "
            "computing 'confirmed free space' via ray-casting at all, so it may not go blind "
            "under fog the same way."
        )

    print(
        "\nCaveats (see this file's + run_defence_check.py's + consistency_layer.py's module "
        "docstrings): the perception-occupancy half checks the FUSED detector's claimed boxes "
        "against each CAV's own free space (P3's cache holds one fused detection set per frame, "
        "not a per-CAV detection, so this isn't CAD's own CAV-vs-CAV detector check), no "
        "lane-area filtering, threshold borrowed from a secondary paper's analysis of CAD, "
        "OR-combination rule not verified against CAD's own source. With only 2 qualifying "
        "scenarios and 10 frames each, treat this gap as a first, small-sample signal to bring "
        "to your supervisor -- not a statistically powered result. A wider scenario/frame count "
        "is needed before this number belongs in the thesis's actual results section."
    )


if __name__ == "__main__":
    main()
