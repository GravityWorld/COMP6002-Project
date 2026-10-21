"""Reads every results/DFL-E*.json (plus baseline_arms*.json) and extracts
one clean final-round summary table per arm/cell/topology -- so report
numbers get copied from one table instead of hand-picked out of raw
per-round JSON. Three known result shapes, auto-detected per file:

  A) "flat run_rounds" (DFL-E08, E10, E11, E12, baseline_arms*): top-level
     dict is itself {arm_or_cell_name: [ {round, elapsed_s, summary:
     {different_vehicle_fpr:{mean,min,max}, same_vehicle_recall:{...}}},
     ... ]}.
  B) "wrapped run_rounds" (DFL-E14): top-level has a "results" key whose
     value has shape A, plus extra metadata alongside it.
  C) "per-vehicle diffusion" (DFL-E13): top-level has "results":
     {topology: [ {round, metrics: {vehicle_id: {overall:{...},
     regime:{...}}}} ]}, plus "seed_vehicle"/"target_bucket"/"summary"
     (hop distance + rounds-to-80%-transfer -- only populated once that
     topology's run has actually finished; a mid-run file still parses,
     just with an empty note instead of the hop/transfer numbers).

Safe to run against partially-written / still-running output (every DFL
script here checkpoints after every round via an atomic tmp-file rename),
and safe to re-run any time -- it only reads, never writes inside results/
except the two SUMMARY files it produces.

Run from the repo root (plain python, no torch/pandas needed):
    python summarize_dfl_results.py
    python summarize_dfl_results.py --results-dir results --out-prefix results/SUMMARY
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def is_flat_run_rounds(d) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    for v in d.values():
        if not isinstance(v, list) or not v:
            return False
        if not isinstance(v[-1], dict) or "summary" not in v[-1]:
            return False
    return True


def fmt(x, nd=4):
    try:
        if x is None:
            return ""
        if isinstance(x, float) and x != x:  # NaN
            return "nan"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def rows_from_flat(file_name: str, flat: dict) -> list:
    rows = []
    for name, history in flat.items():
        if not history:
            continue
        final = history[-1]
        summary = final.get("summary", {})
        dv = summary.get("different_vehicle_fpr", {})
        sv = summary.get("same_vehicle_recall", {})
        rows.append({
            "file": file_name,
            "axis_or_arm": name,
            "final_round": final.get("round"),
            "n_rounds_saved": len(history),
            "dv_fpr_mean": fmt(dv.get("mean")),
            "dv_fpr_min": fmt(dv.get("min")),
            "dv_fpr_max": fmt(dv.get("max")),
            "sv_recall_mean": fmt(sv.get("mean")),
            "sv_recall_min": fmt(sv.get("min")),
            "sv_recall_max": fmt(sv.get("max")),
            "note": "",
        })
    return rows


def rows_from_diffusion(file_name: str, doc: dict) -> list:
    rows = []
    seed_vid = doc.get("seed_vehicle")
    target_bucket = doc.get("target_bucket")
    results = doc.get("results", {})
    summary = doc.get("summary", {})
    for topo, history in results.items():
        if not history:
            continue
        final = history[-1]
        metrics = final.get("metrics", {})
        topo_summary = summary.get(topo, {})
        per_vehicle_summary = topo_summary.get("per_vehicle", {})
        n_reached = sum(1 for v in per_vehicle_summary.values() if v.get("rounds_to_80pct_transfer") is not None)
        n_vehicles = len(per_vehicle_summary) or len(metrics)
        seed_metrics = metrics.get(str(seed_vid), metrics.get(seed_vid, {}))
        seed_regime = seed_metrics.get("regime", {})
        if topo_summary:
            note = (f"seed_final_gain={fmt(topo_summary.get('seed_final_gain'))}; "
                    f"{n_reached}/{n_vehicles} vehicles reached 80%-transfer")
        else:
            note = f"run not finished ({len(history)-1} rounds saved so far) -- summary not yet computed"
        rows.append({
            "file": file_name,
            "axis_or_arm": f"diffusion_{topo} (seed={seed_vid}, bucket={target_bucket})",
            "final_round": final.get("round"),
            "n_rounds_saved": len(history),
            "dv_fpr_mean": fmt(seed_regime.get("different_vehicle_fpr")),
            "dv_fpr_min": "",
            "dv_fpr_max": "",
            "sv_recall_mean": fmt(seed_regime.get("same_vehicle_recall")),
            "sv_recall_min": "",
            "sv_recall_max": "",
            "note": note,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-prefix", type=str, default="results/SUMMARY")
    parser.add_argument("--pattern", type=str, default="*.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_rows = []
    skipped = []

    for path in sorted(results_dir.glob(args.pattern)):
        if path.name.endswith("_log.txt"):
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception as e:
            skipped.append((path.name, f"unreadable JSON: {e}"))
            continue

        if isinstance(doc, dict) and "results" in doc and isinstance(doc["results"], dict):
            inner = doc["results"]
            if is_flat_run_rounds(inner):
                all_rows.extend(rows_from_flat(path.name, inner))
            else:
                all_rows.extend(rows_from_diffusion(path.name, doc))
        elif is_flat_run_rounds(doc):
            all_rows.extend(rows_from_flat(path.name, doc))
        else:
            skipped.append((path.name, "unrecognised shape -- not one of the known run_rounds/diffusion layouts"))

    if not all_rows:
        print("No recognised result rows found.")
    else:
        fieldnames = ["file", "axis_or_arm", "final_round", "n_rounds_saved",
                      "dv_fpr_mean", "dv_fpr_min", "dv_fpr_max",
                      "sv_recall_mean", "sv_recall_min", "sv_recall_max", "note"]
        csv_path = Path(f"{args.out_prefix}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {csv_path}")

        md_path = Path(f"{args.out_prefix}.md")
        with open(md_path, "w") as f:
            f.write("| " + " | ".join(fieldnames) + " |\n")
            f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
            for row in all_rows:
                f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")
        print(f"Wrote markdown table to {md_path}")

        print()
        print(f"{'file':38s} {'axis_or_arm':45s} {'dv_fpr_mean':>12s} {'sv_recall_mean':>15s}")
        for row in all_rows:
            print(f"{row['file']:38s} {row['axis_or_arm']:45s} {row['dv_fpr_mean']:>12s} {row['sv_recall_mean']:>15s}")

    if skipped:
        print("\nSkipped files:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
