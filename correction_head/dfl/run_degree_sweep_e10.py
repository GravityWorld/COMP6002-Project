"""DFL-E10: fixed-degree connectivity sweep.

DFL-E08 (run_baseline_arms.py) only gave us two points on the connectivity
axis: local_only (0 neighbours) and centralised_fedavg (14 neighbours,
T-static full mesh) -- a straight jump from nothing to everything. DFL-E09
then found the REAL T-traffic adjacency averages ~0.2 neighbours/round,
explaining why the real `dfl` arm tracks local_only so closely: the
averaging mechanism almost never actually fires.

This script fills in the gap: k = 2, 5, 10, (14) random (not real-traffic)
neighbours per vehicle, every round, via
correction_head.dfl.topology.build_fixed_degree_topology. It answers "how
much does the averaging mechanism itself help as a function of how many
neighbours you average with" -- a dose-response curve between local_only
and centralised_fedavg, independent of whether that many neighbours is
realistic (it isn't, per E09; that's the point of E09, not this script).

Deliberately reuses run_baseline_arms.run_rounds/evaluate_all_vehicles
unchanged -- same local_update_fn, same gossip_round call, same eval cache,
so these numbers are directly comparable to DFL-E08's local_only (k=0) and
centralised_fedavg (k=14) rows, not a separately-tuned pipeline.

Needs torch + the real P0/P3/P4/P5 cache + a correction_head checkpoint on
disk. Run from the repo root, opencood conda env active:

    python -m correction_head.dfl.run_degree_sweep_e10 \
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt \
        --n-vehicles 15 --n-rounds 10 --n-local-steps 5 \
        --degrees 0,2,5,10,14 \
        --out results\DFL-E10-degree-sweep.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from correction_head.dataset import make_scenario_splits
from correction_head.model import CorrectionHead, count_parameters
from correction_head.dfl.topology import build_fixed_degree_topology, topology_stats
from correction_head.dfl.real_data import (
    build_held_out_cache,
    build_real_vehicle_partition,
    build_real_adjacency,
    flatten_params,
    make_real_local_update_fn,
    DEFAULT_DEVICE,
)
from correction_head.dfl.run_baseline_arms import run_rounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Seed checkpoint, same as run_baseline_arms.py (e.g. v7_bucketfix_w40_model.pt).")
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--proximal-strength", type=float, default=0.0,
                         help="Left at 0 by default, matching DFL-E08's final run, so this "
                              "sweep is a clean read on the averaging mechanism alone -- see "
                              "run_baseline_arms.py's own flag docstring for why this exists.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0,
                         help="Default 0 matches the split train_v1.py used to produce v7's "
                              "locked numbers, and matches run_baseline_arms.py's default.")
    parser.add_argument("--degrees", type=str, default="0,2,5,10,14",
                         help="Comma-separated list of fixed degrees (k) to sweep. k=0 is "
                              "identical to run_baseline_arms.py's local_only arm; k>=n_vehicles-1 "
                              "(14 at the default --n-vehicles 15) is identical to its "
                              "centralised_fedavg arm (full mesh) -- included so this sweep's "
                              "own endpoints are directly comparable to DFL-E08 without having "
                              "to splice two separate runs together.")
    parser.add_argument("--graph-seed", type=int, default=0,
                         help="Seed for the random fixed-degree graph generator itself, "
                              "separate from --seed (torch's seed). Same seed reused across "
                              "every k in the sweep for consistency, not because it matters "
                              "much at this scale.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    degrees = [int(k.strip()) for k in args.degrees.split(",") if k.strip() != ""]

    torch.manual_seed(args.seed)

    split = make_scenario_splits(val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed)
    print(f"Held-out val scenarios ({len(split.val_scenarios)}): {sorted(split.val_scenarios)}")
    if not split.val_scenarios:
        print("ABORTING: val split is empty.")
        return 1

    print("Building held-out eval cache once...")
    t_cache = time.time()
    held_out_cache = build_held_out_cache(split.val_scenarios)
    print(f"held-out cache built in {time.time()-t_cache:.1f}s, {len(held_out_cache)} units")

    print("=" * 78)
    print("Building real vehicle partition from P6 (only used for vehicle_ids + local rows here -- "
          "this sweep's adjacency is synthetic, not the real traffic adjacency)")
    print("=" * 78)
    partition = build_real_vehicle_partition(n_vehicles=args.n_vehicles)
    vehicle_ids, _static_adjacency, _traffic_by_round = build_real_adjacency(partition)
    print(f"{len(vehicle_ids)} vehicles")

    model_template = CorrectionHead()
    real_param_count = count_parameters(model_template)
    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_template.load_state_dict(state_dict)
    seed_flat = flatten_params(model_template)
    assert seed_flat.shape[0] == real_param_count

    vehicle_rows = {vid: group for vid, group in partition.groupby("vehicle_id")}
    diff_veh_w, other_w = 40.0, 1.0
    print(f"\nLocal SGD loss: weighted BCE, different_vehicle_weight={diff_veh_w}, other_weight={other_w}\n")
    local_update_fn = make_real_local_update_fn(
        vehicle_rows, n_local_steps=args.n_local_steps, lr=args.lr,
        different_vehicle_weight=diff_veh_w, other_weight=other_w,
    )

    results = {}

    def checkpoint_cb(arm_name, history):
        results[arm_name] = history
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            tmp_path = Path(args.out).with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            tmp_path.replace(args.out)

    degree_stats = {}
    for k in degrees:
        arm_name = f"degree_{k}"
        adjacency = build_fixed_degree_topology(vehicle_ids, k, seed=args.graph_seed)
        stats = topology_stats(adjacency)
        degree_stats[arm_name] = stats
        print("=" * 78)
        print(f"ARM: {arm_name} -- requested k={k}, realised mean_degree={stats['mean_degree']:.2f}, "
              f"n_isolated={stats['n_isolated']}")
        print("=" * 78)
        results[arm_name] = run_rounds(
            vehicle_ids, seed_flat, lambda r, adj=adjacency: adj, local_update_fn,
            held_out_cache, args.n_rounds, aggregation_rule="average", arm_name=arm_name,
            checkpoint_cb=checkpoint_cb, max_delta_norm=args.max_delta_norm,
            proximal_anchor=seed_flat, proximal_strength=args.proximal_strength,
        )

    print()
    print("=" * 78)
    print("SUMMARY (final round, mean across vehicles)")
    print("=" * 78)
    for k in degrees:
        arm_name = f"degree_{k}"
        final = results[arm_name][-1]["summary"]
        realised = degree_stats[arm_name]["mean_degree"]
        print(f"  k={k:2d} (realised degree={realised:5.2f})  "
              f"dv_fpr={final['different_vehicle_fpr']['mean']:.4f}  "
              f"sv_recall={final['same_vehicle_recall']['mean']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results, "degree_stats": degree_stats}, f, indent=2)
        print(f"\nFull results written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
