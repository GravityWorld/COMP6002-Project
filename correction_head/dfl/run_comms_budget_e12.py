"""DFL-E12: communication-budget sweep (axis 5).

Logged as DFL-E12-comms-budget in decisions_addendum_oct6.md (Thu 15 Oct
2026) -- queued to run after DFL-E11 if GPU time allows overnight, since
it reuses existing plumbing the same way E11 does: comms.py already
implements fp32/int8/1-bit encode/decode and gossip_round() already
tracks per-vehicle bytes_sent via comms_mode -- no new mechanism code,
just a sweep driver, same as E11.

Spec (task1_datasets_pipeline_dfl.md Sec 4): "fp32 / int8 / 1-bit
sign-based Delta w, swept over rounds -> bytes-vs-FPR Pareto curve." This
script runs each comms_mode under T-static topology + plain average
aggregation (no poisoning) for n_rounds, and records both dv_fpr/sv_recall
AND total bytes_sent per round per mode -- the raw numbers a bytes-vs-FPR
Pareto plot is built from directly.

Smaller grid than E11: 3 modes total (vs E11's 27 cells), so this is
queued to run second, after E11, in whatever GPU time is left.

Run from the repo root, opencood conda env active:

    python -m correction_head.dfl.run_comms_budget_e12 ^
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt ^
        --n-vehicles 15 --n-rounds 8 ^
        --out results\DFL-E12-comms-budget.json --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from correction_head.dataset import make_scenario_splits
from correction_head.model import CorrectionHead, count_parameters
from correction_head.dfl.gossip import gossip_round
from correction_head.dfl.run_baseline_arms import summarize, evaluate_all_vehicles
from correction_head.dfl.real_data import (
    build_held_out_cache,
    build_real_adjacency,
    build_real_vehicle_partition,
    flatten_params,
    make_real_local_update_fn,
)

MODES_SUPPORTED = ("fp32", "int8", "1bit")


def run_rounds_comms(vehicle_ids, seed_flat, adjacency, local_update_fn, held_out_cache,
                       n_rounds, comms_mode, cell_name, checkpoint_cb=None,
                       max_delta_norm=None, proximal_anchor=None, proximal_strength=0.0):
    """Same shape as run_baseline_arms.run_rounds(), plus forwarding
    comms_mode into gossip_round and recording bytes_sent per round --
    run_baseline_arms.run_rounds never threaded comms_mode through
    because none of the four baseline arms varied it (all four used the
    fp32 default)."""
    weights = {vid: seed_flat.copy() for vid in vehicle_ids}
    history = []
    for r in range(n_rounds):
        t0 = time.time()
        weights, diag = gossip_round(
            weights, adjacency, local_update_fn, aggregation_rule="average",
            comms_mode=comms_mode, max_delta_norm=max_delta_norm,
            proximal_anchor=proximal_anchor, proximal_strength=proximal_strength,
        )
        metrics = evaluate_all_vehicles(weights, held_out_cache, verbose=False)
        summary = summarize(metrics)
        elapsed = time.time() - t0
        bytes_total = int(sum(diag.bytes_sent.values()))
        print(f"[{cell_name}] round {r+1}/{n_rounds} ({elapsed:.1f}s) "
              f"dv_fpr mean={summary['different_vehicle_fpr']['mean']:.4f} "
              f"sv_recall mean={summary['same_vehicle_recall']['mean']:.4f} "
              f"bytes_total={bytes_total}", flush=True)
        history.append({"round": r + 1, "elapsed_s": elapsed, "summary": summary,
                         "bytes_sent_total": bytes_total,
                         "bytes_sent_per_vehicle": dict(diag.bytes_sent)})
        if checkpoint_cb is not None:
            checkpoint_cb(cell_name, history)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--proximal-strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--modes", type=str, default="fp32,int8,1bit",
                         help="Comma-separated subset of {fp32,int8,1bit}.")
    parser.add_argument("--out", type=str, required=True,
                         help="JSON results file -- also the --resume checkpoint file.")
    parser.add_argument("--resume", action="store_true",
                         help="Skip any mode already present in --out.")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES_SUPPORTED:
            raise SystemExit(f"--modes: '{m}' not in {MODES_SUPPORTED}")

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

    partition = build_real_vehicle_partition(n_vehicles=args.n_vehicles)
    vehicle_ids, static_adjacency, _traffic_by_round = build_real_adjacency(partition)
    print(f"{len(vehicle_ids)} vehicles, using T-static (fully connected) adjacency, aggregation_rule=average")

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
    local_update_fn = make_real_local_update_fn(
        vehicle_rows, n_local_steps=args.n_local_steps, lr=args.lr,
        different_vehicle_weight=diff_veh_w, other_weight=other_w,
    )

    results = {}
    if args.resume and Path(args.out).exists():
        with open(args.out) as f:
            results = json.load(f)
        print(f"--resume: loaded {len(results)} already-completed modes from {args.out}")

    def checkpoint_cb(cell_name, history):
        results[cell_name] = history
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(args.out).with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=2)
        tmp_path.replace(args.out)

    for i, mode in enumerate(modes):
        if args.resume and mode in results:
            print(f"[{i+1}/{len(modes)}] mode={mode}: already done, skipping")
            continue
        print(f"\n[{i+1}/{len(modes)}] mode={mode}")
        run_rounds_comms(
            vehicle_ids, seed_flat, static_adjacency, local_update_fn, held_out_cache,
            args.n_rounds, comms_mode=mode, cell_name=mode,
            checkpoint_cb=checkpoint_cb, max_delta_norm=args.max_delta_norm,
            proximal_anchor=seed_flat, proximal_strength=args.proximal_strength,
        )

    print("\n" + "=" * 78)
    print("SUMMARY (final round, mean dv_fpr / sv_recall / bytes_total, per mode)")
    print("=" * 78)
    for mode in modes:
        if mode not in results:
            continue
        final = results[mode][-1]
        print(f"  {mode:8s} dv_fpr={final['summary']['different_vehicle_fpr']['mean']:.4f}  "
              f"sv_recall={final['summary']['same_vehicle_recall']['mean']:.4f}  "
              f"bytes_total={final['bytes_sent_total']}")

    print(f"\nFull results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
