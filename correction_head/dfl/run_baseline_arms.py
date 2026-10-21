"""Four baseline arms for the DFL depth experiments, per
task1_datasets_pipeline_dfl.md Sec 4: frozen head (floor) / local-only /
centralised FedAvg (ceiling) / DFL. This is "the first step" after the v7
checkpoint was locked and the real-data scaffold (real_data.py,
run_real_scaffold.py) was verified end-to-end -- per the explicit Thu 10
Oct instruction to not drop or de-prioritise any of the five axes and just
get the first concrete result on the board.

All three non-frozen arms reuse gossip_round() unchanged, varying only the
adjacency input and (for local-only) nothing else -- this is deliberate,
not a shortcut: it is how the project's own design keeps the three arms
comparable (same local_update_fn, same aggregation code path, same eval).

  - frozen head  : evaluate the seed checkpoint directly, zero rounds.
  - local-only   : gossip_round with an all-isolated adjacency (every
                    vehicle's neighbour set is empty) -> aggregate_average
                    reduces to pure local SGD, no collaboration at all.
  - centralised  : T-static adjacency (fully connected) + aggregation_rule
    FedAvg (ceiling)  ="average", for N rounds.
  - DFL          : T-traffic per-round adjacency (real distinct rounds from
                    build_real_adjacency, cycled) + aggregation_rule=
                    "average" (same rule as the ceiling arm -- isolates the
                    topology effect specifically from the aggregation-rule
                    effect, which is axis 2's own comparison).

Evaluation per round: every vehicle's OWN current weights evaluated
independently against the fixed held-out val scenarios (the same
make_scenario_splits(seed=0, val_frac=0.2, test_frac=0.2) split train_v1.py
used to produce v7 -- so this reports against the exact same held-out set
v7's locked numbers came from). Reports mean/min/max across vehicles per
round rather than averaging weights first (would bias toward
centralisation) or picking one vehicle arbitrarily.

Needs torch + the real P0/P3/P4/P5 cache + a correction_head checkpoint on
disk. Run from the repo root, opencood conda env active:

    python -m correction_head.dfl.run_baseline_arms \
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt \
        --n-vehicles 15 --n-rounds 10 --n-local-steps 5 --lr 0.01

This is NOT yet the full poisoning x aggregation grid or the diffusion x
topology centrepiece (axes 3 and the centrepiece) -- those are the next
two steps. This script establishes the four reference curves (floor /
local-only / ceiling / DFL) that every later result gets compared against.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from correction_head.dataset import make_scenario_splits
from correction_head.model import CorrectionHead, count_parameters
from correction_head.dfl.gossip import gossip_round
from correction_head.dfl.real_data import (
    build_held_out_cache,
    build_real_adjacency,
    build_real_vehicle_partition,
    evaluate_on_held_out_cached,
    flatten_params,
    make_real_local_update_fn,
    unflatten_into,
    DEFAULT_DEVICE,
)


def summarize(per_vehicle_metrics: dict) -> dict:
    """per_vehicle_metrics: {vid: evaluate_on_held_out(...) dict}. Returns
    mean/min/max different_vehicle_fpr and same_vehicle_recall across
    vehicles, skipping NaNs (empty-bucket vehicles)."""
    fprs = np.array([m["different_vehicle_fpr"] for m in per_vehicle_metrics.values()], dtype=np.float64)
    recalls = np.array([m["same_vehicle_recall"] for m in per_vehicle_metrics.values()], dtype=np.float64)
    fprs = fprs[~np.isnan(fprs)]
    recalls = recalls[~np.isnan(recalls)]

    def stats(arr):
        if arr.size == 0:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
        return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

    return {"different_vehicle_fpr": stats(fprs), "same_vehicle_recall": stats(recalls)}


def weights_to_model(flat: np.ndarray, device=DEFAULT_DEVICE) -> CorrectionHead:
    m = CorrectionHead().to(device)
    unflatten_into(m, flat)
    m.eval()
    return m


def evaluate_all_vehicles(weights: dict, held_out_cache: list, device=DEFAULT_DEVICE, verbose: bool = False) -> dict:
    """Evaluates every vehicle's own weights against a precomputed held-out
    cache (build_held_out_cache(), built ONCE at the start of main()) --
    this is the fix for the ~55s/vehicle/round cost the first real run hit,
    almost none of which was model compute (see build_held_out_cache's
    docstring). Still dedupes identical weight vectors (matters for the
    frozen-head arm, where all 15 vehicles share the same seed checkpoint).
    """
    out = {}
    seen: list = []  # list of (flat_array, result_dict)
    n = len(weights)
    for i, (vid, flat) in enumerate(weights.items()):
        match = next((r for f, r in seen if np.array_equal(f, flat)), None)
        if match is not None:
            out[vid] = match
            continue
        t0 = time.time()
        model = weights_to_model(flat, device=device)
        result = evaluate_on_held_out_cached(model, held_out_cache)
        del model
        if verbose:
            print(f"    vehicle {i+1}/{n} eval took {time.time()-t0:.1f}s", flush=True)
        out[vid] = result
        seen.append((flat, result))
    return out


def run_rounds(vehicle_ids, seed_flat, adjacency_fn, local_update_fn, held_out_cache,
                n_rounds: int, aggregation_rule: str, arm_name: str,
                checkpoint_cb=None, max_delta_norm=None,
                proximal_anchor=None, proximal_strength: float = 0.0) -> list:
    """adjacency_fn(round_idx) -> adjacency dict for that round (lets DFL cycle
    through real traffic rounds while static/local-only/frozen use a fixed one)."""
    weights = {vid: seed_flat.copy() for vid in vehicle_ids}
    history = []
    for r in range(n_rounds):
        t0 = time.time()
        adjacency = adjacency_fn(r)
        # One gossip_round call for the WHOLE network per round -- gossip_round
        # already computes local_update_fn(vid, ...) once per vid in `weights`
        # and aggregates each vid's own delta with its neighbours' deltas using
        # the full adjacency dict (this is the same single-call pattern
        # run_real_scaffold.py's smoke test used, and timed at ~41.5s/round for
        # 15 vehicles). An earlier version of this function called gossip_round
        # once PER VEHICLE, each time re-passing that vehicle's full neighbour
        # set as its own "weights" dict -- for T-static (fully connected) that
        # meant every one of the 15 outer calls redundantly recomputed
        # local_update_fn for all 14 neighbours too, 225 calls/round instead of
        # 15, a 15x slowdown. Fixed to the single-call-per-round form below.
        weights, diag = gossip_round(
            weights, adjacency, local_update_fn, aggregation_rule=aggregation_rule,
            max_delta_norm=max_delta_norm,
            proximal_anchor=proximal_anchor, proximal_strength=proximal_strength,
        )
        metrics = evaluate_all_vehicles(weights, held_out_cache, verbose=True)
        summary = summarize(metrics)
        elapsed = time.time() - t0
        print(f"[{arm_name}] round {r+1}/{n_rounds} ({elapsed:.1f}s) "
              f"dv_fpr mean={summary['different_vehicle_fpr']['mean']:.4f} "
              f"(min={summary['different_vehicle_fpr']['min']:.4f}, max={summary['different_vehicle_fpr']['max']:.4f})  "
              f"sv_recall mean={summary['same_vehicle_recall']['mean']:.4f} "
              f"(min={summary['same_vehicle_recall']['min']:.4f}, max={summary['same_vehicle_recall']['max']:.4f})")
        history.append({"round": r + 1, "elapsed_s": elapsed, "summary": summary})
        if checkpoint_cb is not None:
            checkpoint_cb(arm_name, history)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Seed checkpoint for all four arms (e.g. v7_bucketfix_w40_model.pt).")
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)  # matches train_v1.py's Adam default -- see real_data.py's local_update_fn comment (Sat 11 Oct optimizer fix)
    parser.add_argument("--max-delta-norm", type=float, default=None,
                         help="Optional L2-norm clip on each vehicle's whole local delta "
                              "before aggregation -- a second safety net on top of the "
                              "per-step grad-norm clip inside local_update_fn (Sat 11 Oct). "
                              "run_real_scaffold.py's smoke test used 50.0 for this; left "
                              "unset (off) by default here since the Adam+grad-clip fix "
                              "should already be enough -- pass e.g. --max-delta-norm 50.0 "
                              "if a rerun still shows wild per-round swings.")
    parser.add_argument("--proximal-strength", type=float, default=0.0,
                         help="gossip_round's own proximal-anchor term (Sec 4 step 5, built "
                              "in but never wired up here before Sun 12 Oct): each round, "
                              "pulls that vehicle's updated weights back toward the v7 SEED "
                              "checkpoint by this fraction (0 = off, matches every prior run; "
                              "try e.g. 0.1-0.3). Added because the Sun 12 Oct local-only test "
                              "showed training now stabilizes (confirming the swings do calm "
                              "down) but settles at dv_fpr ~18-20%% -- far worse than the "
                              "frozen floor's 4.21%% -- because 5 Adam steps on a tiny, "
                              "unrepresentative per-vehicle buffer drifts the decision boundary "
                              "away from v7's tuned FPR/recall balance. Anchoring toward v7 "
                              "should let local SGD still adapt without wandering that far.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0,
                         help="Seed for make_scenario_splits -- default 0 matches "
                              "the split train_v1.py used to produce v7's locked numbers.")
    parser.add_argument("--arms", type=str, default="frozen,local_only,centralised_fedavg,dfl",
                         help="Comma-separated subset of {frozen,local_only,centralised_fedavg,dfl} "
                              "to run -- e.g. --arms local_only to test just one arm over more "
                              "rounds for a fraction of the cost (all four arms cost the same "
                              "per round, so --arms local_only --n-rounds 8 is ~1/3 the cost of "
                              "the full default run, not faster per round, just fewer arms).")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to dump the full results dict as JSON.")
    args = parser.parse_args()
    selected_arms = {a.strip() for a in args.arms.split(',') if a.strip()}

    torch.manual_seed(args.seed)  # (np.random.default_rng(args.seed) was created here before but never used -- dropped)

    split = make_scenario_splits(
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed,
    )
    print(f"Held-out val scenarios ({len(split.val_scenarios)}): {sorted(split.val_scenarios)}")
    if not split.val_scenarios:
        print("ABORTING: val split is empty.")
        return 1

    print("Building held-out eval cache once (crops/pair tensors + ground-truth "
          "buckets for every held-out unit) -- every later eval call reuses this "
          "instead of reloading from disk.")
    t_cache = time.time()
    held_out_cache = build_held_out_cache(split.val_scenarios)
    print(f"held-out cache built in {time.time()-t_cache:.1f}s, {len(held_out_cache)} units")

    print("=" * 78)
    print("Building real vehicle partition + adjacency from P6")
    print("=" * 78)
    partition = build_real_vehicle_partition(n_vehicles=args.n_vehicles)
    vehicle_ids, static_adjacency, traffic_by_round = build_real_adjacency(partition)
    print(f"{len(vehicle_ids)} vehicles, {len(traffic_by_round)} distinct real traffic rounds")

    model_template = CorrectionHead()
    real_param_count = count_parameters(model_template)
    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_template.load_state_dict(state_dict)
    seed_flat = flatten_params(model_template)
    assert seed_flat.shape[0] == real_param_count

    vehicle_rows = {vid: group for vid, group in partition.groupby("vehicle_id")}
    diff_veh_w, other_w = 40.0, 1.0  # must match v7's own --different-vehicle-weight/--other-weight
    print(f"\nLocal SGD loss: weighted BCE, different_vehicle_weight={diff_veh_w}, "
          f"other_weight={other_w} (fix for the unweighted-loss bug found Sat 11 Oct -- "
          "if you don't see this line, you're looking at output from before the fix)\n")
    local_update_fn = make_real_local_update_fn(
        vehicle_rows, n_local_steps=args.n_local_steps, lr=args.lr,
        different_vehicle_weight=diff_veh_w, other_weight=other_w,
    )

    results = {}

    def checkpoint_cb(arm_name, history):
        """Writes --out after EVERY round, not just at the end -- so a crash
        (this is exactly what happened the first time this script was run:
        the machine crashed mid-run and nothing had been saved) loses at
        most one round of one arm, not the whole run.

        BUG FIXED Sat 11 Oct (found on a full re-read, not from a failed
        run -- every run so far happened to pass --out so this never
        surfaced): results[arm_name] was only ever set INSIDE the `if not
        args.out: return` guard, so running this script without --out would
        silently leave `results` empty for every arm and print a blank
        SUMMARY with no error. Populating results[arm_name] now happens
        unconditionally; only the disk write is gated on --out."""
        results[arm_name] = history
        if not args.out:
            return
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(args.out).with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=2)
        tmp_path.replace(args.out)  # atomic-ish on same filesystem: never leaves a half-written --out

    if "frozen" in selected_arms:
        print()
        print("=" * 78)
        print("ARM 1/4: FROZEN HEAD (floor) -- seed checkpoint, zero rounds")
        print("=" * 78)
        frozen_metrics = evaluate_all_vehicles({vid: seed_flat for vid in vehicle_ids}, held_out_cache, verbose=True)
        frozen_summary = summarize(frozen_metrics)
        print(f"[frozen] dv_fpr mean={frozen_summary['different_vehicle_fpr']['mean']:.4f}  "
              f"sv_recall mean={frozen_summary['same_vehicle_recall']['mean']:.4f}")
        checkpoint_cb("frozen_head", [{"round": 0, "elapsed_s": 0.0, "summary": frozen_summary}])
    else:
        print("\nSkipping ARM 1/4 (frozen) -- not in --arms\n")

    if "local_only" in selected_arms:
        print()
        print("=" * 78)
        print("ARM 2/4: LOCAL-ONLY -- all-isolated adjacency, no collaboration")
        print("=" * 78)
        isolated_adjacency = {vid: set() for vid in vehicle_ids}
        results["local_only"] = run_rounds(
            vehicle_ids, seed_flat, lambda r: isolated_adjacency, local_update_fn,
            held_out_cache, args.n_rounds, aggregation_rule="average", arm_name="local_only",
            checkpoint_cb=checkpoint_cb, max_delta_norm=args.max_delta_norm,
            proximal_anchor=seed_flat, proximal_strength=args.proximal_strength,
        )
    else:
        print("\nSkipping ARM 2/4 (local_only) -- not in --arms\n")

    if "centralised_fedavg" in selected_arms:
        print()
        print("=" * 78)
        print("ARM 3/4: CENTRALISED FEDAVG (ceiling) -- T-static, average aggregation")
        print("=" * 78)
        results["centralised_fedavg"] = run_rounds(
            vehicle_ids, seed_flat, lambda r: static_adjacency, local_update_fn,
            held_out_cache, args.n_rounds, aggregation_rule="average", arm_name="centralised_fedavg",
            checkpoint_cb=checkpoint_cb, max_delta_norm=args.max_delta_norm,
            proximal_anchor=seed_flat, proximal_strength=args.proximal_strength,
        )
    else:
        print("\nSkipping ARM 3/4 (centralised_fedavg) -- not in --arms\n")

    if "dfl" in selected_arms:
        print()
        print("=" * 78)
        print("ARM 4/4: DFL -- T-traffic per-round adjacency (real rounds, cycled), average aggregation")
        print("=" * 78)
        traffic_rounds_list = sorted(traffic_by_round.keys())
        if not traffic_rounds_list:
            print("WARNING: no real traffic rounds found -- DFL arm will fall back to "
                  "T-static for every round (reduces this arm to a duplicate of the "
                  "centralised arm, which is a real data gap, not a bug in this script).")
            dfl_adjacency_fn = lambda r: static_adjacency
        else:
            def dfl_adjacency_fn(r):
                round_key = traffic_rounds_list[r % len(traffic_rounds_list)]
                adj = traffic_by_round[round_key]
                for vid in vehicle_ids:
                    adj.setdefault(vid, set())
                return adj
        results["dfl"] = run_rounds(
            vehicle_ids, seed_flat, dfl_adjacency_fn, local_update_fn,
            held_out_cache, args.n_rounds, aggregation_rule="average", arm_name="dfl",
            checkpoint_cb=checkpoint_cb, max_delta_norm=args.max_delta_norm,
            proximal_anchor=seed_flat, proximal_strength=args.proximal_strength,
        )
    else:
        print("\nSkipping ARM 4/4 (dfl) -- not in --arms\n")

    print()
    print("=" * 78)
    print("SUMMARY (final round, mean across vehicles)")
    print("=" * 78)
    for arm_name, history in results.items():
        final = history[-1]["summary"]
        print(f"  {arm_name:20s} dv_fpr={final['different_vehicle_fpr']['mean']:.4f}  "
              f"sv_recall={final['same_vehicle_recall']['mean']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
