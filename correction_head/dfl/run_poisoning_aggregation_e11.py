"""DFL-E11: combined poisoning (axis 3) x aggregation-rule (axis 2) grid.

Logged as DFL-E11-poisoning-x-aggregation in decisions_addendum_oct6.md
(Thu 15 Oct 2026) -- the scoping doc (task1_datasets_pipeline_dfl.md Sec 4)
names poisoning x aggregation as the one combined result that's NOT an
acceptable thing to drop ("a missing poisoning curve is not" an acceptable
fallback), so this is the priority grid for the overnight run.

Topology: T-static (fully connected), same as run_baseline_arms.py's
centralised_fedavg arm. Deliberately NOT T-traffic/dfl -- DFL-E09 already
found real traffic adjacency is ~84% isolated per vehicle-round, so a
poisoning attack would almost never even reach a neighbour to corrupt.
Isolating the poisoning x aggregation question from the topology-sparsity
question (already reported via E08/E09) means running this on the one
topology where aggregation actually happens most of the time.

Aggregation rules covered: average, trimmed_mean, median. trust_weighted
is DELIBERATELY EXCLUDED -- aggregation.py's own docstring says its trust
scores are meant to come from P4's occupancy-consistency check / P5's
retrospective-consistency signal, neither of which exists yet as a wired
trust source. Running trust_weighted with trust=1.0 for every neighbour
would just silently degrade to average and misreport as if trust-weighting
had been tested; skipping it and saying so is more honest than that.

Poisoning: f in {0, 10, 20, 30, 40}% malicious vehicles (poisoning.py's
F_VALUES), crossed with both attack models poisoning.py implements --
large_norm (blunt outlier, no defence under plain average) and sign_flip
(bounded-norm adversarial direction, the harder case for the robust
rules). f=0 is run once per rule with no attack (malicious set is empty
regardless of attack choice, so running it twice would be a wasted,
identical cell).

No new mechanism code: gossip_round() already accepts malicious_ids/
attack/rng and dispatches through aggregation.py's RULES exactly as E08's
four arms do; this script only adds the sweep driver + an explicit,
reproducible per-run rng (run_baseline_arms.run_rounds never threaded an
rng into gossip_round because no arm there used poisoning -- gossip_round
defaults to a FRESH unseeded np.random.default_rng() every call if none
is passed, which would make attack_large_norm's random corruption
direction non-reproducible across rounds/reruns. This script's
run_rounds_poisoned() is run_baseline_arms.run_rounds() plus threading one
seeded rng through every round of a given cell, plus forwarding
malicious_ids/attack.)

Checkpointing: writes --out after every grid CELL (not every round --
each cell is already its own run_rounds_poisoned() call with n_rounds
rounds, so a crash loses at most the in-progress cell). Pass --resume to
skip any cell key already present in an existing --out file -- same key
format every time, so rerunning the exact same command after a crash
picks up where it left off.

Run from the repo root, opencood conda env active:

    python -m correction_head.dfl.run_poisoning_aggregation_e11 ^
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt ^
        --n-vehicles 15 --n-rounds 8 ^
        --out results\DFL-E11-poisoning-x-aggregation.json --resume

(Windows cmd line-continuation is ^, not \ -- adjust if pasting into a
different shell.)

Default grid (3 rules x 5 f-values x 2 attacks, minus the f=0 duplicate
per rule) is 27 cells. At roughly the same per-round cost run_baseline_arms
saw on this machine (order of a minute/round at 15 vehicles, fully
connected, non-dedupable weights most rounds), 27 cells x 8 rounds is a
multi-hour run -- exactly the "leave it running overnight" shape. Pass
--n-rounds 5 to shrink it, or narrow --f-values/--attacks/--rules, if you'd
rather trade grid coverage for a shorter run.
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
from correction_head.dfl.poisoning import ATTACKS, select_malicious_vehicles
from correction_head.dfl.run_baseline_arms import summarize, evaluate_all_vehicles
from correction_head.dfl.real_data import (
    build_held_out_cache,
    build_real_adjacency,
    build_real_vehicle_partition,
    flatten_params,
    make_real_local_update_fn,
)

RULES_SUPPORTED = ("average", "trimmed_mean", "median")  # trust_weighted excluded, see module docstring


def run_rounds_poisoned(vehicle_ids, seed_flat, adjacency, local_update_fn, held_out_cache,
                          n_rounds, aggregation_rule, cell_name,
                          malicious_ids, attack, rng, checkpoint_cb=None,
                          max_delta_norm=None, proximal_anchor=None, proximal_strength=0.0):
    """Same shape as run_baseline_arms.run_rounds(), plus forwarding
    malicious_ids/attack/a single persistent rng into gossip_round every
    round (see module docstring for why run_baseline_arms.run_rounds can't
    be reused unchanged here). `adjacency` is fixed (T-static) rather than
    a per-round adjacency_fn -- this grid isolates poisoning x aggregation
    from the topology question, which E08/E09 already cover separately."""
    weights = {vid: seed_flat.copy() for vid in vehicle_ids}
    history = []
    for r in range(n_rounds):
        t0 = time.time()
        weights, diag = gossip_round(
            weights, adjacency, local_update_fn, aggregation_rule=aggregation_rule,
            max_delta_norm=max_delta_norm, malicious_ids=malicious_ids, attack=attack,
            rng=rng, proximal_anchor=proximal_anchor, proximal_strength=proximal_strength,
        )
        metrics = evaluate_all_vehicles(weights, held_out_cache, verbose=False)
        summary = summarize(metrics)
        elapsed = time.time() - t0
        print(f"[{cell_name}] round {r+1}/{n_rounds} ({elapsed:.1f}s) "
              f"dv_fpr mean={summary['different_vehicle_fpr']['mean']:.4f} "
              f"sv_recall mean={summary['same_vehicle_recall']['mean']:.4f}", flush=True)
        history.append({"round": r + 1, "elapsed_s": elapsed, "summary": summary,
                         "bytes_sent_total": int(sum(diag.bytes_sent.values()))})
        if checkpoint_cb is not None:
            checkpoint_cb(cell_name, history)
    return history


def cell_key(rule, f, attack):
    return f"rule={rule}_f={f:.2f}_attack={attack}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--proximal-strength", type=float, default=0.0,
                         help="Same proximal-anchor knob as run_baseline_arms.py. Left at 0 "
                              "by default to match E08's reported numbers; E08/E09 did not "
                              "find a clear benefit from a nonzero value, so there's no "
                              "established reason to turn it on here either.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--rules", type=str, default="average,trimmed_mean,median",
                         help="Comma-separated subset of {average,trimmed_mean,median}. "
                              "trust_weighted is not offered -- see module docstring.")
    parser.add_argument("--f-values", type=str, default="0.0,0.10,0.20,0.30,0.40",
                         help="Comma-separated malicious-vehicle fractions.")
    parser.add_argument("--attacks", type=str, default="large_norm,sign_flip",
                         help="Comma-separated subset of {large_norm,sign_flip}.")
    parser.add_argument("--out", type=str, required=True,
                         help="JSON results file -- also the --resume checkpoint file.")
    parser.add_argument("--resume", action="store_true",
                         help="Skip any cell whose key is already present in --out.")
    args = parser.parse_args()

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    for r in rules:
        if r not in RULES_SUPPORTED:
            raise SystemExit(f"--rules: '{r}' not in {RULES_SUPPORTED} (trust_weighted excluded, see module docstring)")
    f_values = [float(x) for x in args.f_values.split(",") if x.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    for a in attacks:
        if a not in ATTACKS:
            raise SystemExit(f"--attacks: '{a}' not in {tuple(ATTACKS.keys())}")

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
    print(f"{len(vehicle_ids)} vehicles, using T-static (fully connected) adjacency for this grid")

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
        print(f"--resume: loaded {len(results)} already-completed cells from {args.out}")

    def checkpoint_cb(cell_name, history):
        results[cell_name] = history
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(args.out).with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=2)
        tmp_path.replace(args.out)

    cells = []
    for rule in rules:
        for f in f_values:
            if f <= 0.0:
                cells.append((rule, f, "none"))
            else:
                for attack in attacks:
                    cells.append((rule, f, attack))
    print(f"\nGrid: {len(cells)} cells x {args.n_rounds} rounds each\n")

    for i, (rule, f, attack) in enumerate(cells):
        key = cell_key(rule, f, attack)
        if args.resume and key in results:
            print(f"[{i+1}/{len(cells)}] {key}: already done, skipping")
            continue
        select_rng = np.random.default_rng(args.seed * 1000 + int(round(f * 100)))
        malicious_ids = select_malicious_vehicles(vehicle_ids, f, select_rng) if f > 0.0 else []
        attack_rng = np.random.default_rng(args.seed * 7919 + i)
        attack_arg = None if attack == "none" else attack
        print(f"\n[{i+1}/{len(cells)}] {key}  (malicious: {malicious_ids or 'none'})")
        run_rounds_poisoned(
            vehicle_ids, seed_flat, static_adjacency, local_update_fn, held_out_cache,
            args.n_rounds, aggregation_rule=rule, cell_name=key,
            malicious_ids=malicious_ids, attack=attack_arg, rng=attack_rng,
            checkpoint_cb=checkpoint_cb, max_delta_norm=args.max_delta_norm,
            proximal_anchor=seed_flat, proximal_strength=args.proximal_strength,
        )

    print("\n" + "=" * 78)
    print("SUMMARY (final round, mean dv_fpr / sv_recall, per cell)")
    print("=" * 78)
    for rule in rules:
        for f in f_values:
            for attack in (["none"] if f <= 0.0 else attacks):
                key = cell_key(rule, f, attack)
                if key not in results:
                    continue
                final = results[key][-1]["summary"]
                print(f"  {key:45s} dv_fpr={final['different_vehicle_fpr']['mean']:.4f}  "
                      f"sv_recall={final['same_vehicle_recall']['mean']:.4f}")

    print(f"\nFull results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
