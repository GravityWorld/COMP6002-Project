"""DFL-E14: axis 4, personalisation -- shared-only weights vs shared trunk +
per-vehicle local calibration layer.

task1_datasets_pipeline_dfl.md Sec 4: "shared-only weights vs. shared trunk
+ per-vehicle local calibration layer (a vehicle's own dominant sensing-
range/density profile is exactly the kind of local structure this axis is
meant to exploit)." The calibration layer is PairwiseAffinityHead's final
nn.Linear (real_data.CALIBRATION_PARAM_NAMES), per personalisation.py's own
docstring -- everything upstream (the conv/GroupNorm feature extractor,
the first PairwiseAffinityHead linear layer) stays shared and federated;
only that final decision-boundary layer is held per-vehicle and never
gossiped.

Two arms, same topology (T-static -- the "ceiling" condition, same choice
E11/E12 made, to isolate the personalisation question from the real-
topology sparsity DFL-E09 already reported), same checkpoint, same
n_rounds, same held-out split -- run in the same script so the comparison
is apples-to-apples rather than cross-referencing a separate run:

  - shared_only : the ordinary real-data local_update_fn (real_data.
                   make_real_local_update_fn, UNCHANGED) + gossip_round over
                   the FULL parameter vector. Numbers should land close to
                   DFL-E08's centralised_fedavg arm (same setup), run fresh
                   here for a same-script, same-seed comparison point.
  - personalised : gossip_round over the SHARED slice only (calib_mask ==
                   False); each vehicle keeps its own persisted calibration
                   vector (calib_mask == True) which is trained alongside
                   the shared slice every local step (a real forward pass
                   needs the full model) but whose resulting delta is
                   applied directly to that vehicle's own state and NEVER
                   returned to gossip_round -- so it is never averaged with
                   any other vehicle's calibration, round after round.

No change to gossip.py / topology.py / aggregation.py / personalisation.py
-- personalisation.py's split_params/merge_params/calibration_mask_from_
param_shapes are used exactly as designed (previously only exercised
against a synthetic test harness and a real-parameter-count smoke check in
run_real_scaffold.py, never an actual training loop -- this script is that
wiring). The one new function is make_personalised_local_update_fn below,
a copy of real_data.make_real_local_update_fn's training recipe (Adam,
grad-clip, pair-type + global pos_weight loss -- identical hyperparameters)
with the merge-before-forward / split-after-backward step added, because
real_data.py's own calib_mask branch is deliberately left raising
NotImplementedError (no per-vehicle calib state was ever threaded through
it -- see that function's docstring).

Needs torch + the real P0/P3/P4/P5/P6 cache on disk. Run on the real
machine, opencood conda env, from the repo root:

    python -m correction_head.dfl.run_personalisation_axis4_e14 \
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt \
        --n-vehicles 15 --n-rounds 10 --out results\DFL-E14-personalisation.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from correction_head.dataset import make_scenario_splits
from correction_head.model import CorrectionHead
from correction_head.dfl.gossip import gossip_round
from correction_head.dfl import personalisation
from correction_head.dfl.real_data import (
    named_numels,
    flatten_params,
    unflatten_into,
    CALIBRATION_PARAM_NAMES,
    build_real_vehicle_partition,
    build_real_adjacency,
    build_held_out_cache,
    evaluate_on_held_out_cached,
    build_samples_and_pair_weights_for_rows,
    make_real_local_update_fn,
    collate_batches,
    DEFAULT_DEVICE,
    DEFAULT_CROPS_DIR,
    DEFAULT_REPAIR_LABELS_DIR,
)


def make_personalised_local_update_fn(
    vehicle_rows, vehicle_calib: dict, calib_mask: np.ndarray,
    crops_dir=DEFAULT_CROPS_DIR, repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A", sensing_profile=None, violation_lookup=None,
    n_local_steps: int = 5, lr: float = 1e-3, device=DEFAULT_DEVICE,
    different_vehicle_weight: float = 40.0, other_weight: float = 1.0,
):
    """Trains the FULL model (that vehicle's shared slice merged with its
    own persisted calibration slice) for n_local_steps, then splits the
    resulting delta: returns the SHARED part only (what gossip_round is
    allowed to aggregate), and applies the calibration part directly to
    `vehicle_calib[vid]` in place -- that update never leaves this
    function, never gets aggregated with any other vehicle's."""
    cached_batches: dict = {}
    cached_pair_weights: dict = {}
    for vid, rows in vehicle_rows.items():
        samples, pair_weights = build_samples_and_pair_weights_for_rows(
            rows, crops_dir, repair_labels_dir, arm, sensing_profile, violation_lookup,
            different_vehicle_weight=different_vehicle_weight, other_weight=other_weight,
        )
        if samples:
            cached_batches[vid] = tuple(t.to(device) for t in collate_batches(samples))
            cached_pair_weights[vid] = torch.from_numpy(pair_weights).float().to(device)
        else:
            cached_batches[vid] = None
            cached_pair_weights[vid] = None

    all_labels = [cached_batches[vid][4] for vid in cached_batches if cached_batches[vid] is not None]
    if all_labels:
        all_labels_cat = torch.cat(all_labels)
        n_pos = float(all_labels_cat.sum().item())
        n_neg = float(all_labels_cat.numel() - n_pos)
        global_pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    else:
        global_pos_weight = torch.tensor(1.0, device=device)

    def local_update_fn(vid, flat_shared: np.ndarray) -> np.ndarray:
        batch = cached_batches.get(vid)
        if batch is None or batch[4].shape[0] == 0:
            return np.zeros_like(flat_shared)

        full_before = personalisation.merge_params(flat_shared, vehicle_calib[vid], calib_mask)
        model = CorrectionHead().to(device)
        unflatten_into(model, full_before)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        crops, frag_aux, pair_index, pair_aux, labels = batch
        pair_weights = cached_pair_weights[vid]
        for _ in range(n_local_steps):
            optimizer.zero_grad()
            logits = model(crops, frag_aux, pair_index, pair_aux)
            per_pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=global_pos_weight, reduction="none",
            )
            loss = (per_pair_loss * pair_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        full_after = flatten_params(model)
        full_delta = full_after - full_before
        shared_delta, calib_delta = personalisation.split_params(full_delta, calib_mask)
        vehicle_calib[vid] = vehicle_calib[vid] + calib_delta
        return shared_delta

    return local_update_fn


def evaluate_all_vehicles(full_weights: dict, held_out_cache: list, device=DEFAULT_DEVICE) -> dict:
    out = {}
    for vid, flat in full_weights.items():
        model = CorrectionHead().to(device)
        unflatten_into(model, flat)
        model.eval()
        out[vid] = evaluate_on_held_out_cached(model, held_out_cache)
        del model
    return out


def summarize(per_vehicle_metrics: dict) -> dict:
    fprs = np.array([m["different_vehicle_fpr"] for m in per_vehicle_metrics.values()], dtype=np.float64)
    recalls = np.array([m["same_vehicle_recall"] for m in per_vehicle_metrics.values()], dtype=np.float64)
    fprs = fprs[~np.isnan(fprs)]
    recalls = recalls[~np.isnan(recalls)]

    def stats(arr):
        if arr.size == 0:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
        return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

    return {"different_vehicle_fpr": stats(fprs), "same_vehicle_recall": stats(recalls)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--arms", type=str, default="shared_only,personalised")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                         help="Skip an arm already fully complete (n_rounds+1 history "
                              "entries) in --out. Does not resume a partially completed "
                              "arm -- reruns it from round 0 (same coarse granularity as "
                              "DFL-E13; this script is two arms, not a 27-cell grid).")
    args = parser.parse_args()
    selected_arms = {a.strip() for a in args.arms.split(',') if a.strip()}

    torch.manual_seed(args.seed)

    split = make_scenario_splits(val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed)
    print(f"Held-out val scenarios ({len(split.val_scenarios)}): {sorted(split.val_scenarios)}")
    if not split.val_scenarios:
        print("ABORTING: val split is empty.")
        return 1

    print("Building held-out eval cache once...")
    t0 = time.time()
    held_out_cache = build_held_out_cache(split.val_scenarios)
    print(f"held-out cache built in {time.time()-t0:.1f}s, {len(held_out_cache)} units")

    print("=" * 78)
    print("Building real vehicle partition + T-static adjacency from P6")
    print("=" * 78)
    partition = build_real_vehicle_partition(n_vehicles=args.n_vehicles)
    vehicle_ids, static_adjacency, _traffic_by_round = build_real_adjacency(partition)
    print(f"{len(vehicle_ids)} vehicles (T-static, fully connected, for this axis)")

    vehicle_rows = {vid: group for vid, group in partition.groupby("vehicle_id")}

    model_template = CorrectionHead()
    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_template.load_state_dict(state_dict)
    seed_flat = flatten_params(model_template)

    numels = named_numels(model_template)
    calib_mask = personalisation.calibration_mask_from_param_shapes(
        numels, calibration_layer_names=CALIBRATION_PARAM_NAMES
    )
    n_calib = int(calib_mask.sum())
    print(f"calibration layer: {CALIBRATION_PARAM_NAMES} -- {n_calib:,} / {calib_mask.size:,} "
          f"params ({100*n_calib/calib_mask.size:.2f}%) held per-vehicle, never federated")

    diff_veh_w, other_w = 40.0, 1.0

    results = {}
    if args.resume and args.out and Path(args.out).exists():
        with open(args.out) as f:
            prior = json.load(f)
        for name, history in prior.get("results", {}).items():
            if len(history) == args.n_rounds + 1:
                results[name] = history
        if results:
            print(f"--resume: loaded already-complete arms: {list(results.keys())}")

    def checkpoint_write():
        if not args.out:
            return
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(args.out).with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"calibration_param_names": list(CALIBRATION_PARAM_NAMES),
                        "n_calib_params": n_calib, "n_total_params": int(calib_mask.size),
                        "results": results}, f, indent=2)
        tmp.replace(args.out)

    if "shared_only" in selected_arms and "shared_only" not in results:
        print()
        print("=" * 78)
        print("ARM 1/2: SHARED_ONLY -- full model federated, T-static, average aggregation")
        print("=" * 78)
        local_update_fn = make_real_local_update_fn(
            vehicle_rows, n_local_steps=args.n_local_steps, lr=args.lr,
            different_vehicle_weight=diff_veh_w, other_weight=other_w,
        )
        weights = {vid: seed_flat.copy() for vid in vehicle_ids}
        history = [{"round": 0, "summary": summarize(evaluate_all_vehicles(weights, held_out_cache))}]
        for r in range(args.n_rounds):
            t0 = time.time()
            weights, _diag = gossip_round(weights, static_adjacency, local_update_fn, aggregation_rule="average")
            summary = summarize(evaluate_all_vehicles(weights, held_out_cache))
            elapsed = time.time() - t0
            print(f"[shared_only] round {r+1}/{args.n_rounds} ({elapsed:.1f}s) "
                  f"dv_fpr mean={summary['different_vehicle_fpr']['mean']:.4f} "
                  f"sv_recall mean={summary['same_vehicle_recall']['mean']:.4f}")
            history.append({"round": r + 1, "summary": summary})
            results["shared_only"] = history
            checkpoint_write()
    elif "shared_only" in results:
        print("\n[shared_only] already complete, skipping\n")

    if "personalised" in selected_arms and "personalised" not in results:
        print()
        print("=" * 78)
        print("ARM 2/2: PERSONALISED -- shared trunk federated, calibration layer per-vehicle local")
        print("=" * 78)
        shared_seed, calib_seed = personalisation.split_params(seed_flat, calib_mask)
        vehicle_calib = {vid: calib_seed.copy() for vid in vehicle_ids}
        local_update_fn = make_personalised_local_update_fn(
            vehicle_rows, vehicle_calib, calib_mask,
            n_local_steps=args.n_local_steps, lr=args.lr,
            different_vehicle_weight=diff_veh_w, other_weight=other_w,
        )
        shared_weights = {vid: shared_seed.copy() for vid in vehicle_ids}

        def full_weights():
            return {vid: personalisation.merge_params(shared_weights[vid], vehicle_calib[vid], calib_mask)
                    for vid in vehicle_ids}

        history = [{"round": 0, "summary": summarize(evaluate_all_vehicles(full_weights(), held_out_cache))}]
        for r in range(args.n_rounds):
            t0 = time.time()
            shared_weights, _diag = gossip_round(
                shared_weights, static_adjacency, local_update_fn, aggregation_rule="average",
            )
            summary = summarize(evaluate_all_vehicles(full_weights(), held_out_cache))
            elapsed = time.time() - t0
            print(f"[personalised] round {r+1}/{args.n_rounds} ({elapsed:.1f}s) "
                  f"dv_fpr mean={summary['different_vehicle_fpr']['mean']:.4f} "
                  f"sv_recall mean={summary['same_vehicle_recall']['mean']:.4f}")
            history.append({"round": r + 1, "summary": summary})
            results["personalised"] = history
            checkpoint_write()
    elif "personalised" in results:
        print("\n[personalised] already complete, skipping\n")

    print()
    print("=" * 78)
    print("SUMMARY (final round, mean across vehicles)")
    print("=" * 78)
    for arm_name, history in results.items():
        final = history[-1]["summary"]
        print(f"  {arm_name:14s} dv_fpr={final['different_vehicle_fpr']['mean']:.4f}  "
              f"sv_recall={final['same_vehicle_recall']['mean']:.4f}")

    checkpoint_write()
    if args.out:
        print(f"\nFull results written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
