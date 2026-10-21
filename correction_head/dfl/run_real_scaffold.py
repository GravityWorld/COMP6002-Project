"""Real-data DFL scaffold smoke test -- points the full scaffold (topology/
aggregation/gossip/poisoning/personalisation/comms, all built and verified
synthetically in run_synthetic_scaffold.py) at the real P6 vehicle
partition and the real correction-head checkpoint, per the Oct 7 gate:
"point the full DFL scaffold at real P6 rows."

Only the pieces that must be real differ from run_synthetic_scaffold.py:
  - vehicle_ids + adjacency: from P6's real build_vehicle_partition() /
    build_edge_list_traffic(), via real_data.build_real_adjacency()
    (axis 1 -- topology).
  - local_update_fn: real torch forward/backward on that vehicle's real
    buffered samples, via real_data.make_real_local_update_fn() (the
    gossip round itself, feeding axis 2/3/5).
  - calibration mask: from the real model's own named_parameters(), via
    real_data.CALIBRATION_PARAM_NAMES (axis 4).
Aggregation rules (axis 2), poisoning (axis 3), and comms encode/decode
(axis 5) are reused completely unchanged from aggregation.py/poisoning.py/
comms.py -- they operate on flat numpy arrays regardless of what produced
them, exactly as designed.

Needs torch + the real P0/P3/P4/P5 cache + the v7 checkpoint on disk.
Run from the repo root, opencood conda env active:

    python -m correction_head.dfl.run_real_scaffold \
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt \
        --n-vehicles 15 --n-local-steps 5 --lr 0.01

Exit code is 0 iff every check below passes. This is a SMOKE TEST (one
real gossip round, a handful of real local SGD steps) -- it establishes
"the scaffold runs end-to-end on real data," not a trained DFL result.
The full five-axes depth runs (poisoning x aggregation grid, diffusion x
topology, etc., per task1_datasets_pipeline_dfl.md Sec 4) come after this
passes.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from correction_head.model import CorrectionHead, count_parameters
from correction_head.dfl import comms, poisoning, personalisation
from correction_head.dfl.aggregation import RULES
from correction_head.dfl.gossip import gossip_round
from correction_head.dfl.real_data import (
    CALIBRATION_PARAM_NAMES,
    build_real_adjacency,
    build_real_vehicle_partition,
    flatten_params,
    make_real_local_update_fn,
    named_numels,
    unflatten_into,
)

failures = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a correction_head checkpoint (e.g. "
                              "v7_bucketfix_w40_model.pt) to seed every "
                              "vehicle's starting weights. If omitted, "
                              "every vehicle starts from a fresh random init.")
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    print("=" * 78)
    print("Building real vehicle partition from P6 (this reads the real manifest")
    print("+ repair_labels, may take a few seconds)")
    print("=" * 78)
    partition = build_real_vehicle_partition(n_vehicles=args.n_vehicles)
    vehicle_ids, static_adjacency, traffic_by_round = build_real_adjacency(partition)
    n_rows_per_vehicle = partition.groupby("vehicle_id").size()
    print(f"{len(vehicle_ids)} vehicles, {len(partition)} total (scenario,frame,cav) rows, "
          f"{len(traffic_by_round)} distinct rounds for T-traffic")
    print(f"rows per vehicle: min={n_rows_per_vehicle.min()}, "
          f"max={n_rows_per_vehicle.max()}, mean={n_rows_per_vehicle.mean():.1f}")
    check("real partition produced at least 2 vehicles", len(vehicle_ids) >= 2)
    check("T-static adjacency is symmetric and fully connected",
          all(len(static_adjacency[v]) == len(vehicle_ids) - 1 for v in vehicle_ids))

    model_template = CorrectionHead()
    real_param_count = count_parameters(model_template)
    print(f"\nreal model param count: {real_param_count:,}")

    if args.checkpoint:
        print(f"loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model_template.load_state_dict(state_dict)
    seed_flat = flatten_params(model_template)
    check("seed param vector length matches real param count",
          seed_flat.shape[0] == real_param_count)

    print()
    print("=" * 78)
    print("Assigning each vehicle's real rows + building local SGD closures")
    print("=" * 78)
    vehicle_rows = {vid: group for vid, group in partition.groupby("vehicle_id")}
    local_update_fn = make_real_local_update_fn(
        vehicle_rows, n_local_steps=args.n_local_steps, lr=args.lr,
    )
    n_usable_vehicles = sum(
        1 for vid in vehicle_ids
        if not np.array_equal(
            local_update_fn(vid, seed_flat), np.zeros_like(seed_flat)
        )
    )
    check(
        "at least one vehicle has a real, non-empty cached batch to train on",
        n_usable_vehicles > 0,
        f"{n_usable_vehicles}/{len(vehicle_ids)} vehicles have usable real data "
        "(most rows have no cached BEV crop -- only 204/5826 frames were "
        "crop-cached -- this is expected, not a bug)",
    )

    print()
    print("=" * 78)
    print("ONE REAL GOSSIP ROUND -- T-static topology, average aggregation, fp32 comms")
    print("=" * 78)
    weights = {vid: seed_flat.copy() for vid in vehicle_ids}
    t_round0 = time.time()
    new_weights, diag = gossip_round(
        weights, static_adjacency, local_update_fn,
        aggregation_rule="average", comms_mode="fp32",
        max_delta_norm=50.0, rng=rng,
    )
    round_seconds = time.time() - t_round0
    any_vehicle_changed = any(
        not np.array_equal(weights[v], new_weights[v]) for v in vehicle_ids
    )
    check(
        "one real gossip round runs end-to-end without error",
        True,  # reaching this line without an exception is the check
    )
    check(
        "at least one vehicle's weights actually changed after the round",
        any_vehicle_changed,
        f"round wall-clock: {round_seconds:.2f}s for {len(vehicle_ids)} vehicles, "
        f"{args.n_local_steps} local steps each",
    )
    bytes_sent_fp32 = next(iter(diag.bytes_sent.values()))
    check(
        "real bytes-sent-per-vehicle matches the real param count at fp32",
        bytes_sent_fp32 == real_param_count * 4,
        f"{bytes_sent_fp32:,} bytes ({bytes_sent_fp32 / (1024*1024):.3f} MB)",
    )

    print()
    print("=" * 78)
    print("AXIS 3 -- poisoning, real param count, real T-static topology (cheap check only --")
    print("full f-sweep is the depth run, not this smoke test)")
    print("=" * 78)
    malicious = poisoning.select_malicious_vehicles(vehicle_ids, 0.2, rng)
    poisoned_weights, poisoned_diag = gossip_round(
        {vid: seed_flat.copy() for vid in vehicle_ids}, static_adjacency, local_update_fn,
        aggregation_rule="median", comms_mode="fp32", max_delta_norm=50.0,
        malicious_ids=malicious, attack="large_norm", rng=rng,
    )
    check(
        "gossip_round accepts real poisoning args (malicious_ids/attack) without error",
        poisoned_diag.malicious_ids == malicious,
        f"{len(malicious)}/{len(vehicle_ids)} vehicles marked malicious at f=0.2",
    )

    print()
    print("=" * 78)
    print("AXIS 4 -- personalisation, real calibration mask from the real model")
    print("=" * 78)
    numels = named_numels(model_template)
    calib_mask = personalisation.calibration_mask_from_param_shapes(
        numels, calibration_layer_names=CALIBRATION_PARAM_NAMES
    )
    check("real calibration mask size matches real param count", calib_mask.size == real_param_count)
    expected_calib_count = sum(n for name, n in numels if name in CALIBRATION_PARAM_NAMES)
    check(
        "real calibration mask marks exactly the named final-layer params",
        int(calib_mask.sum()) == expected_calib_count,
        f"{int(calib_mask.sum()):,} calibration params "
        f"({100*calib_mask.sum()/real_param_count:.2f}% of the model) -- "
        f"{CALIBRATION_PARAM_NAMES}",
    )

    print()
    print("=" * 78)
    print("AXIS 5 -- comms byte sizes at the real param count")
    print("=" * 78)
    sample_delta = seed_flat * 0.01
    for mode in ("fp32", "int8", "1bit"):
        rel_err, n_bytes = comms.reconstruction_error(sample_delta, mode)
        check(
            f"'{mode}' round-trips at the real {real_param_count:,}-param count",
            n_bytes == comms.bytes_for_mode(real_param_count, mode),
            f"{n_bytes:,} bytes ({n_bytes/(1024*1024):.3f} MB), rel. error={rel_err:.4f}",
        )

    print()
    print("=" * 78)
    total_seconds = time.time() - t0
    print(f"total wall-clock: {total_seconds:.1f}s")
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED -- DFL scaffold runs end-to-end on REAL P6 rows + the real head.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
