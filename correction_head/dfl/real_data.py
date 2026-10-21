"""Real-data wiring for the DFL scaffold -- Oct 7 gate deliverable: "point
the full DFL scaffold at real P6 rows". Replaces run_synthetic_scaffold.py's
toy local_update_fn and fake vehicle_ids/adjacency with real ones, while
reusing topology.py/aggregation.py/gossip.py/poisoning.py/comms.py/
personalisation.py completely unchanged -- per gossip.py's own docstring,
swapping local_update_fn for a real-data version is "the ONLY change
needed to move this module from synthetic rows to real ones."

Needs torch + the real P0/P3/P4/P5 cache on disk -- not runnable in the
cloud sandbox or the device-bridge's own Linux VM, same as every other
correction_head/*.py script that touches the real model. Run on the real
machine, opencood conda env. See run_real_scaffold.py for the smoke test
that actually exercises this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from correction_head.model import CorrectionHead
from correction_head.dataset import (
    DEFAULT_CROPS_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_REPAIR_LABELS_DIR,
    build_sample,
    collate_batches,
    load_scenario_crops,
    load_violation_lookup,
)

# ---------------------------------------------------------------------------
# flatten/unflatten -- the glue gossip.py's own docstring calls out as
# "not this module's job" (aggregation.py) and "needs torch" (personalisation.py).
# Order is whatever model.named_parameters() yields, which is deterministic
# for a given architecture (insertion order of submodules/params) -- the
# SAME CorrectionHead() instance (or a fresh one built the same way) must be
# used on both ends of a flatten/unflatten round-trip, which every caller
# below respects (one shared `_TEMPLATE_NAMED_NUMELS` computed once).
# ---------------------------------------------------------------------------


def named_numels(model: torch.nn.Module) -> List[Tuple[str, int]]:
    return [(n, p.numel()) for n, p in model.named_parameters()]


def flatten_params(model: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate(
            [p.detach().cpu().numpy().reshape(-1) for _, p in model.named_parameters()]
        ).astype(np.float64)


def unflatten_into(model: torch.nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            n = p.numel()
            chunk = flat[offset : offset + n]
            p.copy_(torch.from_numpy(chunk.reshape(p.shape).astype(np.float32)))
            offset += n
    assert offset == flat.shape[0], (offset, flat.shape[0])


# Calibration layer for axis 4 (personalisation) -- the real head's final
# nn.Linear, per personalisation.py's own module docstring ("the intended
# calibration layer is PairwiseAffinityHead's final nn.Linear"). mlp is
# Sequential(Linear, ReLU, Linear) -> index 2 is that final Linear.
CALIBRATION_PARAM_NAMES = ("pair_head.mlp.2.weight", "pair_head.mlp.2.bias")


# Fri 9 Oct 2026: moved local SGD onto the GPU (was silently CPU-only --
# the per-gossip-round cost this surfaced, ~155s for 15 vehicles x 5 local
# steps, matches a CPU bottleneck, not real workload size: train_v1.py's
# own GPU runs process the ENTIRE 80,999-pair training set in ~27s/epoch).
# This does NOT touch the separate, already-measured CPU-only affordability
# number (111.2ms/local-SGD-step, Oct 4 architecture gate, handover.md) --
# that number is the citable "a real car's own CPU can do this" evidence
# and stays CPU-only by design; this module's GPU use is purely about
# making the DFL depth-experiment SIMULATION (one machine standing in for
# 10-50 virtual vehicles) run fast enough to finish before the results
# freeze -- task1_datasets_pipeline_dfl.md Sec 4 itself describes this
# simulation as running "minutes on one GPU", so this is the plan, not a
# shortcut.
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Real vehicle partition -- thin wrapper over P6_features.features, kept
# here (not imported directly by run_real_scaffold.py) so this module is
# the one place that knows both P6's API and the dfl/ package's expected
# vehicle-id/adjacency shapes.
# ---------------------------------------------------------------------------


def build_real_vehicle_partition(
    manifest_path=DEFAULT_MANIFEST_PATH,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    n_vehicles: int = 15,
) -> pd.DataFrame:
    """Returns P6's build_vehicle_partition() output over every (scenario,
    frame, cav) row in the real manifest -- most of these rows will have
    no cached BEV crop (only 204/5826 frames were crop-cached, 6 per
    scenario, see P3_perception.perception_cache's --max-scenarios run);
    those rows survive the partition but build_samples_for_rows() below
    filters them out at the build_sample() stage, same as every other
    script in this repo. arm='A' (clean) throughout -- no degradation_meta
    passed, matching every other DFL-axis script's "clean benign" default."""
    from P6_features.features import build_vehicle_partition, compute_sensing_profile_from_repair_labels

    manifest = pd.read_parquet(manifest_path)
    sensing_profile = compute_sensing_profile_from_repair_labels(repair_labels_dir)
    partition = build_vehicle_partition(manifest, n_vehicles=n_vehicles, sensing_profile=sensing_profile)
    return partition


def build_real_adjacency(vehicle_partition: pd.DataFrame):
    """Returns (vehicle_ids, static_adjacency, traffic_adjacency_by_round).

    static_adjacency: Dict[vehicle_id, Set[vehicle_id]] -- T-static, via
      dfl.topology.build_static_topology (identical function used by the
      synthetic harness; only the vehicle_ids are real).
    traffic_adjacency_by_round: Dict[round_id, Dict[vehicle_id, Set]] --
      T-traffic, built DIRECTLY from P6_features.features.build_edge_list_
      traffic's real scenario-co-occurrence edge list (round_id =
      "{scenario_id}/{frame_id}", exactly as P6 defines it) rather than
      going through dfl.topology.build_traffic_topology, which expects a
      synthetic per-vehicle schedule dict -- the real data already comes
      out of P6 as a per-round edge list, so converting that directly is
      less code than reshaping it into the synthetic function's input
      shape just to reshape it back out again.
    """
    from P6_features.features import build_edge_list_traffic
    from correction_head.dfl import topology as topology_mod

    vehicle_ids = sorted(vehicle_partition["vehicle_id"].unique().tolist())
    static_adjacency = topology_mod.build_static_topology(vehicle_ids)

    traffic_edges = build_edge_list_traffic(vehicle_partition)
    traffic_by_round: Dict[str, Dict[int, set]] = {}
    for round_id, group in traffic_edges.groupby("round_id"):
        adjacency = {v: set() for v in vehicle_ids}
        for _, row in group.iterrows():
            a, b = row["vehicle_a"], row["vehicle_b"]
            adjacency[a].add(b)
            adjacency[b].add(a)
        traffic_by_round[round_id] = adjacency
    return vehicle_ids, static_adjacency, traffic_by_round


# ---------------------------------------------------------------------------
# Real per-vehicle sample assembly + local SGD -- the actual swap gossip.py
# asks for.
# ---------------------------------------------------------------------------


def build_samples_for_rows(
    rows: pd.DataFrame,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
):
    """rows: a slice of build_real_vehicle_partition()'s output (any subset
    of [scenario_id, frame_id, cav_id] rows -- typically one vehicle's own
    assigned rows). Groups by scenario_id so load_scenario_crops() (which
    loads one scenario's whole crop cache at a time) runs once per
    scenario, not once per row. Returns a list[CorrectionHeadSample],
    skipping any (frame, cav) with no cached crop (build_sample() returns
    None for those -- same convention as every build_samples_for_scenario*
    function in dataset.py)."""
    from correction_head.dataset import load_degradation_config

    degradation_config = load_degradation_config()
    samples = []
    for scenario_id, group in rows.groupby("scenario_id"):
        crop_lookup, _n_dropped = load_scenario_crops(crops_dir, scenario_id, arm)
        for _, row in group.iterrows():
            sample = build_sample(
                scenario_id,
                row["frame_id"],
                row["cav_id"],
                crop_lookup,
                repair_labels_dir,
                arm,
                degradation_config,
                sensing_profile,
                violation_lookup,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def build_samples_and_pair_weights_for_rows(
    rows: pd.DataFrame,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
    different_vehicle_weight: float = 40.0,
    other_weight: float = 1.0,
):
    """Same grouping/skip semantics as build_samples_for_rows(), but also
    computes each row's build_pair_type_weights() (dataset.py) alongside
    its build_sample() -- the exact same per-pair loss weight train_v1.py
    used to produce v7 (--different-vehicle-weight 40.0 --other-weight
    1.0). Added because the first real run_baseline_arms.py run showed
    EVERY trained arm's different-vehicle FPR getting WORSE round over
    round (3.2%->5.3% over 10 rounds, all three arms) -- the local SGD
    step was using a plain unweighted BCEWithLogitsLoss(), silently erasing
    the ratio=40 upweighting that v7's entire different-vehicle-FPR result
    depends on, on every single local step. Returns (samples, pair_weights)
    where pair_weights is a 1D np.float32 array aligned row-for-row with
    collate_batches(samples)'s pair_index (both iterate the same
    (scenario_id, frame_id, cav_id) rows in the same order and skip the
    same ones, so no offset bookkeeping is needed -- just concatenate in
    order)."""
    from correction_head.dataset import build_pair_type_weights, load_degradation_config

    degradation_config = load_degradation_config()
    samples = []
    weight_parts = []
    for scenario_id, group in rows.groupby("scenario_id"):
        crop_lookup, _n_dropped = load_scenario_crops(crops_dir, scenario_id, arm)
        for _, row in group.iterrows():
            sample = build_sample(
                scenario_id, row["frame_id"], row["cav_id"], crop_lookup,
                repair_labels_dir, arm, degradation_config, sensing_profile, violation_lookup,
            )
            if sample is None:
                continue
            w = build_pair_type_weights(
                scenario_id, row["frame_id"], row["cav_id"], crop_lookup, repair_labels_dir,
                different_vehicle_weight=different_vehicle_weight, other_weight=other_weight,
            )
            if w is None:
                continue
            samples.append(sample)
            weight_parts.append(w)
    pair_weights = np.concatenate(weight_parts, axis=0) if weight_parts else np.zeros((0,), dtype=np.float32)
    return samples, pair_weights


def make_real_local_update_fn(
    vehicle_rows: Dict[int, pd.DataFrame],
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    sensing_profile: pd.DataFrame | None = None,
    violation_lookup: dict | None = None,
    n_local_steps: int = 5,
    lr: float = 0.01,
    calib_mask: np.ndarray | None = None,
    device: torch.device = DEFAULT_DEVICE,
    different_vehicle_weight: float = 40.0,
    other_weight: float = 1.0,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Builds the real-data local_update_fn gossip_round() expects:
    (vehicle_id, flat_shared_params) -> flat_delta, same contract as the
    synthetic harness's toy version, swapped for n_local_steps of real SGD
    on that vehicle's own real buffered (crop, frag_aux, pair_index,
    pair_aux, label) batch.

    Each vehicle's batch is collated ONCE, up front (not per gossip round
    -- matches the "buffer ~200 frames" spec, a fixed local buffer a
    vehicle trains against repeatedly across rounds, not a fresh sample
    each round). A vehicle with zero usable samples (no cached crop in any
    of its assigned rows) gets a no-op update_fn that returns an all-zero
    delta -- an honest "this vehicle has nothing to learn from this round"
    rather than crashing the whole gossip round.

    calib_mask (axis 4, optional): if given, `flat_shared_params` is
    understood to be the SHARED partition only (personalisation.split_
    params's output) -- the local SGD step still updates the FULL model
    (shared + calibration together, since a real forward pass needs both),
    but the returned delta covers only the shared slice, matching what
    gossip_round() is allowed to aggregate. The calibration slice's own
    update is applied by the caller directly to that vehicle's persistent
    PersonalisedVehicleState.calib, outside gossip_round() entirely -- see
    run_real_scaffold.py's axis-4 real-data section.
    """
    cached_batches: Dict[int, tuple] = {}
    cached_pair_weights: Dict[int, torch.Tensor] = {}
    for vid, rows in vehicle_rows.items():
        samples, pair_weights = build_samples_and_pair_weights_for_rows(
            rows, crops_dir, repair_labels_dir, arm, sensing_profile, violation_lookup,
            different_vehicle_weight=different_vehicle_weight, other_weight=other_weight,
        )
        if samples:
            # moved to `device` ONCE here, not per gossip round -- the whole
            # point of caching a vehicle's batch up front.
            cached_batches[vid] = tuple(t.to(device) for t in collate_batches(samples))
            cached_pair_weights[vid] = torch.from_numpy(pair_weights).float().to(device)
        else:
            cached_batches[vid] = None
            cached_pair_weights[vid] = None

    # pos_weight computed ONCE from every vehicle's COMBINED labels, not per
    # vehicle. Fixed Sat 11 Oct, second instability round: a per-vehicle
    # pos_weight from a tiny local buffer (as few as 26 rows) is a noisy
    # estimate that can swing huge for a vehicle with almost no positive
    # labels -- combined with the already-strong 40x different-vehicle
    # weight, this gave nearly every vehicle its own badly-calibrated loss
    # scale and produced wild per-round oscillation (dv_fpr bouncing
    # 4.8%->24%->27%->13%->18% even with Adam + grad clipping). train_v1.py
    # computes pos_weight exactly once, from the WHOLE aggregate train
    # split, for exactly this reason -- one stable number shared by
    # everyone, matching that rule properly instead of per-sample-group.
    all_labels = [cached_batches[vid][4] for vid in cached_batches if cached_batches[vid] is not None]
    if all_labels:
        all_labels_cat = torch.cat(all_labels)
        n_pos = float(all_labels_cat.sum().item())
        n_neg = float(all_labels_cat.numel() - n_pos)
        global_pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    else:
        global_pos_weight = torch.tensor(1.0, device=device)

    def local_update_fn(vid: int, flat_shared: np.ndarray) -> np.ndarray:
        batch = cached_batches.get(vid)
        if batch is None or batch[4].shape[0] == 0:
            return np.zeros_like(flat_shared)

        if calib_mask is not None:
            # NOT WIRED YET: the calibration slice must come from that
            # vehicle's own PERSISTED PersonalisedVehicleState.calib (never
            # from a fresh random init, which is all a bare CorrectionHead()
            # would supply here) -- this closure only ever receives the
            # shared-partition vector, with no per-vehicle calib state
            # threaded through. Fail loudly rather than silently train
            # against a random calibration slice; wire this up (thread a
            # `vehicle_calib: Dict[vid, np.ndarray]` into this factory,
            # keyed the same as vehicle_rows) before using calib_mask here
            # for real, e.g. run_real_scaffold.py's axis-4 section currently
            # only checks the mask itself, not an actual personalised
            # real-data training loop.
            raise NotImplementedError(
                "make_real_local_update_fn's calib_mask path has no real "
                "per-vehicle calibration state to merge in yet -- see the "
                "comment just above this line before using it for real."
            )
        model = CorrectionHead().to(device)
        full_before = flat_shared.copy()
        unflatten_into(model, full_before)

        pair_weights = cached_pair_weights[vid]
        pos_weight = global_pos_weight
        # Adam, not SGD -- train_v1.py (the script that produced v7) used
        # torch.optim.Adam(lr=1e-3), never plain SGD. The first run with the
        # weighted loss (Sat 11 Oct) used SGD at lr=0.01 and produced wild,
        # divergent per-round swings (dv_fpr bouncing 19%->30%->16%->33%,
        # per-vehicle recall ranging 0%-100%) -- plain SGD has no
        # per-parameter adaptive scaling or momentum, so a vehicle with a
        # tiny local buffer (as few as 26 rows) whose batch happens to
        # contain even one 40x-upweighted different-vehicle-negative pair
        # takes one huge, unstabilized step. Switching to Adam (matching
        # train_v1.py exactly) plus a grad-norm clip as a second safety net
        # fixes this at the source instead of just lowering lr and hoping.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        crops, frag_aux, pair_index, pair_aux, labels = batch
        for _ in range(n_local_steps):
            optimizer.zero_grad()
            logits = model(crops, frag_aux, pair_index, pair_aux)
            # pos_weight for the overall pos/neg class imbalance, pair_weights
            # for the different-vehicle-negative bucket upweighting -- same
            # two-factor recipe train_v1.py used to produce v7
            # (--different-vehicle-weight 40.0 --other-weight 1.0), applied
            # here per local SGD step so DFL training doesn't quietly erase
            # the tuning that gave v7 its locked 3.68% different-vehicle FPR.
            per_pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight, reduction="none",
            )
            loss = (per_pair_loss * pair_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        full_after = flatten_params(model)
        full_delta = full_after - full_before
        if calib_mask is not None:
            return full_delta[~calib_mask]
        return full_delta

    return local_update_fn



# ---------------------------------------------------------------------------
# Shared held-out evaluation -- used by every baseline-arm / depth-run
# script so "different-vehicle FPR" means exactly the same thing here as
# it does in evaluate_pair_types.py (same classify_pair/confusion logic,
# imported directly rather than re-implemented, so the two can never
# silently drift apart).
# ---------------------------------------------------------------------------


def build_held_out_cache(
    val_scenarios,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    violation_lookup: dict | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> list:
    """Precomputes everything evaluate_on_held_out() needs that does NOT
    depend on the model: crops/frag_aux/pair_index/pair_aux tensors (moved
    to `device` once), plus the per-pair ground-truth bucket + label,
    already filtered to drop "mixed/ambiguous" pairs.

    Added after the first real run_baseline_arms.py pass showed
    evaluate_on_held_out() taking ~55s PER VEHICLE on the real held-out set
    (7 scenarios) -- with 15 vehicles re-evaluated every round across 3
    trained arms x 10 rounds, that's 15*55s*10*3 ~= 7+ hours, almost none of
    it model compute: the held-out set never changes, so reloading crops,
    rebuilding samples, and reloading ground-truth points.npz 450 times is
    pure waste. Call this ONCE per run and pass the result into
    evaluate_on_held_out_cached() for every vehicle/round -- that function
    does only a forward pass + sigmoid + bucket confusion, which should be
    a tiny fraction of 55s.
    """
    from correction_head.dataset import discover_frame_cav_pairs, load_degradation_config
    from correction_head.evaluate_pair_types import classify_pair, fragment_vehicle_ids
    from correction_head.visualize_corrections import build_sample_with_fragment_ids

    degradation_config = load_degradation_config()
    units = []
    for scenario_id in sorted(val_scenarios):
        try:
            crop_lookup, _ = load_scenario_crops(crops_dir, scenario_id, arm)
        except FileNotFoundError:
            continue
        for frame_id, cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
            built = build_sample_with_fragment_ids(
                scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, arm,
                degradation_config, violation_lookup=violation_lookup,
            )
            if built is None:
                continue
            crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order = built
            if pair_index.shape[0] == 0:
                continue
            points_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_points.npz"
            if not points_path.exists():
                continue
            d = np.load(points_path)
            vid_of_frag = fragment_vehicle_ids(fragment_ids_in_order, d["fragment_id"], d["point_vehicle_id"])

            buckets = np.array([
                classify_pair(vid_of_frag[fragment_ids_in_order[i]], vid_of_frag[fragment_ids_in_order[j]])
                for i, j in pair_index
            ])
            keep = buckets != "mixed/ambiguous"
            if not keep.any():
                continue

            units.append({
                "crops": torch.from_numpy(crops).float().to(device),
                "frag_aux": torch.from_numpy(frag_aux).float().to(device),
                "pair_index": torch.from_numpy(pair_index[keep]).long().to(device),
                "pair_aux": torch.from_numpy(pair_aux[keep]).float().to(device),
                "labels": labels[keep].astype(np.float32),
                "buckets": buckets[keep],
            })
    return units


def evaluate_on_held_out_cached(model: torch.nn.Module, cache: list, threshold: float = 0.5) -> dict:
    """Same return shape/semantics as evaluate_on_held_out(), but runs over a
    `cache` built once by build_held_out_cache() instead of rebuilding
    samples from disk on every call -- this is the hot path every
    baseline-arm/depth-run script should use once it has a cache; only use
    evaluate_on_held_out() directly for a one-off eval where building a
    cache isn't worth it."""
    from correction_head.evaluate_pair_types import confusion

    device = next(model.parameters()).device
    all_labels, all_scores, all_buckets = [], [], []
    with torch.no_grad():
        for u in cache:
            scores = model(
                u["crops"].to(device), u["frag_aux"].to(device),
                u["pair_index"].to(device), u["pair_aux"].to(device),
            ).sigmoid().cpu().numpy()
            all_scores.append(scores)
            all_labels.append(u["labels"])
            all_buckets.append(u["buckets"])

    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    all_scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    all_buckets = np.concatenate(all_buckets) if all_buckets else np.array([], dtype=object)
    preds = (all_scores > threshold).astype(np.float32)

    dv_mask = all_buckets == "different-vehicle"
    sv_mask = all_buckets == "same-vehicle"
    dv = confusion(preds[dv_mask], all_labels[dv_mask]) if dv_mask.any() else {"fpr": float("nan"), "n": 0}
    sv = confusion(preds[sv_mask], all_labels[sv_mask]) if sv_mask.any() else {"recall": float("nan"), "n": 0}
    return {
        "different_vehicle_fpr": dv["fpr"],
        "same_vehicle_recall": sv["recall"],
        "n_different_vehicle_pairs": dv["n"],
        "n_same_vehicle_pairs": sv["n"],
    }


def evaluate_on_held_out(
    model: torch.nn.Module,
    val_scenarios,
    crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR,
    arm: str = "A",
    violation_lookup: dict | None = None,
    threshold: float = 0.5,
) -> dict:
    """Runs `model` (already on its own device, already .eval()'d by the
    caller -- this function does not mutate train/eval mode) over every
    held-out (scenario, frame, cav) unit in `val_scenarios`, classifies
    every pair into evaluate_pair_types.py's four buckets using the real
    ground-truth vehicle ids, and returns a dict with the two numbers every
    baseline-arm/depth-run script reports:
        {"different_vehicle_fpr": float, "same_vehicle_recall": float,
         "n_different_vehicle_pairs": int, "n_same_vehicle_pairs": int}
    NaN for either metric if that bucket has zero pairs in this val set
    (same convention as evaluate_pair_types.py's own confusion())."""
    from correction_head.dataset import discover_frame_cav_pairs, load_degradation_config
    from correction_head.evaluate_pair_types import classify_pair, confusion, fragment_vehicle_ids
    from correction_head.visualize_corrections import build_sample_with_fragment_ids

    device = next(model.parameters()).device
    degradation_config = load_degradation_config()

    all_labels, all_scores, all_buckets = [], [], []
    for scenario_id in sorted(val_scenarios):
        try:
            crop_lookup, _ = load_scenario_crops(crops_dir, scenario_id, arm)
        except FileNotFoundError:
            continue
        for frame_id, cav_id in discover_frame_cav_pairs(repair_labels_dir, scenario_id):
            built = build_sample_with_fragment_ids(
                scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, arm,
                degradation_config, violation_lookup=violation_lookup,
            )
            if built is None:
                continue
            crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order = built
            if pair_index.shape[0] == 0:
                continue
            points_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_points.npz"
            if not points_path.exists():
                continue
            d = np.load(points_path)
            vid_of_frag = fragment_vehicle_ids(fragment_ids_in_order, d["fragment_id"], d["point_vehicle_id"])

            with torch.no_grad():
                scores = model(
                    torch.from_numpy(crops).float().to(device),
                    torch.from_numpy(frag_aux).float().to(device),
                    torch.from_numpy(pair_index).long().to(device),
                    torch.from_numpy(pair_aux).float().to(device),
                ).sigmoid().cpu().numpy()

            for (i, j), lbl, score in zip(pair_index, labels, scores):
                frag_i, frag_j = fragment_ids_in_order[i], fragment_ids_in_order[j]
                bucket = classify_pair(vid_of_frag[frag_i], vid_of_frag[frag_j])
                if bucket == "mixed/ambiguous":
                    continue
                all_labels.append(lbl)
                all_scores.append(score)
                all_buckets.append(bucket)

    all_labels = np.array(all_labels, dtype=np.float32)
    all_scores = np.array(all_scores, dtype=np.float32)
    all_buckets = np.array(all_buckets)
    preds = (all_scores > threshold).astype(np.float32)

    dv_mask = all_buckets == "different-vehicle"
    sv_mask = all_buckets == "same-vehicle"
    dv = confusion(preds[dv_mask], all_labels[dv_mask]) if dv_mask.any() else {"fpr": float("nan"), "n": 0}
    sv = confusion(preds[sv_mask], all_labels[sv_mask]) if sv_mask.any() else {"recall": float("nan"), "n": 0}
    return {
        "different_vehicle_fpr": dv["fpr"],
        "same_vehicle_recall": sv["recall"],
        "n_different_vehicle_pairs": dv["n"],
        "n_same_vehicle_pairs": sv["n"],
    }
