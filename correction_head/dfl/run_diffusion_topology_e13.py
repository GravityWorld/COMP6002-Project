"""DFL-E13: the diffusion x topology centrepiece.

Seeds ONE vehicle with dense local experience of its own clustering-failure
regime -- picked via P6's real range/density `profile_bucket` (the vehicle
whose own rows are most dominated by the highest-range / lowest-density
bucket, i.e. the regime the Thu 24 Sep audit found GROUND_SWALLOWED
concentrates in: mean range 46.7m vs CLEAN's 26.5m). Every other vehicle
starts naive -- the unmodified frozen checkpoint, zero extra steps, exactly
like the floor arm in DFL-E08. Runs gossip for `--n-rounds` under T-static
and, separately, T-traffic, tracking each vehicle's REGIME-SPECIFIC
same_vehicle_recall each round (the metric that most directly reflects
"repair competence" here -- same-vehicle recall is literally whether a
correction correctly re-links fragments/points that belong to one real
vehicle, which is what fixes GROUND_SWALLOWED/MERGED/SPLIT), then reports:

  - hop distance from the seed vehicle under each topology. T-static: every
    vehicle is 1 hop from everyone (fully connected) -- trivial by
    construction, included for completeness/contrast against T-traffic
    rather than as an interesting number on its own. T-traffic: BFS over
    the UNION of every real per-round adjacency graph observed across all
    distinct traffic rounds (an edge exists if two vehicles were EVER
    neighbours in ANY observed round) -- a fixed graph, since per-round
    adjacency alone is too sparse for a stable per-vehicle hop notion
    (DFL-E09: mean degree 0.265, 83.6% of vehicle-rounds fully isolated). A
    vehicle unreachable from the seed in this union graph gets hop=None.
  - rounds-to-80%-transfer per vehicle: the first round at which that
    vehicle's own gain over its own round-0 regime recall reaches 80% of
    the seed's own total gain (round-0 -> final round), or None if it never
    does within `--n-rounds`. This is the number the write-up plots against
    hop distance -- "do the cars actually teach each other."

No new code in gossip.py / topology.py / aggregation.py -- this reuses
gossip_round() and both topology generators completely unchanged, same as
every other DFL-axis script. The four genuinely new pieces are: (a) the
regime-tagged held-out eval (build_regime_tagged_held_out_cache below,
real_data.build_held_out_cache's own loop plus one profile_bucket lookup
per unit -- real_data.py itself is untouched), (b) picking the seed vehicle
from the real partition's own profile_bucket column, (c) the warm-start
step (one extra local_update_fn call with many more local steps, applied to
the seed vehicle only, before round 1), and (d) the union-graph BFS
hop-distance + rounds-to-80%-transfer computation.

Needs torch + the real P0/P3/P4/P5/P6 cache on disk -- not runnable in the
cloud sandbox or the device-bridge's own Linux VM, same as every other
correction_head/*.py script that touches the real model. Run on the real
machine, opencood conda env, from the repo root:

    python -m correction_head.dfl.run_diffusion_topology_e13 \
        --checkpoint correction_head\checkpoints\v7_bucketfix_w40_model.pt \
        --n-vehicles 15 --n-rounds 10 --out results\DFL-E13-diffusion-topology.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from correction_head.dataset import (
    make_scenario_splits,
    DEFAULT_CROPS_DIR,
    DEFAULT_REPAIR_LABELS_DIR,
    discover_frame_cav_pairs,
    load_degradation_config,
    load_scenario_crops,
)
from correction_head.model import CorrectionHead
from correction_head.evaluate_pair_types import classify_pair, confusion, fragment_vehicle_ids
from correction_head.visualize_corrections import build_sample_with_fragment_ids
from correction_head.dfl.gossip import gossip_round
from correction_head.dfl.real_data import (
    flatten_params,
    unflatten_into,
    make_real_local_update_fn,
    build_real_vehicle_partition,
    build_real_adjacency,
    DEFAULT_DEVICE,
)


def build_regime_tagged_held_out_cache(
    val_scenarios, sensing_profile, crops_dir=DEFAULT_CROPS_DIR,
    repair_labels_dir=DEFAULT_REPAIR_LABELS_DIR, arm: str = "A",
    violation_lookup=None, device=DEFAULT_DEVICE,
) -> list:
    """Same loop/semantics as real_data.build_held_out_cache -- rebuilt here
    rather than imported because the ONE thing it's missing for this
    experiment is tagging each unit with its own (scenario, frame, cav)'s
    profile_bucket, so a regime-specific subset can be sliced out
    afterward. real_data.py itself is left completely unchanged."""
    profile_lookup = {
        (r.scenario_id, r.frame_id, r.cav_id): r.profile_bucket
        for r in sensing_profile.itertuples()
    }
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
                "profile_bucket": profile_lookup.get((scenario_id, frame_id, cav_id), "unknown"),
            })
    return units


def evaluate_cache_subset(model, cache_subset: list) -> dict:
    """Same confusion-bucket logic as real_data.evaluate_on_held_out_cached,
    inlined so this script has no dependency on being called with the FULL
    held-out cache -- cache_subset may be any slice."""
    if not cache_subset:
        return {"different_vehicle_fpr": float("nan"), "same_vehicle_recall": float("nan"),
                "n_different_vehicle_pairs": 0, "n_same_vehicle_pairs": 0}
    device = next(model.parameters()).device
    all_labels, all_scores, all_buckets = [], [], []
    with torch.no_grad():
        for u in cache_subset:
            scores = model(
                u["crops"].to(device), u["frag_aux"].to(device),
                u["pair_index"].to(device), u["pair_aux"].to(device),
            ).sigmoid().cpu().numpy()
            all_scores.append(scores)
            all_labels.append(u["labels"])
            all_buckets.append(u["buckets"])
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_buckets = np.concatenate(all_buckets)
    preds = (all_scores > 0.5).astype(np.float32)
    dv_mask = all_buckets == "different-vehicle"
    sv_mask = all_buckets == "same-vehicle"
    dv = confusion(preds[dv_mask], all_labels[dv_mask]) if dv_mask.any() else {"fpr": float("nan"), "n": 0}
    sv = confusion(preds[sv_mask], all_labels[sv_mask]) if sv_mask.any() else {"recall": float("nan"), "n": 0}
    return {"different_vehicle_fpr": dv["fpr"], "same_vehicle_recall": sv["recall"],
            "n_different_vehicle_pairs": dv["n"], "n_same_vehicle_pairs": sv["n"]}


def weights_to_model(flat: np.ndarray, device=DEFAULT_DEVICE) -> CorrectionHead:
    m = CorrectionHead().to(device)
    unflatten_into(m, flat)
    m.eval()
    return m


def pick_seed_vehicle(partition):
    """Picks the vehicle_id whose rows are most dominated by the
    highest-range / lowest-density profile_bucket seen in this partition --
    the regime the Thu 24 Sep audit found GROUND_SWALLOWED concentrates in.
    Works for whatever n_buckets compute_sensing_profile_from_repair_labels
    used (reads the actual max/min bucket indices present, doesn't hardcode
    3). Falls back to "most rows in the max-range bucket, ignoring density"
    if no vehicle has any row in the exact target bucket (possible given
    the project's own documented 2-qualifying-scenario limitation)."""
    buckets = partition["profile_bucket"]
    known = buckets[buckets.str.startswith("range")]
    if known.empty:
        raise RuntimeError(
            "no rows have a real profile_bucket (all 'unknown') -- cannot pick "
            "a regime-representative seed vehicle. Check that "
            "P4_defence/cache/repair_labels/ has real _points.npz output."
        )
    parsed = known.str.extract(r"range(\d+)_density(\d+)").astype(int)
    max_r = int(parsed[0].max())
    min_d = int(parsed[1].min())
    target = f"range{max_r}_density{min_d}"
    counts = partition.loc[partition["profile_bucket"] == target, "vehicle_id"].value_counts()
    if counts.empty:
        extracted = partition["profile_bucket"].str.extract(r"range(\d+)_density(\d+)")
        relaxed_mask = extracted[0].astype(float) == max_r
        counts = partition.loc[relaxed_mask.fillna(False), "vehicle_id"].value_counts()
        print(f"WARNING: no vehicle has any row in {target} exactly -- falling back to "
              f"max-range-only ({max_r}) for seed selection.")
    seed_vid = counts.index[0]
    return seed_vid, target


def union_graph_hops(vehicle_ids, traffic_by_round, seed_vid):
    """BFS hop distance from seed_vid over the UNION of every real per-round
    adjacency graph in traffic_by_round. Returns {vehicle_id: hop_or_None}."""
    union_adj = {v: set() for v in vehicle_ids}
    for adjacency in traffic_by_round.values():
        for v, neigh in adjacency.items():
            union_adj.setdefault(v, set()).update(neigh)
    hops = {v: None for v in vehicle_ids}
    hops[seed_vid] = 0
    q = deque([seed_vid])
    while q:
        cur = q.popleft()
        for nxt in union_adj.get(cur, set()):
            if hops.get(nxt) is None and nxt != seed_vid:
                hops[nxt] = hops[cur] + 1
                q.append(nxt)
    return hops


def rounds_to_80pct(trajectory, baseline, seed_final_gain):
    """trajectory: this vehicle's regime same_vehicle_recall for rounds
    1..n (round 0 excluded). Returns the 1-indexed round at which
    (trajectory[i] - baseline) first reaches 0.8 * seed_final_gain, or None
    if it never does, or if seed_final_gain <= 0 (seed itself didn't
    improve -- "80% of nothing" is undefined, not zero)."""
    if seed_final_gain is None or seed_final_gain <= 0 or np.isnan(seed_final_gain):
        return None
    for i, val in enumerate(trajectory):
        if np.isnan(val):
            continue
        if (val - baseline) >= 0.8 * seed_final_gain:
            return i + 1
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-vehicles", type=int, default=15)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-local-steps", type=int, default=5)
    parser.add_argument("--seed-warm-steps", type=int, default=50,
                         help="Local SGD steps given ONLY to the seed vehicle before "
                              "round 1, on its own regime-heavy buffered data -- this "
                              "is the 'dense experience' half of the spec. Every other "
                              "vehicle gets zero extra steps (starts at the unmodified "
                              "frozen checkpoint, exactly like DFL-E08's floor arm).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--topologies", type=str, default="static,traffic")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                         help="If --out already has a FULLY complete topology run in it, "
                              "skip re-running that topology. Does not resume a partially "
                              "completed topology (reruns it from round 0) -- this "
                              "experiment is short enough (~2 topologies x n_rounds, no "
                              "27-cell grid like E11) that this coarse granularity is fine.")
    args = parser.parse_args()
    selected_topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]

    torch.manual_seed(args.seed)

    split = make_scenario_splits(val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed)
    print(f"Held-out val scenarios ({len(split.val_scenarios)}): {sorted(split.val_scenarios)}")
    if not split.val_scenarios:
        print("ABORTING: val split is empty.")
        return 1

    from P6_features.features import compute_sensing_profile_from_repair_labels
    sensing_profile = compute_sensing_profile_from_repair_labels(DEFAULT_REPAIR_LABELS_DIR)
    print(f"sensing profile computed over {len(sensing_profile)} real (scenario,frame,cav) rows")

    print("Building regime-tagged held-out eval cache once...")
    t0 = time.time()
    cache = build_regime_tagged_held_out_cache(split.val_scenarios, sensing_profile)
    print(f"cache built in {time.time()-t0:.1f}s, {len(cache)} units")

    print("=" * 78)
    print("Building real vehicle partition + adjacency from P6")
    print("=" * 78)
    partition = build_real_vehicle_partition(n_vehicles=args.n_vehicles)
    vehicle_ids, static_adjacency, traffic_by_round = build_real_adjacency(partition)
    print(f"{len(vehicle_ids)} vehicles, {len(traffic_by_round)} distinct real traffic rounds")

    seed_vid, target_bucket = pick_seed_vehicle(partition)
    print(f"Seed vehicle: {seed_vid}  (target regime bucket: {target_bucket})")

    regime_cache = [u for u in cache if u["profile_bucket"] == target_bucket]
    print(f"Regime-specific held-out subset: {len(regime_cache)}/{len(cache)} units match {target_bucket}")
    if len(regime_cache) < 5:
        print("WARNING: regime-specific held-out subset is very small ({}) -- treat its "
              "per-round numbers as indicative, not a trustworthy curve (same small-eval-"
              "set caveat already flagged for DFL-E10).".format(len(regime_cache)))

    model_template = CorrectionHead()
    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_template.load_state_dict(state_dict)
    seed_flat = flatten_params(model_template)

    vehicle_rows = {vid: group for vid, group in partition.groupby("vehicle_id")}
    diff_veh_w, other_w = 40.0, 1.0
    local_update_fn = make_real_local_update_fn(
        vehicle_rows, n_local_steps=args.n_local_steps, lr=args.lr,
        different_vehicle_weight=diff_veh_w, other_weight=other_w,
    )

    print(f"\nWarm-starting seed vehicle {seed_vid} with {args.seed_warm_steps} local SGD "
          f"steps on its own regime-heavy buffer -- every other vehicle stays at the "
          f"unmodified frozen checkpoint until round 1...")
    warm_update_fn = make_real_local_update_fn(
        {seed_vid: vehicle_rows[seed_vid]}, n_local_steps=args.seed_warm_steps, lr=args.lr,
        different_vehicle_weight=diff_veh_w, other_weight=other_w,
    )
    seed_warm_delta = warm_update_fn(seed_vid, seed_flat)
    seed_warm_flat = seed_flat + seed_warm_delta
    print("warm-start done\n")

    def eval_all(weights):
        out = {}
        for vid, flat in weights.items():
            model = weights_to_model(flat)
            overall = evaluate_cache_subset(model, cache)
            regime = evaluate_cache_subset(model, regime_cache)
            del model
            out[vid] = {"overall": overall, "regime": regime}
        return out

    results = {}
    if args.resume and args.out and Path(args.out).exists():
        with open(args.out) as f:
            prior = json.load(f)
        for name, history in prior.get("results", {}).items():
            if len(history) == args.n_rounds + 1:
                results[name] = history
        if results:
            print(f"--resume: loaded already-complete topologies: {list(results.keys())}")

    def run_topology(name, adjacency_fn):
        if name in results:
            print(f"[{name}] already complete, skipping")
            return results[name]
        weights = {vid: (seed_warm_flat.copy() if vid == seed_vid else seed_flat.copy()) for vid in vehicle_ids}
        history = [{"round": 0, "metrics": eval_all(weights)}]
        for r in range(args.n_rounds):
            t0 = time.time()
            adjacency = adjacency_fn(r)
            weights, _diag = gossip_round(weights, adjacency, local_update_fn, aggregation_rule="average")
            metrics = eval_all(weights)
            elapsed = time.time() - t0
            seed_regime_recall = metrics[seed_vid]["regime"]["same_vehicle_recall"]
            print(f"[{name}] round {r+1}/{args.n_rounds} ({elapsed:.1f}s) "
                  f"seed regime sv_recall={seed_regime_recall:.4f}")
            history.append({"round": r + 1, "metrics": metrics})
            results[name] = history
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(args.out).with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump({"seed_vehicle": str(seed_vid), "target_bucket": target_bucket,
                                "results": results}, f, indent=2)
                tmp.replace(args.out)
        return history

    if "static" in selected_topologies:
        print("\n" + "=" * 78 + "\nTOPOLOGY: T-STATIC\n" + "=" * 78)
        run_topology("static", lambda r: static_adjacency)

    if "traffic" in selected_topologies:
        print("\n" + "=" * 78 + "\nTOPOLOGY: T-TRAFFIC\n" + "=" * 78)
        traffic_rounds_list = sorted(traffic_by_round.keys())
        if not traffic_rounds_list:
            print("WARNING: no real traffic rounds -- falling back to T-static for this arm too.")
            run_topology("traffic", lambda r: static_adjacency)
        else:
            def traffic_adjacency_fn(r):
                key = traffic_rounds_list[r % len(traffic_rounds_list)]
                adj = traffic_by_round[key]
                for vid in vehicle_ids:
                    adj.setdefault(vid, set())
                return adj
            run_topology("traffic", traffic_adjacency_fn)

    static_hops = {v: (0 if v == seed_vid else 1) for v in vehicle_ids}
    traffic_hops = union_graph_hops(vehicle_ids, traffic_by_round, seed_vid)

    summary = {}
    for name, hops in (("static", static_hops), ("traffic", traffic_hops)):
        if name not in results:
            continue
        history = results[name]
        baseline = {v: history[0]["metrics"][str(v) if str(v) in history[0]["metrics"] else v]["regime"]["same_vehicle_recall"] for v in vehicle_ids}
        final_seed = history[-1]["metrics"][seed_vid]["regime"]["same_vehicle_recall"]
        seed_final_gain = final_seed - baseline[seed_vid]
        per_vehicle = {}
        for v in vehicle_ids:
            trajectory = [h["metrics"][v]["regime"]["same_vehicle_recall"] for h in history[1:]]
            r80 = rounds_to_80pct(trajectory, baseline[v], seed_final_gain)
            per_vehicle[v] = {
                "hop_distance": hops.get(v),
                "rounds_to_80pct_transfer": r80,
                "baseline_regime_recall": baseline[v],
                "final_regime_recall": trajectory[-1] if trajectory else None,
            }
        summary[name] = {"seed_final_gain": seed_final_gain, "per_vehicle": per_vehicle}

    print("\n" + "=" * 78)
    print("SUMMARY: hop distance vs rounds-to-80%-transfer")
    print("=" * 78)
    for name, s in summary.items():
        print(f"\n-- {name} (seed_final_gain={s['seed_final_gain']:.4f}) --")
        for v, d in sorted(s["per_vehicle"].items(), key=lambda kv: (kv[1]["hop_distance"] is None, kv[1]["hop_distance"] or 0)):
            fr = d["final_regime_recall"]
            fr_str = f"{fr:.4f}" if fr is not None and not np.isnan(fr) else "nan"
            print(f"  vehicle {v}: hop={d['hop_distance']}  rounds_to_80pct={d['rounds_to_80pct_transfer']}  "
                  f"baseline={d['baseline_regime_recall']:.4f}  final={fr_str}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed_vehicle": str(seed_vid), "target_bucket": target_bucket,
                        "results": results, "summary": summary}, f, indent=2)
        print(f"\nFull results written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
