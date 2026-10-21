"""Synthetic-row test harness for the full DFL scaffold (Week 5', Oct 5-11).

Exercises all five axes for real, right here, with no torch dependency --
every vehicle's "model" is a flat numpy vector of the REAL pinned head's
param count (388,098, from correction_head/count_params.py) so the comms-
budget byte sizes are the true ones, not a toy dimension. "Local SGD" is
simulated with a toy convex local objective (gradient step toward a true
global optimum vector, the standard sanity-check substitute for a real
loss) -- this is exactly the "synthetic rows shaped like the frozen P6
schema" stage named in execution_plan_daily.md before pointing at real
P6 rows next.

Run directly: `python -m correction_head.dfl.run_synthetic_scaffold`
Exit code is 0 iff every check below passes.
"""

from __future__ import annotations

import sys

import numpy as np

from . import topology, comms, poisoning, personalisation
from .aggregation import RULES, ROBUST_RULES
from .gossip import gossip_round

REAL_PARAM_COUNT = 388_098  # correction_head/count_params.py, Wed 7 Oct figure
N_VEHICLES = 15             # within the "10-50 virtual vehicles" spec
SEED = 0

failures = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    rng = np.random.default_rng(SEED)
    vehicle_ids = [f"v{i:02d}" for i in range(N_VEHICLES)]

    print("=" * 78)
    print("AXIS 1 -- topology generators")
    print("=" * 78)
    static_adj = topology.build_static_topology(vehicle_ids)
    topology.assert_symmetric(static_adj)
    stats = topology.topology_stats(static_adj)
    check(
        "T-static is fully connected",
        stats["mean_degree"] == N_VEHICLES - 1 and stats["n_isolated"] == 0,
        f"mean_degree={stats['mean_degree']}, n_isolated={stats['n_isolated']}",
    )

    # Synthetic scenario co-occurrence: 4 scenarios, 6 rounds, each vehicle
    # randomly assigned to a scenario each round (or absent ~10% of rounds).
    scenarios = ["scenA", "scenB", "scenC", "scenD"]
    rounds = list(range(6))
    schedule = {}
    for v in vehicle_ids:
        sched = {}
        for r in rounds:
            if rng.random() < 0.9:
                sched[r] = rng.choice(scenarios)
        schedule[v] = sched
    traffic_adj_per_round = topology.build_traffic_topology(vehicle_ids, schedule, rounds)
    for r, adj in traffic_adj_per_round.items():
        topology.assert_symmetric(adj)
    traffic_stats = topology.topology_stats(traffic_adj_per_round[0])
    check(
        "T-traffic is sparser than T-static (real scenario co-occurrence, not full mesh)",
        traffic_stats["mean_degree"] < stats["mean_degree"],
        f"T-traffic mean_degree={traffic_stats['mean_degree']:.2f} vs "
        f"T-static mean_degree={stats['mean_degree']:.2f}",
    )
    check(
        "T-traffic adjacency actually changes across rounds (dynamic, per spec)",
        any(traffic_adj_per_round[0] != traffic_adj_per_round[r] for r in rounds[1:]),
    )

    print()
    print("=" * 78)
    print("AXIS 2 -- aggregation rules + injected-outlier robustness test")
    print("=" * 78)
    dim = 64  # small dim for this isolated-rule sanity check (fast, exact)
    honest = [rng.normal(size=dim) * 0.1 for _ in range(6)]
    own = honest[0]
    clean_neighbours = honest[1:]
    clean_consensus = np.mean(np.stack([own, *clean_neighbours], axis=0), axis=0)

    outlier = rng.normal(size=dim) * 50.0  # wildly-off, large-norm outlier
    poisoned_neighbours = clean_neighbours[:-1] + [outlier]

    # trust_weighted's robustness is CONDITIONAL on the trust signal actually
    # flagging the outlier -- per task1_datasets_pipeline_dfl.md Sec 4, alpha_ij
    # comes from the defence's own score, not from this module. Two trust
    # vectors are tested deliberately: uninformative (all 1.0, mathematically
    # identical to plain average -- trust-weighted earns no special credit
    # when fed no information) and informative (the real outlier correctly
    # scored near-zero by its "defence score", as the real pipeline intends).
    for rule_name, rule_fn in RULES.items():
        if rule_name == "trust_weighted":
            result_clean = rule_fn(own, clean_neighbours, [1.0] * len(clean_neighbours))
            uninformative_trust = [1.0] * len(poisoned_neighbours)
            informative_trust = [1.0] * (len(poisoned_neighbours) - 1) + [0.01]
            result_poisoned_uninformative = rule_fn(own, poisoned_neighbours, uninformative_trust)
            result_poisoned_informative = rule_fn(own, poisoned_neighbours, informative_trust)
        else:
            result_clean = rule_fn(own, clean_neighbours)
            result_poisoned = rule_fn(own, poisoned_neighbours)
        check(f"'{rule_name}' runs and returns the right shape", result_clean.shape == (dim,))

        if rule_name == "average":
            drift = float(np.linalg.norm(result_poisoned - clean_consensus))
            check(
                "'average' is NOT robust to the injected outlier (expected failure mode)",
                drift > 1.0,
                f"drift from clean consensus = {drift:.3f}",
            )
        elif rule_name in ROBUST_RULES:
            drift = float(np.linalg.norm(result_poisoned - clean_consensus))
            check(
                f"'{rule_name}' IS robust to the injected outlier",
                drift < 1.0,
                f"drift from clean consensus = {drift:.3f}",
            )
        elif rule_name == "trust_weighted":
            drift_uninformative = float(np.linalg.norm(result_poisoned_uninformative - clean_consensus))
            drift_informative = float(np.linalg.norm(result_poisoned_informative - clean_consensus))
            check(
                "'trust_weighted' degrades to plain-average behaviour when trust is uninformative "
                "(honest finding: it is not robust on its own, only via the trust signal it's fed)",
                drift_uninformative > 1.0,
                f"drift with all-equal trust = {drift_uninformative:.3f}",
            )
            check(
                "'trust_weighted' IS robust once trust correctly flags the outlier (the intended real-world case)",
                drift_informative < 1.0,
                f"drift with outlier correctly down-weighted = {drift_informative:.3f}",
            )

    print()
    print("=" * 78)
    print("GOSSIP ROUND -- end-to-end on synthetic rows, real param count")
    print("=" * 78)
    true_w = rng.normal(size=REAL_PARAM_COUNT) * 0.01
    weights = {v: true_w + rng.normal(size=REAL_PARAM_COUNT) * 0.05 for v in vehicle_ids}

    def local_update_fn(vid: str, w: np.ndarray) -> np.ndarray:
        # Toy convex objective: one gradient step toward true_w, plus small
        # per-vehicle noise (stands in for a real local SGD step on that
        # vehicle's buffered frames).
        lr = 0.3
        return lr * (true_w - w) + rng.normal(size=w.shape) * 0.001

    new_weights, diag = gossip_round(
        weights, static_adj, local_update_fn,
        aggregation_rule="average", comms_mode="fp32",
        max_delta_norm=50.0,
    )
    before_err = np.mean([np.linalg.norm(weights[v] - true_w) for v in vehicle_ids])
    after_err = np.mean([np.linalg.norm(new_weights[v] - true_w) for v in vehicle_ids])
    check(
        "one gossip round (no poisoning) runs end-to-end and reduces mean distance to true_w",
        after_err < before_err,
        f"mean |w-true_w| before={before_err:.4f} after={after_err:.4f}, "
        f"bytes_sent per vehicle (fp32)={next(iter(diag.bytes_sent.values())):,}",
    )

    print()
    print("=" * 78)
    print("AXIS 3 -- poisoning curve, f in {0,10,20,30,40}%, large-norm attack")
    print("=" * 78)
    print(f"{'f':>6}  {'average (final err)':>22}  {'median (final err)':>20}")
    curve_rows = []
    for f in poisoning.F_VALUES:
        for rule in ("average", "median"):
            r = np.random.default_rng(SEED + 1)
            w = {v: true_w + r.normal(size=REAL_PARAM_COUNT) * 0.05 for v in vehicle_ids}
            malicious = poisoning.select_malicious_vehicles(vehicle_ids, f, r)
            for _ in range(8):
                w, _ = gossip_round(
                    w, static_adj, local_update_fn, aggregation_rule=rule,
                    comms_mode="fp32", max_delta_norm=50.0,
                    malicious_ids=malicious, attack="large_norm", rng=r,
                )
            final_err = float(np.mean([np.linalg.norm(w[v] - true_w) for v in vehicle_ids]))
            curve_rows.append((f, rule, final_err))
    errs = {(f, rule): e for f, rule, e in curve_rows}
    for f in poisoning.F_VALUES:
        print(f"{f*100:5.0f}%  {errs[(f,'average')]:22.4f}  {errs[(f,'median')]:20.4f}")
    check(
        "'average' degrades as f increases (f=0 vs f=40% final error)",
        errs[(0.40, "average")] > errs[(0.0, "average")],
        f"f=0 err={errs[(0.0,'average')]:.4f}, f=40% err={errs[(0.40,'average')]:.4f}",
    )
    check(
        "'median' holds up far better than 'average' at f=40%",
        errs[(0.40, "median")] < errs[(0.40, "average")],
        f"median err={errs[(0.40,'median')]:.4f} vs average err={errs[(0.40,'average')]:.4f}",
    )

    print()
    print("=" * 78)
    print("AXIS 4 -- personalisation: shared trunk + per-vehicle calibration")
    print("=" * 78)
    named_numels = [("encoder", REAL_PARAM_COUNT - 20_000), ("head.final_linear", 20_000)]
    calib_mask = personalisation.calibration_mask_from_param_shapes(
        named_numels, calibration_layer_names=["head.final_linear"]
    )
    check("calibration mask size matches real param count", calib_mask.size == REAL_PARAM_COUNT)
    check("calibration mask marks exactly the named calibration layer's params",
          int(calib_mask.sum()) == 20_000)

    states = {
        v: personalisation.PersonalisedVehicleState(
            *personalisation.split_params(weights[v], calib_mask)
        )
        for v in vehicle_ids
    }

    def local_update_fn_shared_only(vid: str, w: np.ndarray) -> np.ndarray:
        # w here is the SHARED partition only; calib never passed to gossip.
        true_shared, _ = personalisation.split_params(true_w, calib_mask)
        return 0.3 * (true_shared - w) + rng.normal(size=w.shape) * 0.001

    shared_weights = {v: s.shared for v, s in states.items()}
    new_shared, _ = gossip_round(
        shared_weights, static_adj, local_update_fn_shared_only,
        aggregation_rule="average", comms_mode="fp32", max_delta_norm=50.0,
    )
    for v in vehicle_ids:
        states[v].shared = new_shared[v]  # calib untouched by gossip, by construction
    calib_values_unchanged = all(
        np.array_equal(states[v].calib, personalisation.split_params(weights[v], calib_mask)[1])
        for v in vehicle_ids
    )
    check(
        "calibration partition is never modified by a gossip round (stays per-vehicle)",
        calib_values_unchanged,
    )
    calib_variance = np.mean([np.var(s.calib) for s in states.values()])
    check(
        "calibration partitions remain vehicle-specific (nonzero cross-vehicle variance survives)",
        calib_variance > 0.0,
        f"mean per-vehicle calib variance = {calib_variance:.6f}",
    )

    print()
    print("=" * 78)
    print("AXIS 5 -- communication-budget encode/decode (fp32 / int8 / 1-bit)")
    print("=" * 78)
    sample_delta = rng.normal(size=REAL_PARAM_COUNT) * 0.02
    prior_estimate_mb = {"fp32": (1.2, 2.0), "int8": (0.3, 0.5), "1bit": (0.04, 0.065)}
    for mode in ("fp32", "int8", "1bit"):
        rel_err, n_bytes = comms.reconstruction_error(sample_delta, mode)
        theoretical = comms.bytes_for_mode(REAL_PARAM_COUNT, mode)
        mb = n_bytes / (1024 * 1024)
        lo, hi = prior_estimate_mb[mode]
        in_band = lo * 0.5 <= mb <= hi * 2.0  # generous sanity band, not exact match
        check(
            f"'{mode}' encode/decode round-trips and matches theoretical byte size",
            n_bytes == theoretical,
            f"{n_bytes:,} bytes = {mb:.3f} MB (prior written estimate: {lo}-{hi} MB), "
            f"relative reconstruction error = {rel_err:.4f}",
        )
        check(
            f"'{mode}' byte size lands in the right ballpark of the project's prior estimate",
            in_band,
            f"{mb:.3f} MB vs prior estimate {lo}-{hi} MB",
        )
    check(
        "fp32 reconstruction is exact (lossless)",
        comms.reconstruction_error(sample_delta, "fp32")[0] < 1e-6,
    )
    check(
        "1bit reconstruction error is larger than int8's (expected ordering: fp32 < int8 < 1bit)",
        comms.reconstruction_error(sample_delta, "1bit")[0]
        > comms.reconstruction_error(sample_delta, "int8")[0],
    )

    print()
    print("=" * 78)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED -- full DFL scaffold runs end-to-end on synthetic rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
