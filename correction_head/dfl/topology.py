"""Axis 1 -- topology generators.

T-static: idealised fully-connected graph over the fleet -- every vehicle
is everyone else's neighbour, every round. Matches
task1_datasets_pipeline_dfl.md Sec 4's `edge_list_static`.

T-traffic: realistic, sparse, dynamic graph -- two vehicles are neighbours
in round t iff they are already collaborating in round t (scenario
co-occurrence), per Sec 4: "Topology = the traffic, for T-traffic
specifically." No server, no aggregator; this is genuinely non-IID by
geography and local sensing conditions, not a synthetic sparsification of
T-static.

Both generators return a plain adjacency dict {vehicle_id: set(neighbour_ids)},
undirected (symmetric) and self-excluding, which is what gossip.py consumes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set


def build_static_topology(vehicle_ids: Sequence[str]) -> Dict[str, Set[str]]:
    """T-static: every vehicle is neighbours with every other vehicle."""
    ids = list(vehicle_ids)
    if len(ids) < 2:
        return {v: set() for v in ids}
    return {v: set(ids) - {v} for v in ids}


def build_traffic_topology(
    vehicle_ids: Sequence[str],
    scenario_of_vehicle_round: Dict[str, Dict[int, str]],
    rounds: Sequence[int],
) -> Dict[int, Dict[str, Set[str]]]:
    """T-traffic: per-round adjacency from real scenario co-occurrence.

    `scenario_of_vehicle_round[vehicle_id][round] = scenario_id` (or the key
    is absent/None if that vehicle isn't active that round -- a vehicle with
    no entry for a round has no neighbours that round, it just sits out).
    This is exactly the P6 scenario-partition + per-round presence a real
    pointed-at-P6-rows version will supply; here it is passed in directly so
    the function itself has no dependency on P6's file formats and can be
    exercised with synthetic rows.

    Returns {round: {vehicle_id: set(neighbour_ids)}} -- a fresh (possibly
    different) adjacency every round, matching the "dynamic" half of the
    T-traffic spec.
    """
    ids = list(vehicle_ids)
    per_round: Dict[int, Dict[str, Set[str]]] = {}
    for r in rounds:
        active_scenario: Dict[str, str] = {}
        for v in ids:
            sched = scenario_of_vehicle_round.get(v, {})
            if r in sched and sched[r] is not None:
                active_scenario[v] = sched[r]
        adjacency: Dict[str, Set[str]] = {v: set() for v in ids}
        # Group active vehicles by scenario; everyone sharing a scenario in
        # this round is mutually a neighbour (co-occurrence = collaboration).
        by_scenario: Dict[str, List[str]] = {}
        for v, s in active_scenario.items():
            by_scenario.setdefault(s, []).append(v)
        for members in by_scenario.values():
            for v in members:
                adjacency[v] |= set(members) - {v}
        per_round[r] = adjacency
    return per_round


def topology_stats(adjacency: Dict[str, Set[str]]) -> Dict[str, float]:
    """Quick sanity numbers for a single-round adjacency dict: used by the
    test harness and worth keeping around for the real topology-comparison
    axis's write-up (mean degree, isolated-vehicle count)."""
    n = len(adjacency)
    if n == 0:
        return {"n_vehicles": 0, "mean_degree": 0.0, "n_isolated": 0}
    degrees = [len(neigh) for neigh in adjacency.values()]
    return {
        "n_vehicles": n,
        "mean_degree": sum(degrees) / n,
        "n_isolated": sum(1 for d in degrees if d == 0),
    }


def assert_symmetric(adjacency: Dict[str, Set[str]]) -> None:
    """Both generators must produce an undirected graph -- gossip.py's
    aggregation step assumes "I am your neighbour iff you are mine"."""
    for v, neighbours in adjacency.items():
        assert v not in neighbours, f"{v} is listed as its own neighbour"
        for u in neighbours:
            assert v in adjacency.get(u, set()), (
                f"asymmetric edge: {v} -> {u} exists but {u} -> {v} does not"
            )


def build_fixed_degree_topology(
    vehicle_ids: Sequence[str], k: int, seed: int = 0
) -> Dict[str, Set[str]]:
    """DFL-E10: a synthetic "dial" between local-only (k=0) and T-static
    (k=n-1, identical to build_static_topology) -- every vehicle gets
    (approximately) exactly k random neighbours, ignoring whether those
    vehicles' underlying real rows ever actually co-occurred. This is NOT a
    realism claim about connectivity; it isolates how much the averaging
    mechanism itself helps as a pure function of how many neighbours you
    average with, independent of whether that many neighbours is realistic
    (DFL-E09 found the real T-traffic graph averages ~0.2 neighbours/round,
    far below any point this generator can usefully probe at k>=1 for
    comparison against).

    Uses a configuration-model-style random pairing with restart-on-failure
    (fine for the ~15-vehicle scale this project uses; not written for
    large n). When n*k is odd (always true when both n and k are odd --
    e.g. 15 vehicles at k=5 -- since a graph's total degree must be even by
    the handshake lemma), one random vehicle's target degree is bumped by 1
    so the degree sequence is realisable; the realised adjacency may
    therefore be "k, or k+1 for exactly one vehicle" rather than exactly k
    for everyone -- call topology_stats() on the result to see the actual
    realised mean degree, and report that, not the requested k, if they
    differ.
    """
    import random as _random

    ids = list(vehicle_ids)
    n = len(ids)
    if k <= 0:
        return {v: set() for v in ids}
    if k >= n - 1:
        return build_static_topology(ids)

    rng = _random.Random(seed)
    target_degree = {v: k for v in ids}
    if (n * k) % 2 == 1:
        target_degree[rng.choice(ids)] += 1

    max_attempts = 2000
    for _ in range(max_attempts):
        stubs: List[str] = []
        for v in ids:
            stubs.extend([v] * target_degree[v])
        rng.shuffle(stubs)
        adjacency: Dict[str, Set[str]] = {v: set() for v in ids}
        remaining = stubs[:]
        ok = True
        while remaining:
            a = remaining.pop()
            candidates = [i for i, b in enumerate(remaining) if b != a and b not in adjacency[a]]
            if not candidates:
                ok = False
                break
            idx = rng.choice(candidates)
            b = remaining.pop(idx)
            adjacency[a].add(b)
            adjacency[b].add(a)
        if ok and not remaining:
            return adjacency
    raise RuntimeError(
        f"build_fixed_degree_topology: failed to realise a degree-{k} random "
        f"graph for {n} vehicles after {max_attempts} attempts"
    )
