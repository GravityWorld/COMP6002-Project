"""DFL-E09: how many neighbours does a vehicle actually average with per
round under the real traffic-based adjacency (dfl arm), vs the static
fully-connected graph (centralised_fedavg arm)?

Deliberately avoids importing correction_head.dfl.real_data (which pulls in
torch at module level) -- this only needs the same two building blocks it
wraps: P6_features.features.build_vehicle_partition/build_edge_list_traffic
and correction_head.dfl.topology.build_static_topology. Replicates
build_real_vehicle_partition / build_real_adjacency's logic exactly (see
real_data.py lines 100-152) so the numbers match what run_baseline_arms.py
actually used for DFL-E08.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from P6_features.features import build_vehicle_partition, build_edge_list_traffic, compute_sensing_profile_from_repair_labels
from correction_head.dfl import topology as topology_mod

DEFAULT_REPAIR_LABELS_DIR = REPO_ROOT / "P4_defence" / "cache" / "repair_labels"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "P0_scenario_indexing" / "cache" / "manifest.parquet"

N_VEHICLES = 15

manifest = pd.read_parquet(DEFAULT_MANIFEST_PATH)
sensing_profile = compute_sensing_profile_from_repair_labels(DEFAULT_REPAIR_LABELS_DIR)
partition = build_vehicle_partition(manifest, n_vehicles=N_VEHICLES, sensing_profile=sensing_profile)

vehicle_ids = sorted(partition["vehicle_id"].unique().tolist())
static_adjacency = topology_mod.build_static_topology(vehicle_ids)

traffic_edges = build_edge_list_traffic(partition)
traffic_by_round = {}
for round_id, group in traffic_edges.groupby("round_id"):
    adjacency = {v: set() for v in vehicle_ids}
    for _, row in group.iterrows():
        a, b = row["vehicle_a"], row["vehicle_b"]
        adjacency[a].add(b)
        adjacency[b].add(a)
    traffic_by_round[round_id] = adjacency

static_degrees = [len(neigh) for neigh in static_adjacency.values()]
print(f"vehicles: {len(vehicle_ids)}")
print(f"static adjacency (centralised_fedavg): degree per vehicle = {static_degrees} -> mean={np.mean(static_degrees):.2f}")

print(f"\ndistinct real traffic rounds available: {len(traffic_by_round)}")

all_degrees = []
for round_id, adjacency in traffic_by_round.items():
    degs = [len(neigh) for neigh in adjacency.values()]
    all_degrees.extend(degs)

all_degrees = np.array(all_degrees)
print(f"across ALL real rounds x all {N_VEHICLES} vehicles ({len(all_degrees)} (vehicle,round) pairs):")
print(f"  mean degree = {all_degrees.mean():.3f}")
print(f"  median degree = {np.median(all_degrees):.1f}")
print(f"  pct with 0 neighbours (fully isolated that round) = {100*(all_degrees==0).mean():.1f}%")
print(f"  pct with >=1 neighbour = {100*(all_degrees>0).mean():.1f}%")
print(f"  max degree seen = {all_degrees.max()}")
vals, counts = np.unique(all_degrees, return_counts=True)
print(f"  distribution (degree: count): {dict(zip(vals.tolist(), counts.tolist()))}")

traffic_rounds_list = sorted(traffic_by_round.keys())
n_rounds_e08 = 10
used_degrees = []
print(f"\nthe exact 10 rounds DFL-E08 cycled through (modulo {len(traffic_rounds_list)}):")
for r in range(n_rounds_e08):
    round_key = traffic_rounds_list[r % len(traffic_rounds_list)]
    adjacency = traffic_by_round[round_key]
    degs = [len(adjacency.get(v, set())) for v in vehicle_ids]
    used_degrees.extend(degs)
    print(f"  round {r+1}/10 -> real round_id={round_key!r}: mean degree={np.mean(degs):.2f}, per-vehicle={degs}")

used_degrees = np.array(used_degrees)
print(f"\nmean degree over the exact 10 rounds E08 used: {used_degrees.mean():.3f} (vs static's {np.mean(static_degrees):.1f})")
