"""
P6 -- Feature assembly + splits. Implemented 17 Sep 2026 (Week 2, compressed
into one sitting -- originally Mon 14 / Tue 15 Sep, the GATE day).

Per scope_and_timeline_oct19.md / task1_datasets_pipeline_dfl.md, this stage
emits three artefacts, all of which every one of Part 3's five DFL axes reads
afterward -- schema correctness here matters more than almost anything else
in the pipeline. If a field needs to be added after today, add it as a NEW
OPTIONAL column with a default, never change an existing one.

**Real deviation from the original schema, flagged explicitly:** the plan
specifies partitioning by town x degradation_type x severity. This OPV2V
extract's yaml has no `town` field at all (confirmed during P0 -- see
decisions.md, Thu 17 Sep entry) -- `index.py`'s manifest already writes
`town=None` for every row. `scenario_id` is used as the town proxy instead
(each scenario is, in practice, one fixed location/layout, which is what
`town` would have captured anyway) -- documented here as a substitution, not
a silent guess.

**Second real limitation, flagged explicitly:** the manifest currently has
only 2 qualifying scenarios (`2021_09_03_20_08_37`, `2021_09_03_22_43_57`),
not the dozens a realistic non-IID fleet partition assumes. The functions
below are written to work correctly for any number of scenarios, but with
only 2, a genuine train/val/test split by scenario is close to degenerate
(you cannot hold out a scenario for test without either losing a whole
"town" from training or from testing). This is carried forward as an open
risk, not silently smoothed over -- more qualifying scenarios extracted from
`culver_data` (or unlocking `n_cav >= 3` from a larger OPV2V pull) would fix
this properly; documenting it now is what the schema-freeze discipline asks
for instead of a retrofit in October.

**Thu 1 Oct 2026 addition -- the local range/density partition key.**
`task1_datasets_pipeline_dfl.md` §2 specifies partitioning by "scenario +
local range/density profile + degradation type/severity" -- only the
scenario+degradation half was implemented on 17 Sep; the range/density half
was left as an open TODO (flagged repeatedly in decisions.md, Fri 25 Sep
through Wed 30 Sep). `compute_sensing_profile_from_repair_labels()` below
closes that gap by deriving each (scenario, frame, cav)'s own local
range/density profile directly from the REAL data
`run_cluster_repair_labelling.py` already wrote to
`P4_defence/cache/repair_labels/` -- the same real per-point arrays behind
the 996-instance audit and the two visualisation scripts, not a fresh
guess. `build_vehicle_partition()` now takes this as an optional
`sensing_profile` argument and folds a `profile_bucket` column into the
partition key alongside `(scenario_id, degradation_type, severity)`.

This changes `partition_key`'s string format (an extra `|<profile_bucket>`
suffix, `|unknown` when no profile is supplied) rather than adding a purely
new column -- normally "never change an existing column" per this module's
own freeze discipline above, but nothing downstream has consumed this
schema yet (no DFL axis has run against real data as of Thu 1 Oct 2026 --
see decisions.md), so there is no already-run result this could silently
invalidate. Existing callers that don't pass `sensing_profile` are
unaffected in relative behaviour (same-key-same-vehicle, distinct-key-very-
likely-distinct-vehicle) -- no existing test hardcodes an exact vehicle_id
integer, only these relative properties.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _explode_manifest_to_cav_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario_id, frame_id, cav_id) instead of one row per
    frame with a cav_ids list -- every function below operates at CAV
    granularity, since that's what gets partitioned into virtual vehicles."""
    rows = manifest.explode("cav_ids").rename(columns={"cav_ids": "cav_id"})
    rows["cav_id"] = rows["cav_id"].astype(str)
    return rows.reset_index(drop=True)


def _stable_bucket(key: str, n_buckets: int) -> int:
    """Deterministic hash -> bucket index, stable across runs/machines
    (unlike Python's built-in hash(), which is salted per-process) -- this
    is what makes vehicle_id assignment reproducible given the same
    (scenario, degradation type, severity) partition key."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


def compute_sensing_profile_from_repair_labels(
    labels_dir: str | Path,
    n_buckets: int = 3,
) -> pd.DataFrame:
    """
    Derives each (scenario_id, frame_id, cav_id)'s own local range/density
    profile directly from run_cluster_repair_labelling.py's real per-point
    output (`<labels_dir>/<scenario>/<frame>_<cav>_points.npz`) -- the same
    real data behind the 996-instance audit and both visualisation scripts,
    not a synthetic or hand-picked proxy.

    Per row, computes two summary statistics over that (frame, cav)'s
    non-ground point cloud (`xyz`, in the CAV's own local lidar frame):
      - mean_range_m: mean radial distance of its points from the CAV's own
        sensor origin. The audit's own headline finding (Thu 24 Sep 2026)
        is that GROUND_SWALLOWED is range-correlated (mean range 46.7m vs
        26.5m for CLEAN) -- a CAV whose typical points sit further out is
        exactly the one whose local scenes skew toward that failure mode.
      - clustered_fraction: fraction of non-ground points DBSCAN actually
        assigned to a real cluster (raw_cluster_id >= 0), i.e. 1 minus the
        noise fraction. This stands in for local point density without
        recomputing DBSCAN's own eps-neighbourhood logic a second time --
        a sparser local scene produces more noise under the same eps/
        min_samples, which is the audit's other headline mechanism (a real
        vehicle can have >=10 points and still end up entirely noise if
        those points aren't densely packed enough -- see
        explain_clustering_failure_mechanisms.py panel 2).

    Each statistic is independently bucketed into `n_buckets` quantile bins
    (pd.qcut, duplicates='drop' so a degenerate all-identical column doesn't
    crash) and combined into one categorical `profile_bucket` string, e.g.
    "range2_density0", for build_vehicle_partition to fold into its
    partition key.

    Returns a DataFrame with columns:
        [scenario_id, frame_id, cav_id, mean_range_m, clustered_fraction,
         profile_bucket]

    Rows for a (scenario, frame, cav) whose _points.npz is missing or has
    zero non-ground points are skipped (nothing to compute a profile from);
    build_vehicle_partition's own left-join + fillna('unknown') handles the
    resulting gaps.
    """
    labels_dir = Path(labels_dir)
    records = []
    for npz_path in sorted(labels_dir.glob("*/*_points.npz")):
        scenario_id = npz_path.parent.name
        stem = npz_path.name[: -len("_points.npz")]
        # stem is "{frame_id}_{cav_id}" per run_cluster_repair_labelling.py's
        # own naming -- frame_id is always OPV2V's fixed-width zero-padded
        # numeric timestamp (e.g. "003572"), so splitting on the first
        # underscore is unambiguous and doesn't depend on cav_id never
        # containing one.
        frame_id, cav_id = stem.split("_", 1)

        data = np.load(npz_path)
        xyz = data["xyz"]
        raw_cluster_id = data["raw_cluster_id"]
        if xyz.shape[0] == 0:
            continue

        mean_range_m = float(np.linalg.norm(xyz[:, :2], axis=1).mean())
        clustered_fraction = float(np.mean(raw_cluster_id >= 0))
        records.append(
            {
                "scenario_id": scenario_id,
                "frame_id": frame_id,
                "cav_id": cav_id,
                "mean_range_m": mean_range_m,
                "clustered_fraction": clustered_fraction,
            }
        )

    profile = pd.DataFrame(
        records,
        columns=["scenario_id", "frame_id", "cav_id", "mean_range_m", "clustered_fraction"],
    )
    if profile.empty:
        profile["profile_bucket"] = pd.Series(dtype=str)
        return profile

    range_bucket = pd.qcut(profile["mean_range_m"], q=n_buckets, labels=False, duplicates="drop")
    density_bucket = pd.qcut(profile["clustered_fraction"], q=n_buckets, labels=False, duplicates="drop")
    profile["profile_bucket"] = (
        "range" + range_bucket.astype(str) + "_density" + density_bucket.astype(str)
    )
    return profile


def build_vehicle_partition(
    manifest: pd.DataFrame,
    n_vehicles: int,
    degradation_meta: pd.DataFrame | None = None,
    sensing_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Partition (scenario, frame, cav) rows into `n_vehicles` virtual vehicles,
    by town(=scenario_id proxy) x local range/density profile x
    degradation_type x severity -- per the plan, this partition IS the
    non-IID split every DFL axis trains against, not a convenience.

    degradation_meta: optional DataFrame with columns
        [scenario_id, cav_id, degradation_type, severity] (from P2's config,
        keyed the same way P2's seeded RNG is keyed -- (scenario, type,
        severity)). If not supplied, degradation_type='none', severity=0.

    sensing_profile: optional DataFrame as returned by
        compute_sensing_profile_from_repair_labels() -- must have a
        `profile_bucket` column plus enough of [scenario_id, frame_id,
        cav_id] to join on. Joined at (scenario_id, frame_id, cav_id)
        granularity when `frame_id` is present in both sides, else at
        (scenario_id, cav_id) granularity (for a caller that pre-aggregates
        one profile per CAV across its frames). Rows with no match get
        profile_bucket='unknown' rather than being dropped -- a CAV with no
        repair-labelling output yet (e.g. outside the 2 currently-qualifying
        scenarios) still gets partitioned, just without this axis
        discriminating it from other 'unknown' rows. If not supplied at
        all, every row gets profile_bucket='unknown' (this axis becomes a
        no-op, matching this function's pre-Thu-1-Oct behaviour).

    Returns a DataFrame with columns:
        [scenario_id, frame_id, cav_id, degradation_type, severity,
         profile_bucket, partition_key, vehicle_id]
    """
    if n_vehicles < 1:
        raise ValueError("n_vehicles must be >= 1")

    rows = _explode_manifest_to_cav_rows(manifest)[["scenario_id", "frame_id", "cav_id"]]

    if degradation_meta is not None:
        rows = rows.merge(
            degradation_meta, on=["scenario_id", "cav_id"], how="left", validate="many_to_one"
        )
        rows["degradation_type"] = rows["degradation_type"].fillna("none")
        rows["severity"] = rows["severity"].fillna(0).astype(int)
    else:
        rows["degradation_type"] = "none"
        rows["severity"] = 0

    if sensing_profile is not None and len(sensing_profile):
        join_cols = ["scenario_id", "frame_id", "cav_id"] if "frame_id" in sensing_profile.columns else [
            "scenario_id",
            "cav_id",
        ]
        profile_slim = sensing_profile[join_cols + ["profile_bucket"]].drop_duplicates(join_cols)
        rows = rows.merge(profile_slim, on=join_cols, how="left")
    else:
        rows["profile_bucket"] = None
    rows["profile_bucket"] = rows["profile_bucket"].fillna("unknown")

    rows["partition_key"] = (
        rows["scenario_id"].astype(str)
        + "|"
        + rows["degradation_type"].astype(str)
        + "|"
        + rows["severity"].astype(str)
        + "|"
        + rows["profile_bucket"].astype(str)
    )
    rows["vehicle_id"] = rows["partition_key"].map(lambda k: _stable_bucket(k, n_vehicles))
    return rows


def build_edge_list_static(vehicle_ids) -> pd.DataFrame:
    """
    T-static topology: fully connected. Every distinct pair of vehicle ids
    is an edge (undirected -- one row per unordered pair, per the plan's
    "idealised fully-connected graph" description).
    """
    unique_ids = sorted(set(vehicle_ids))
    edges = [
        (a, b) for i, a in enumerate(unique_ids) for b in unique_ids[i + 1 :]
    ]
    return pd.DataFrame(edges, columns=["vehicle_a", "vehicle_b"])


def build_edge_list_traffic(vehicle_partition: pd.DataFrame) -> pd.DataFrame:
    """
    T-traffic topology: derived from scenario co-occurrence. Two vehicles
    are neighbours in round t iff they're collaborating in round t -- here,
    "round t" = one (scenario_id, frame_id) -- the CAVs present in the same
    frame of the same scenario are, by construction, the ones actually
    collaborating in OPV2V's real capture. This is sparse and dynamic by
    construction (unlike T-static): a vehicle's neighbour set can differ
    frame to frame if its underlying CAV rotates across scenarios, since
    the partition (build_vehicle_partition) maps CAVs to virtual vehicles
    by scenario+degradation key, not by physical CAV identity alone.

    vehicle_partition: the DataFrame returned by build_vehicle_partition
        (needs scenario_id, frame_id, vehicle_id columns).

    Returns a DataFrame with columns [round_id, vehicle_a, vehicle_b], one
    row per co-occurring unordered pair per round. `round_id` is
    "{scenario_id}/{frame_id}" so it's directly traceable back to source
    data -- useful for the diffusion-experiment axis later, which needs to
    know hop distance through actual rounds, not just a static graph.
    """
    required = {"scenario_id", "frame_id", "vehicle_id"}
    missing = required - set(vehicle_partition.columns)
    if missing:
        raise ValueError(f"vehicle_partition missing required columns: {missing}")

    edges = []
    grouped = vehicle_partition.groupby(["scenario_id", "frame_id"])["vehicle_id"].apply(
        lambda s: sorted(set(s))
    )
    for (scenario_id, frame_id), vids in grouped.items():
        round_id = f"{scenario_id}/{frame_id}"
        for i, a in enumerate(vids):
            for b in vids[i + 1 :]:
                edges.append((round_id, a, b))

    return pd.DataFrame(edges, columns=["round_id", "vehicle_a", "vehicle_b"])


@dataclass
class ScenarioSplit:
    train_scenarios: set = field(default_factory=set)
    val_scenarios: set = field(default_factory=set)
    test_scenarios: set = field(default_factory=set)

    def split_for(self, scenario_id: str) -> str:
        if scenario_id in self.train_scenarios:
            return "train"
        if scenario_id in self.val_scenarios:
            return "val"
        if scenario_id in self.test_scenarios:
            return "test"
        raise KeyError(f"scenario_id {scenario_id!r} not in any split")


def build_scenario_split(
    manifest: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0
) -> ScenarioSplit:
    """
    Split by scenario (and, transitively, town via the scenario_id proxy --
    see module docstring), never by frame -- frame-level splits leak
    adjacent frames across the split and inflate every downstream number
    (execution_plan_daily.md, Mon 14 Sep).

    **Known degenerate case, current data:** with only 2 qualifying
    scenarios, a 3-way split forces at least one split to be empty. This
    function does NOT silently paper over that -- it will produce an empty
    val or test set rather than fabricate a fake split, and the caller
    (or a downstream test) should treat an empty split as a real signal
    that more scenarios are needed before this axis is meaningful, not
    ignore it.
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in [0,1) and sum to < 1")

    scenario_ids = sorted(manifest["scenario_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = list(scenario_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))

    val = set(shuffled[:n_val])
    test = set(shuffled[n_val : n_val + n_test])
    train = set(shuffled[n_val + n_test :])

    return ScenarioSplit(train_scenarios=train, val_scenarios=val, test_scenarios=test)
