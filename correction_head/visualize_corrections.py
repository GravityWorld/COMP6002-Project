"""Before/after gallery for the correction head -- Sat 4 Oct 2026, post-gate.

Shows, per real (scenario, frame, cav) unit, three panels of the SAME
real point cloud:

  1. BEFORE  -- raw DBSCAN fragments (`fragment_id`), the actual candidate
     granularity P4's clustering pass produced and the only thing the
     correction head sees as "separate objects" before it does anything.
  2. AFTER (head) -- the trained head's own prediction: every candidate
     pair in `pairs.parquet` is scored by the real checkpoint, pairs
     scored above --threshold are unioned (connected components), and
     points are coloured by their fragment's resulting merged-group id.
  3. GROUND TRUTH -- `point_vehicle_id`, P5's oracle real-object identity
     for every point. This is the target a perfect repair would produce;
     it is NOT what the head is told at inference time, only what it was
     trained against.

Nothing here is synthetic: same `_points.npz` / `_fragments.parquet` /
`_pairs.parquet` real files correction_head/dataset.py and
P4_defence/visualize_repair_examples.py already read, and the same
CorrectionHead architecture / forward pass train_v1.py trained for the
Sun 4 Oct gate result. The only new code is the union-find grouping and
the plotting -- no model or data-loading logic is duplicated with a
different definition; the aux-feature construction below mirrors
dataset.py's build_sample() line for line (same filtering order) so the
per-fragment index used for plotting lines up exactly with what the
checkpoint was scored on.

REQUIRES a trained checkpoint. train_v1.py did not save one before this
script was written (Sat 4 Oct 2026) -- it only wrote CSV/JSON summaries.
Re-run it once to get `correction_head/runs/v1_holdout_model.pt`:

    python -m correction_head.train_v1

Then run this, from the repo root, same opencood conda env (needs torch,
matplotlib, shapely -- not runnable in the cloud sandbox or the
device-bridge VM, same as every other correction_head/*.py script):

    python -m correction_head.visualize_corrections \
        --checkpoint correction_head/runs/v1_holdout_model.pt \
        --scenarios 2021_08_22_07_52_02 2021_08_24_11_37_54 \
        --out correction_head/runs/before_after_gallery.png

Added Sun 5 Oct 2026: a fourth panel, DENOISE BASELINE -- the existing
"just merge nearby fragments" method (denoise_baseline.py's own threshold,
read from its summary JSON) scored on the SAME fragments/pairs as the
head, so the figure now shows the actual contribution this project is
making: raw fragmentation -> what the pre-existing naive distance rule
would do -> what the learned head does -> the real answer. Without this
panel the gallery only showed the head in isolation, which doesn't
demonstrate improvement over anything -- the quantitative CLEAN-rate/FPR
comparison already existed in decisions.md, this makes it visible.

With no --scenarios given, defaults to the checkpoint's own held-out VAL
scenarios (recorded inside the .pt file) -- the fair, honest choice: those
are scenarios the head never trained on, same as the headline gate number.
Pass --scenarios explicitly (including a train scenario) if you want to
see a scenario the head DID train on for contrast -- the gallery's title
for each row says which split that scenario was in, so it's never
ambiguous which kind of result you're looking at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPoint

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_head import dataset as dataset_module
from correction_head.dataset import (
    DEFAULT_CROPS_DIR,
    DEFAULT_REPAIR_LABELS_DIR,
    discover_frame_cav_pairs,
    load_degradation_config,
    load_scenario_crops,
)
from correction_head.model import CorrectionHead

BACKGROUND_VEHICLE_ID = -1  # same convention as P4_defence/visualize_repair_examples.py


class UnionFind:
    """Plain union-find over 0..n-1, no deps. Used to turn the head's
    thresholded pairwise predictions into merged fragment groups -- this
    IS the "lightweight greedy/graph post-process" model.py's own
    docstring says sits outside the model (never built until now)."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


def build_sample_with_fragment_ids(
    scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, arm, degradation_config,
    violation_lookup=None,
):
    """Same filtering/ordering as dataset.build_sample(), but additionally
    returns the real fragment_id for every row in order -- build_sample()
    itself doesn't carry that through, and this script needs it to map
    model outputs back onto real points for plotting.

    Thu 8 Oct 2026 fix: this used to always zero-fill free_occ_conflict_area/
    tracking_affinity (frag_aux indices 6/7) and always build a 2-column
    pair_aux, regardless of what the checkpoint being scored was actually
    trained with -- a real bug, caught today: every evaluate_pair_types.py
    run against a --violation-lookup-trained checkpoint (both the raw-scale
    and log1p occviol runs) silently scored the model against the WRONG
    frag_aux inputs, and a checkpoint trained with the new interposed-object
    pair feature (added the same day) crashes here outright (pair_aux shape
    mismatch) rather than scoring wrong, which is how this was caught.
    violation_lookup takes the same dict dataset.py's build_sample()/
    load_violation_lookup() use (None -> zero-fill, same as before this
    fix, so an old call site still behaves identically). pair_aux is now
    always 3 columns (dist_norm, abs_delta_log_n, interposed_object_signal),
    matching dataset.py's build_sample() exactly -- see its
    _pairwise_interposed_signal() for what the third column is."""
    frag_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_fragments.parquet"
    pair_path = Path(repair_labels_dir) / scenario_id / f"{frame_id}_{cav_id}_pairs.parquet"
    if not frag_path.exists() or not pair_path.exists():
        return None

    frags = pd.read_parquet(frag_path)
    pairs = pd.read_parquet(pair_path)

    has_crop = frags["fragment_id"].apply(lambda fid: (str(frame_id), str(cav_id), int(fid)) in crop_lookup)
    frags = frags[has_crop].reset_index(drop=True)  # mirrors build_sample() exactly
    if frags.empty:
        return None

    kept_ids = set(frags["fragment_id"].tolist())
    pairs = pairs[pairs["fragment_i"].isin(kept_ids) & pairs["fragment_j"].isin(kept_ids)].reset_index(drop=True)

    from correction_head.dataset import (
        AUX_FEATURE_NAMES,
        CANDIDATE_PAIR_RADIUS_M,
        _LIDAR_MAX_RANGE_M,
        _degradation_severity,
        _fragment_extent_norm,
        _pairwise_interposed_signal,
    )

    local_index = {fid: i for i, fid in enumerate(frags["fragment_id"].tolist())}
    n = len(frags)
    crops = np.stack(
        [crop_lookup[(str(frame_id), str(cav_id), int(fid))] for fid in frags["fragment_id"]], axis=0
    ).astype(np.float32)

    deg_severity = _degradation_severity(arm, scenario_id, cav_id, degradation_config)
    centroids = frags[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    range_from_ego_norm = np.clip(np.linalg.norm(centroids, axis=1) / _LIDAR_MAX_RANGE_M, 0.0, 1.0)
    bearing_norm = np.arctan2(centroids[:, 1], centroids[:, 0]) / np.pi
    density = 0.5  # same "unknown" fallback dataset.py uses when no sensing_profile is passed

    occ_violation = 0.0
    perc_violation = 0.0
    if violation_lookup is not None:
        v = violation_lookup.get((scenario_id, frame_id))
        if v is not None:
            raw_occ_violation, raw_perc_violation = v
            occ_violation = float(np.log1p(raw_occ_violation))
            perc_violation = float(np.log1p(raw_perc_violation))

    frag_extent_norm = _fragment_extent_norm(
        scenario_id, frame_id, cav_id, repair_labels_dir, frags["fragment_id"].tolist()
    )

    frag_aux = np.stack(
        [
            np.log1p(frags["n_points"].to_numpy(dtype=np.float32)),
            frags["is_recovered_noise"].to_numpy(dtype=np.float32),
            range_from_ego_norm.astype(np.float32),
            bearing_norm.astype(np.float32),
            np.full(n, density, dtype=np.float32),
            np.full(n, deg_severity, dtype=np.float32),
            np.full(n, occ_violation, dtype=np.float32),  # free_occ_conflict_area -- real if violation_lookup given
            np.full(n, perc_violation, dtype=np.float32),  # tracking_affinity -- real if violation_lookup given
            frag_extent_norm,  # frag_extent_norm -- computed for real, same as dataset.py build_sample()
        ],
        axis=1,
    ).astype(np.float32)
    assert frag_aux.shape[1] == len(AUX_FEATURE_NAMES)

    if pairs.empty:
        pair_index = np.zeros((0, 2), dtype=np.int64)
        pair_aux = np.zeros((0, 3), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.float32)
    else:
        n_points_by_frag = dict(zip(frags["fragment_id"], frags["n_points"]))
        i_local = pairs["fragment_i"].map(local_index).to_numpy(dtype=np.int64)
        j_local = pairs["fragment_j"].map(local_index).to_numpy(dtype=np.int64)
        pair_index = np.stack([i_local, j_local], axis=1)
        dist_norm = pairs["centroid_distance_m"].to_numpy(dtype=np.float32) / np.float32(CANDIDATE_PAIR_RADIUS_M)
        log_n_i = np.log1p(pairs["fragment_i"].map(n_points_by_frag).to_numpy(dtype=np.float32))
        log_n_j = np.log1p(pairs["fragment_j"].map(n_points_by_frag).to_numpy(dtype=np.float32))
        centroid_by_frag = dict(zip(frags["fragment_id"].tolist(), map(tuple, centroids)))
        interposed_signal = _pairwise_interposed_signal(
            scenario_id,
            frame_id,
            cav_id,
            repair_labels_dir,
            pairs["fragment_i"].tolist(),
            pairs["fragment_j"].tolist(),
            centroid_by_frag,
        )
        pair_aux = np.stack(
            [dist_norm, np.abs(log_n_i - log_n_j), interposed_signal], axis=1
        ).astype(np.float32)
        labels = pairs["affinity_label"].to_numpy(dtype=np.float32)

    fragment_ids_in_order = frags["fragment_id"].to_numpy()
    return crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order


def hull_or_points(points_xy):
    if len(points_xy) < 3:
        return points_xy
    hull = MultiPoint(points_xy).convex_hull
    if hull.geom_type != "Polygon":
        return points_xy
    return np.array(hull.exterior.coords)


def false_merge_group_ids(group_id, point_vehicle_id):
    """A merge is FALSE, provably, when one predicted group contains
    points from >=2 DIFFERENT real vehicles (ground truth point_vehicle_id,
    background/-1 excluded) -- not a visual judgment call, a fact the
    ground truth settles exactly. A group that merges one real vehicle
    with background-only fragments is a different, softer error (merging
    something real with nothing) and is deliberately NOT flagged here --
    this stays to the strict, ground-truth-provable case."""
    bad = set()
    for gid in set(group_id[group_id >= 0].tolist()):
        vids = set(point_vehicle_id[(group_id == gid) & (point_vehicle_id != BACKGROUND_VEHICLE_ID)].tolist())
        if len(vids) >= 2:
            bad.add(gid)
    return bad


def draw_groups(ax, xy, group_id, title, caption, unscored_mask=None, false_merge_ids=None):
    """One panel: every point coloured by its group id (tab20, wrapping),
    background/ungrouped points (-1) in dimgray. Draws a convex-hull
    outline per group with >=3 points so merged/split groups are visually
    obvious at a glance, same visual language as
    P4_defence/visualize_repair_examples.py's hull_xy().

    unscored_mask, when given, marks points whose fragment belonged to a
    real DBSCAN cluster but was never scored by the head (no cached BEV
    crop -- dataset.py's own GAP #2). Drawn as a distinct orange '+' so
    they are never visually confused with (a) true background/noise
    points (dimgray 'x', genuinely not part of any cluster) or (b)
    fragments the head scored and chose not to merge (their own tab20
    colour, singleton group). Collapsing "never seen by the model" into
    "left alone" would misrepresent what the AFTER panel actually shows --
    this caught itself as a real plotting bug on the first real run
    (Sat 4 Oct 2026): 316 raw fragments in row 1 but only 202 were scored.

    false_merge_ids, when given (see false_merge_group_ids() above), marks
    groups that provably merged >=2 different real vehicles: drawn with a
    bold red dashed outline ON TOP of the normal thin coloured hull, so a
    false merge is never left to eyeballing "does this blob look too big" --
    it's the ground truth itself saying this group is wrong."""
    if unscored_mask is None:
        unscored_mask = np.zeros(len(xy), dtype=bool)
    if false_merge_ids is None:
        false_merge_ids = set()
    ungrouped = (group_id < 0) & ~unscored_mask
    ax.scatter(xy[ungrouped, 0], xy[ungrouped, 1], s=8, c="dimgray", marker="x",
               linewidths=0.7, zorder=1, label="_nolegend_")
    if unscored_mask.any():
        ax.scatter(xy[unscored_mask, 0], xy[unscored_mask, 1], s=14, c="darkorange", marker="+",
                   linewidths=1.0, zorder=1, label="_nolegend_")
    grouped_ids = sorted(set(group_id[(~ungrouped) & (~unscored_mask) & (group_id >= 0)].tolist()))
    cmap = plt.get_cmap("tab20")
    for i, gid in enumerate(grouped_ids):
        mask = (group_id == gid) & (~unscored_mask)
        if not mask.any():
            continue
        color = cmap(i % 20)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, color=color, zorder=2, label="_nolegend_")
        if mask.sum() >= 3:
            hull = hull_or_points(xy[mask])
            ax.add_patch(MplPolygon(hull, closed=True, fill=False, edgecolor=color, linewidth=1.6, zorder=2))
            if gid in false_merge_ids:
                ax.add_patch(MplPolygon(hull, closed=True, fill=False, edgecolor="red", linewidth=2.4,
                                         linestyle="--", zorder=4, label="_nolegend_"))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "v1_holdout_model.pt"))
    parser.add_argument("--scenarios", type=str, nargs="*", default=None,
                         help="Scenario ids to show. Defaults to the checkpoint's own held-out val scenarios.")
    parser.add_argument("--max-units-per-scenario", type=int, default=2,
                         help="Cap on how many (frame, cav) units to plot per scenario, so the figure stays readable.")
    parser.add_argument("--units", type=str, nargs="*", default=None,
                         help="Exact (scenario_id:frame_id:cav_id) triples to plot instead of the first "
                              "--max-units-per-scenario units per scenario. Use this to pull specific "
                              "units out of evaluate_declustering.py's --out-json per_unit list (e.g. the "
                              "ones with the highest head_false_merges), rather than an arbitrary sample. "
                              "Example: --units 2021_08_22_07_52_02:000070:641")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Sigmoid threshold above which a pair is unioned into one group (after-merge).")
    parser.add_argument("--arm", type=str, default="A")
    parser.add_argument("--crops-dir", type=str, default=str(DEFAULT_CROPS_DIR))
    parser.add_argument("--repair-labels-dir", type=str, default=str(DEFAULT_REPAIR_LABELS_DIR))
    parser.add_argument("--out", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "before_after_gallery.png"))
    parser.add_argument("--denoise-summary", type=str,
                         default=str(_REPO_ROOT / "correction_head" / "runs" / "denoise_baseline_summary.json"),
                         help="denoise_baseline.py's own output JSON -- its threshold_dist_norm_lt is read "
                              "from here so the baseline panel uses the SAME threshold already logged in "
                              "decisions.md, not a re-picked one. Overridden by --denoise-threshold if given.")
    parser.add_argument("--denoise-threshold", type=float, default=None,
                         help="Override the denoise baseline's distance threshold directly, skipping --denoise-summary.")
    parser.add_argument(
        "--violation-lookup",
        type=str,
        default=None,
        help="Thu 8 Oct 2026 fix: path to the CSV compute_violation_scalars.py writes. "
             "Pass the SAME lookup the checkpoint was trained with (train_v1.py's own "
             "--violation-lookup), or free_occ_conflict_area/tracking_affinity will be "
             "zero-filled here regardless of what the model actually learned on -- this "
             "was a real bug (build_sample_with_fragment_ids had no way to pass this at "
             "all) until today.",
    )
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run `python -m correction_head.train_v1` first "
              f"(it now saves one at that default path as of this script being added) -- "
              f"see this script's own module docstring.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CorrectionHead().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_scenarios = set(ckpt.get("train_scenarios", []))
    val_scenarios = set(ckpt.get("val_scenarios", []))
    print(f"Loaded checkpoint: seed={ckpt.get('seed')} epochs={ckpt.get('epochs')}")
    print(f"  trained on {len(train_scenarios)} scenarios, held out {len(val_scenarios)}: {sorted(val_scenarios)}")

    import json as _json
    if args.denoise_threshold is not None:
        denoise_threshold = args.denoise_threshold
        print(f"Denoise baseline threshold (from --denoise-threshold): dist_norm < {denoise_threshold:.2f}")
    else:
        denoise_summary_path = Path(args.denoise_summary)
        if not denoise_summary_path.exists():
            print(f"No denoise baseline summary at {denoise_summary_path} -- run "
                  f"`python -m correction_head.denoise_baseline` first, or pass --denoise-threshold directly. "
                  f"Falling back to the already-logged threshold (dist_norm < 0.68, decisions.md) so the "
                  f"gallery can still be produced.")
            denoise_threshold = 0.68
        else:
            denoise_threshold = _json.loads(denoise_summary_path.read_text())["threshold_dist_norm_lt"]
            print(f"Denoise baseline threshold (from {denoise_summary_path.name}): dist_norm < {denoise_threshold:.2f}")

    scenarios = args.scenarios if args.scenarios else sorted(val_scenarios)
    if not scenarios:
        print("No scenarios given and checkpoint has no recorded val_scenarios -- pass --scenarios explicitly.")
        return

    degradation_config = load_degradation_config()
    repair_labels_dir = Path(args.repair_labels_dir)

    violation_lookup = None
    if args.violation_lookup:
        from correction_head.dataset import load_violation_lookup
        violation_lookup = load_violation_lookup(args.violation_lookup)
        if violation_lookup is None:
            print(f"WARNING: --violation-lookup {args.violation_lookup} not found -- "
                  f"free_occ_conflict_area/tracking_affinity will be zero-filled.")
        else:
            print(f"Loaded {len(violation_lookup)} (scenario, frame) violation rows from {args.violation_lookup}")

    rows = []  # each: (scenario_id, frame_id, cav_id, split_label, xyz, raw_cluster_id, point_vehicle_id, point_frag_id, head_group_by_frag)
    if args.units:
        # Exact units requested (e.g. known false-merge cases) -- group by
        # scenario so we only load each scenario's crop cache once.
        wanted = []
        for u in args.units:
            parts = u.split(":")
            if len(parts) != 3:
                print(f"Skipping malformed --units entry {u!r}: expected scenario_id:frame_id:cav_id")
                continue
            wanted.append(tuple(parts))
        by_scenario = {}
        for scenario_id, frame_id, cav_id in wanted:
            by_scenario.setdefault(scenario_id, []).append((frame_id, cav_id))
        scenarios = list(by_scenario.keys())
    else:
        by_scenario = None

    for scenario_id in scenarios:
        split_label = "TRAIN (seen)" if scenario_id in train_scenarios else (
            "VAL (held-out)" if scenario_id in val_scenarios else "unlabeled split")
        try:
            crop_lookup, _ = load_scenario_crops(args.crops_dir, scenario_id, args.arm)
        except FileNotFoundError as e:
            print(f"Skipping {scenario_id}: {e}")
            continue

        if by_scenario is not None:
            units = by_scenario[scenario_id]
        else:
            units = discover_frame_cav_pairs(repair_labels_dir, scenario_id)[: args.max_units_per_scenario]
        for frame_id, cav_id in units:
            built = build_sample_with_fragment_ids(
                scenario_id, frame_id, cav_id, crop_lookup, repair_labels_dir, args.arm, degradation_config,
                violation_lookup=violation_lookup,
            )
            if built is None:
                print(f"Skipping {scenario_id}/{frame_id}/{cav_id}: no usable fragments (see dataset.py GAP #2).")
                continue
            crops, frag_aux, pair_index, pair_aux, labels, fragment_ids_in_order = built

            n_frag = crops.shape[0]
            uf_head = UnionFind(n_frag)
            uf_denoise = UnionFind(n_frag)
            if pair_index.shape[0] > 0:
                with torch.no_grad():
                    t_crops = torch.from_numpy(crops).float().to(device)
                    t_frag_aux = torch.from_numpy(frag_aux).float().to(device)
                    t_pair_index = torch.from_numpy(pair_index).long().to(device)
                    t_pair_aux = torch.from_numpy(pair_aux).float().to(device)
                    logits = model(t_crops, t_frag_aux, t_pair_index, t_pair_aux)
                    head_scores = logits.sigmoid().cpu().numpy()
                # denoise baseline uses ONLY pair_aux[:, 0] (dist_norm) -- same single signal
                # denoise_baseline.py's own held-out run used, scored on these exact same
                # pairs so the two "after" panels are a fair, apples-to-apples comparison,
                # not two different subsets of the scene.
                dist_norm = pair_aux[:, 0]
                for (i, j), hs, dn in zip(pair_index, head_scores, dist_norm):
                    if hs > args.threshold:
                        uf_head.union(int(i), int(j))
                    if dn < denoise_threshold:
                        uf_denoise.union(int(i), int(j))
            head_group_of_frag = dict(zip(fragment_ids_in_order.tolist(), uf_head.groups().tolist()))
            denoise_group_of_frag = dict(zip(fragment_ids_in_order.tolist(), uf_denoise.groups().tolist()))

            points_path = repair_labels_dir / scenario_id / f"{frame_id}_{cav_id}_points.npz"
            if not points_path.exists():
                print(f"Skipping plot for {scenario_id}/{frame_id}/{cav_id}: no {points_path.name} found.")
                continue
            d = np.load(points_path)
            rows.append(dict(
                scenario_id=scenario_id, frame_id=frame_id, cav_id=cav_id, split_label=split_label,
                xy=d["xyz"][:, :2], raw_cluster_id=d["raw_cluster_id"], point_vehicle_id=d["point_vehicle_id"],
                point_fragment_id=d["fragment_id"], head_group_of_frag=head_group_of_frag,
                denoise_group_of_frag=denoise_group_of_frag,
                n_fragments_scored=n_frag, n_pairs_scored=int(pair_index.shape[0]),
            ))

    if not rows:
        print("Nothing plotted -- no usable (scenario, frame, cav) units found. See messages above.")
        return

    # One PNG per (scenario, frame, CAV) unit -- NOT one combined multi-row
    # gallery. A single scene is dense enough on its own (a few hundred
    # fragments); stacking several units in one figure made it hard to
    # look at any one of them closely, which is the opposite of what a
    # "see it in action" figure is for.
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_stem = out_path.stem  # e.g. "before_after_gallery" -> per-unit files prefixed with this
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for row in rows:
        xy = row["xy"]
        in_any_fragment = row["point_fragment_id"] >= 0

        before_group = np.where(in_any_fragment, row["point_fragment_id"], -1)
        scored_frag_ids = set(row["head_group_of_frag"].keys())
        unscored_mask = in_any_fragment & ~np.isin(row["point_fragment_id"], list(scored_frag_ids))

        def fragment_groups_to_points(group_of_frag):
            g = np.full(len(xy), -1, dtype=np.int64)
            for frag_id, grp in group_of_frag.items():
                g[row["point_fragment_id"] == frag_id] = grp
            return g

        denoise_group = fragment_groups_to_points(row["denoise_group_of_frag"])
        head_group = fragment_groups_to_points(row["head_group_of_frag"])
        truth_group = np.where(row["point_vehicle_id"] != BACKGROUND_VEHICLE_ID, row["point_vehicle_id"], -1)

        n_before = len(set(before_group[before_group >= 0].tolist()))
        n_denoise = len(set(denoise_group[denoise_group >= 0].tolist()))
        n_after = len(set(head_group[head_group >= 0].tolist()))
        n_truth = len(set(truth_group[truth_group >= 0].tolist()))
        n_unscored_fragments = len(set(row["point_fragment_id"][unscored_mask].tolist()))
        header = f"{row['scenario_id']}  frame {row['frame_id']}  CAV {row['cav_id']}  [{row['split_label']}]"

        denoise_false = false_merge_group_ids(denoise_group, row["point_vehicle_id"])
        head_false = false_merge_group_ids(head_group, row["point_vehicle_id"])

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), squeeze=True)

        draw_groups(axes[0], xy, before_group,
                    "BEFORE -- raw DBSCAN fragments",
                    f"{header}\n{n_before} raw fragments (model's input granularity)")

        denoise_caption = (f"{row['n_fragments_scored']} fragments -> {n_denoise} merged groups "
                            f"(dist_norm < {denoise_threshold:.2f}, {row['n_pairs_scored']} pairs)"
                            f"\nred dashed = {len(denoise_false)} FALSE merge(s) (span >=2 real vehicles)")
        if n_unscored_fragments:
            denoise_caption += f"\norange '+' = {n_unscored_fragments} fragments never scored (no crop)"
        draw_groups(axes[1], xy, denoise_group,
                    "EXISTING METHOD -- denoise-first distance threshold",
                    denoise_caption, unscored_mask=unscored_mask, false_merge_ids=denoise_false)

        after_caption = (f"{row['n_fragments_scored']} fragments -> {n_after} merged groups "
                         f"(threshold={args.threshold}, {row['n_pairs_scored']} pairs)"
                         f"\nred dashed = {len(head_false)} FALSE merge(s) (span >=2 real vehicles)")
        if n_unscored_fragments:
            after_caption += f"\norange '+' = {n_unscored_fragments} fragments never scored (no crop)"
        draw_groups(axes[2], xy, head_group,
                    "OUR CONTRIBUTION -- learned correction head",
                    after_caption, unscored_mask=unscored_mask, false_merge_ids=head_false)

        draw_groups(axes[3], xy, truth_group,
                    "GROUND TRUTH -- real object identity",
                    f"{n_truth} real vehicles (oracle target, seen by neither method at inference)")

        fig.suptitle(f"Existing method vs. this project's correction head vs. ground truth -- {header}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        unit_name = f"{out_stem}_{row['scenario_id']}_{row['frame_id']}_{row['cav_id']}.png"
        unit_path = out_dir / unit_name
        fig.savefig(unit_path, dpi=150)
        plt.close(fig)
        saved_paths.append(unit_path)
        print(f"Saved: {unit_path}  (denoise false merges: {len(denoise_false)}, head false merges: {len(head_false)})")

    print(f"\n{len(saved_paths)} image(s) written to {out_dir}")


if __name__ == "__main__":
    main()
