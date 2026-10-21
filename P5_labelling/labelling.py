"""
P5 -- Cluster-correction + condition labelling.

Condition labelling + synthetic attack arm implemented 17 Sep 2026 (Week 2,
compressed into one sitting -- originally Sun 13 Sep). Cluster-correction
labelling (the correction head's actual supervised target) added 25 Sep 2026
once the clustering-failure design was locked in -- see decisions.md and
task1_datasets_pipeline_dfl.md §2/§4.

y_flag, y_cause in {none, noise, attack}, + degradation type/severity.
Report FPR and TPR separately, NEVER accuracy -- class imbalance (few
attacked frames, many clean/degraded ones) makes accuracy meaningless. This
constraint carries into P7's evaluation harness too, not just here. The same
"never bare accuracy" rule applies to cluster-repair below: report
cluster-repair accuracy, FPR and TPR separately (task1_datasets_pipeline_dfl.md
§2), not one combined number.

**Attack arm design (condition C):** per scope_and_timeline_oct19.md, this is
a synthetic signature, explicitly NOT a reimplementation of a published
attack (ROBOSAC/MADE/MATE's actual attacks are out of scope). The signature
chosen here is a "ghost object" injection: a malicious CAV fabricates a
claimed-occupied polygon inside a region another CAV's sensor has already
confirmed empty (free space) -- i.e. it reports seeing something that isn't
there. This maps directly onto the free/occupied violation mechanism P4
already implements (`free_vs_occupied_violation` in consistency_layer.py),
so an attacked frame is, by construction, one degrees of freedom would
predict `free_vs_occupied_violation` should catch -- giving P7 a TPR number
that's actually testing the defence's real mechanism, not a strawman.

**Cluster-correction labelling (added 25 Sep 2026):** the supervised target
for Part 2's correction head, per task1_datasets_pipeline_dfl.md §2: "for
every candidate cluster fragment / noise point, whether it belongs with
another fragment (repairs a MERGED or SPLIT case) or constitutes a missed
real object (repairs a GROUND_SWALLOWED case) -- derived directly from the
same GT-box cross-referencing P4 uses, so pretraining has real oracle
supervision, not a pseudo-label."

The oracle chain is: for every non-ground point, find which real GT
vehicle's footprint (if any) actually contains it
(`label_point_vehicle_membership`) -- this is ground truth, not a guess.
Grouping points by that true vehicle id, independent of what DBSCAN's raw
clustering said, gives the REPAIRED cluster assignment
(`build_cluster_repair_target`) -- comparing it against DBSCAN's raw
cluster_labels is what shows, per point, whether a merge/split/recovery is
needed, and directly reproduces the CLEAN/MERGED/SPLIT/GROUND_SWALLOWED
taxonomy from consistency_layer.py / cluster_failure_audit.py as a
consequence, not a separate rule. Finally `pairwise_same_object_affinity`
turns the same oracle into the actual training signal the correction head
predicts (task1_datasets_pipeline_dfl.md §4: "a pairwise same-object
affinity over candidate fragments").

This intentionally only repairs VEHICLE clusters. Points that fall inside no
GT vehicle box (static structure -- walls, poles, curbs) are background and
are never assigned a repaired cluster or an affinity, whatever DBSCAN did
with them -- out of scope for this labelling task.

**Candidate fragments (added 26 Sep 2026):** §4's actual I/O unit for the
correction head is "candidate cluster fragment / noise point," not a raw
point pair -- `build_candidate_fragments` gives noise points a recoverable
fragment identity (via a second, looser re-clustering pass, so
GROUND_SWALLOWED vehicles have something to promote) and
`fragment_majority_vehicle_id` extends the same point-level oracle to score
fragment-level `pairwise_same_object_affinity`. See the extended comment
above `build_candidate_fragments` for the MERGED-cluster simplification this
implies. `run_cluster_repair_labelling.py` (P4_defence/) is the end-to-end
script that ties all of this together against real per-frame data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


@dataclass
class LabelResult:
    y_flag: bool           # did the defence flag this frame/pair?
    y_cause: str            # "none" | "noise" | "attack" -- ground truth, not the defence's guess
    degradation_type: str   # "none" | "fog" | "compression"
    severity: int            # 0 if none
    arm: str                 # "A" (clean) | "B" (degraded benign) | "C" (clean + attack)


def label_condition(flag_decision: bool, degradation_meta: dict | None, attack_meta: dict | None) -> LabelResult:
    """
    Assigns ground-truth cause and arm for one (scenario, frame[, CAV pair])
    row, given what was actually injected upstream (P2's degradation_meta,
    this file's attack_meta) and what the defence actually decided
    (flag_decision, from P4/run_defence_check.py).

    degradation_meta: {"type": "fog"|"compression", "severity": 1|2} or None
        if this CAV/frame had no degradation injected.
    attack_meta: {"injected": True, ...} or None if no attack was injected.

    Precedence: attack > degradation > none. A frame is never simultaneously
    labelled "noise" and "attack" -- arms are mutually exclusive per the
    plan's three-arm design (A clean benign / B degraded benign, no attack /
    C clean + attack). If both were somehow injected on the same row, that's
    a caller bug (arms shouldn't overlap), and attack takes precedence in the
    label so it's at least not silently miscounted as a benign false
    positive.
    """
    attack_injected = bool(attack_meta) and attack_meta.get("injected", False)
    degradation_type = (degradation_meta or {}).get("type", "none")
    severity = int((degradation_meta or {}).get("severity", 0))
    degraded = degradation_type != "none" and severity > 0

    if attack_injected:
        y_cause = "attack"
        arm = "C"
    elif degraded:
        y_cause = "noise"
        arm = "B"
    else:
        y_cause = "none"
        arm = "A"

    return LabelResult(
        y_flag=bool(flag_decision),
        y_cause=y_cause,
        degradation_type=degradation_type,
        severity=severity,
        arm=arm,
    )


# --- Synthetic attack signature: ghost-object injection ---

GHOST_OBJECT_AREA_M2_RANGE = (3.0, 8.0)  # roughly car-to-van footprint, per the realistic-car-
                                          # sized-footprint check used in run_defence_check.py's
                                          # own synthetic dry-run (16 Sep 2026, decisions.md)


def inject_synthetic_attack(
    free_space: Polygon, seed: int, area_m2: float | None = None
) -> Polygon:
    """
    Fabricates one "ghost object" occupied-space claim placed inside a
    victim CAV's own confirmed-free space -- the attacker (a different CAV)
    falsely reports an object where the victim's sensor has already
    confirmed there is nothing. This is the synthetic attack signature for
    condition C.

    free_space: the victim CAV's get_free_space() polygon (already aligned
        to a shared frame, e.g. ego's, via align_polygon_to_ego -- must be
        called on ALIGNED polygons for the placement to land inside the
        actual shared free-space region).
    seed: RNG seed for reproducibility, keyed by caller the same way P2's
        degradation seeds are (scenario, frame, attacker_cav_id).
    area_m2: ghost object footprint; if None, drawn uniformly from
        GHOST_OBJECT_AREA_M2_RANGE.

    Returns a Polygon (a small square, area_m2 in size) placed at a random
    point interior to free_space. Raises ValueError if free_space is empty
    or has no interior area to place a ghost object inside (nothing to
    attack -- the victim's sensor didn't confirm any free space here).
    """
    if free_space.is_empty or free_space.area <= 0:
        raise ValueError("free_space is empty -- no confirmed-empty region to place a ghost object in")

    rng = np.random.default_rng(seed)
    if area_m2 is None:
        area_m2 = rng.uniform(*GHOST_OBJECT_AREA_M2_RANGE)
    half_side = (area_m2 ** 0.5) / 2.0

    minx, miny, maxx, maxy = free_space.bounds
    # Rejection-sample a center point that's genuinely interior to
    # free_space (its bounding box may include area outside the polygon,
    # e.g. for the star-shaped ray-cast polygons get_free_space() produces).
    for _ in range(200):
        cx = rng.uniform(minx, maxx)
        cy = rng.uniform(miny, maxy)
        candidate = Polygon(
            [
                (cx - half_side, cy - half_side),
                (cx + half_side, cy - half_side),
                (cx + half_side, cy + half_side),
                (cx - half_side, cy + half_side),
            ]
        )
        if free_space.contains(candidate):
            return candidate

    raise ValueError(
        "could not place a ghost object fully inside free_space after 200 attempts -- "
        "free_space may be too small or too irregularly shaped for the requested area_m2"
    )


# --- Cluster-correction (oracle repair target) labelling ---


def points_in_rotated_box_xy(p_xy: np.ndarray, box_corner_xy: np.ndarray) -> np.ndarray:
    """
    Same function as P4_defence/cluster_failure_audit.py's
    points_in_rotated_box_xy -- duplicated here (rather than imported) so
    P5_labelling's own dependencies stay light (numpy + shapely only).
    Importing cluster_failure_audit.py directly would pull in open3d,
    pandas and pyyaml transitively, none of which this labelling module
    otherwise needs -- the same reasoning cluster_failure_audit.py itself
    gives for not importing box_utils.py's create_bbx (see that file's
    module docstring).

    box_corner_xy: 4 corners (bottom face) of the box footprint, in the
    winding order create_bbx() produces: 0,1,2,3 = bottom face, i.e.
    edge (corner0 -> corner1) and edge (corner0 -> corner3) are the box's
    two in-plane axes.
    """
    edge1 = box_corner_xy[1] - box_corner_xy[0]
    edge2 = box_corner_xy[3] - box_corner_xy[0]
    p_rel = p_xy - box_corner_xy[0].reshape(1, -1)
    l1 = (p_rel @ edge1) / np.dot(edge1, edge1)
    l2 = (p_rel @ edge2) / np.dot(edge2, edge2)
    return (l1 >= 0) & (l1 <= 1) & (l2 >= 0) & (l2 <= 1)


BACKGROUND_VEHICLE_ID = -1  # sentinel: point falls inside no GT vehicle's footprint


def label_point_vehicle_membership(points_xy: np.ndarray, gt_boxes: dict) -> np.ndarray:
    """
    Ground-truth "which real object owns this point" -- the oracle P5's
    cluster-correction labelling is built on (task1_datasets_pipeline_dfl.md
    §2: "derived directly from the same GT-box cross-referencing P4 uses").

    points_xy: (N, 2) non-ground points, in the CAV's own local lidar frame.
    gt_boxes: {vehicle_id: box_corner_xy} -- each vehicle's GT footprint (4
        corners, same winding as points_in_rotated_box_xy expects), already
        in the SAME frame as points_xy. P4 produces these (e.g. via
        cluster_failure_audit.project_vehicle_to_lidar); this function
        takes the footprints as given rather than recomputing them, keeping
        the projection math and the labelling logic separately testable.

    Returns an (N,) int array: the vehicle_id whose footprint contains that
    point, or BACKGROUND_VEHICLE_ID (-1) if the point falls inside no GT
    vehicle's box (static structure -- walls, poles, curbs -- not this
    labelling task's concern; see module docstring).

    A point that falls inside more than one vehicle's footprint (rare --
    only possible for badly overlapping GT boxes) is assigned to whichever
    vehicle_id is encountered first in gt_boxes' iteration order. This is a
    deliberate, simple tie-break, not a nearest-centroid resolution --
    flagged here rather than silently doing something fancier.
    """
    points_xy = np.asarray(points_xy)
    n = len(points_xy)
    vehicle_id = np.full(n, BACKGROUND_VEHICLE_ID, dtype=int)
    for vid, box_xy in gt_boxes.items():
        inside = points_in_rotated_box_xy(points_xy, np.asarray(box_xy))
        unassigned = vehicle_id == BACKGROUND_VEHICLE_ID
        vehicle_id[unassigned & inside] = vid
    return vehicle_id


@dataclass
class ClusterRepairTarget:
    """
    Oracle-derived correction-head training target for one frame/CAV.

    raw_cluster_id: DBSCAN's own (unrepaired) cluster label per point, -1
        for noise -- consistency_layer.py's cluster_points() output,
        unchanged.
    point_vehicle_id: ground truth from label_point_vehicle_membership():
        the real vehicle each point belongs to, BACKGROUND_VEHICLE_ID (-1)
        if none.
    repaired_cluster_id: what the correction head should output. Every
        point sharing the same point_vehicle_id gets the same
        repaired_cluster_id (namespaced to point_vehicle_id's own values,
        so repaired_cluster_id IS point_vehicle_id) regardless of
        raw_cluster_id -- background points always stay -1. Diffing
        raw_cluster_id against repaired_cluster_id per point is what
        recovers the CLEAN/MERGED/SPLIT/GROUND_SWALLOWED taxonomy as a
        consequence: a MERGED cluster's points fan out into >=2 different
        repaired_cluster_id groups; a SPLIT vehicle's points collapse from
        >=2 raw_cluster_id groups into one repaired_cluster_id; a
        GROUND_SWALLOWED vehicle's points move from raw_cluster_id == -1
        (noise) to a real (non background) repaired_cluster_id.
    """
    raw_cluster_id: np.ndarray
    point_vehicle_id: np.ndarray
    repaired_cluster_id: np.ndarray


def build_cluster_repair_target(cluster_labels: np.ndarray, point_vehicle_id: np.ndarray) -> ClusterRepairTarget:
    """
    Builds the oracle repair target by grouping points by their TRUE vehicle
    id (point_vehicle_id), completely independent of what DBSCAN's raw
    cluster_labels said. See ClusterRepairTarget's docstring for how this
    recovers each failure mode.
    """
    cluster_labels = np.asarray(cluster_labels)
    point_vehicle_id = np.asarray(point_vehicle_id)
    if cluster_labels.shape != point_vehicle_id.shape:
        raise ValueError(
            f"cluster_labels shape {cluster_labels.shape} != "
            f"point_vehicle_id shape {point_vehicle_id.shape} -- must be "
            "per-point arrays over the same point set"
        )
    # repaired_cluster_id IS point_vehicle_id by construction (see docstring)
    # -- kept as its own field/name rather than reusing point_vehicle_id
    # directly so callers reading a ClusterRepairTarget don't have to know
    # that equivalence to use the field that represents "what the head
    # should output".
    repaired_cluster_id = point_vehicle_id.copy()
    return ClusterRepairTarget(
        raw_cluster_id=cluster_labels,
        point_vehicle_id=point_vehicle_id,
        repaired_cluster_id=repaired_cluster_id,
    )


def pairwise_same_object_affinity(point_vehicle_id: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    The correction head's actual training target (task1_datasets_pipeline_dfl.md
    §4: the head "outputs a pairwise same-object affinity over candidate
    fragments"). For arbitrary point-index arrays i, j (same length,
    paired elementwise -- candidate fragment pairs the caller wants scored,
    not necessarily all pairs), returns a bool array: True iff that pair
    truly belongs to the same real vehicle.

    Background points (point_vehicle_id == BACKGROUND_VEHICLE_ID) never
    have affinity with anything, even each other -- two points that are
    both "not any known GT vehicle" are not thereby known to be the SAME
    thing, so this deliberately does not assert an affinity there.

    Note this operates on whatever id array the caller hands it -- it is
    used both at the raw POINT level (as documented above) and, via
    fragment_majority_vehicle_id below, at the FRAGMENT level for the
    actual correction-head training pairs (see build_candidate_fragments'
    module-level note on why fragments, not raw points, are the real unit).
    """
    point_vehicle_id = np.asarray(point_vehicle_id)
    i = np.asarray(i)
    j = np.asarray(j)
    same_vehicle = point_vehicle_id[i] == point_vehicle_id[j]
    not_background = point_vehicle_id[i] != BACKGROUND_VEHICLE_ID
    return same_vehicle & not_background


# --- Candidate fragments (added 26 Sep 2026, building the end-to-end P4->P5
# labelling script per decisions.md's "Next" line) ---
#
# task1_datasets_pipeline_dfl.md §4 is explicit that the correction head's
# actual I/O unit is a "candidate cluster fragment / noise point," not a raw
# point: it "outputs a pairwise same-object affinity over candidate
# fragments." raw DBSCAN clusters (cluster_points()'s output, ids >= 0) are
# the obvious fragments for MERGED/SPLIT repair. But GROUND_SWALLOWED
# vehicles are, by definition, entirely noise (raw_cluster_id == -1) -- if
# fragments were ONLY raw clusters, a ground-swallowed vehicle would have no
# fragment at all for the head to promote into a recovered cluster. So noise
# points get a second, deliberately looser re-clustering pass
# (build_candidate_fragments below) purely to give them a candidate identity;
# it does not change raw_cluster_id or anything upstream, and it does not by
# itself decide whether a fragment is "real" -- that's still the oracle
# (point_vehicle_id) via fragment_majority_vehicle_id.
#
# Known, documented simplification: a raw cluster whose points span two real
# vehicles (a MERGED cluster) stays ONE fragment here -- splitting it isn't
# something fragment-pairwise affinity can express on its own (affinity is a
# relation BETWEEN fragments, not a rule for dividing one). The point-level
# repaired_cluster_id from build_cluster_repair_target already carries that
# distinction in full; this fragment/affinity view is the correction head's
# actual I/O contract, not a replacement for it. Both are written by
# run_cluster_repair_labelling.py.
#
# Empirically confirmed against real data while building the labelling
# script (26 Sep 2026): recovering a GROUND_SWALLOWED vehicle's noise into
# ONE clean fragment is NOT guaranteed either -- a real swallowed vehicle's
# points can still fragment into several recovered pieces (or leave a few
# points unrecovered) under the looser pass, since it's still just spatial
# proximity within a noisier point set. That's not a bug to paper over: it's
# exactly why fragment-level pairwise affinity, not a single recovery rule,
# is the correction head's actual job -- it has to learn to stitch those
# pieces back together the same way it stitches a SPLIT vehicle's fragments.


def build_candidate_fragments(
    points_xy: np.ndarray,
    raw_cluster_id: np.ndarray,
    noise_eps_m: float,
    noise_min_samples: int = 3,
) -> np.ndarray:
    """
    fragment_id per point: raw_cluster_id's own values for already-clustered
    points (raw_cluster_id >= 0, unchanged), new ids for noise points that a
    second, looser DBSCAN pass over ONLY the noise subset manages to group
    (offset past raw_cluster_id's max so they never collide with real
    cluster ids), and -1 for points that stay unclustered even in that
    loosened pass (a single stray point isn't a candidate fragment).

    noise_eps_m: pass consistency_layer.DBSCAN_EPS_M explicitly -- not
    imported here to keep this module's own dependencies to numpy + shapely
    (+ this function's local sklearn import) rather than pulling in
    consistency_layer's open3d/pandas/pyyaml chain. Using the SAME eps as
    the real pipeline (just a smaller min_samples) is deliberate: this is
    meant to recover groups the real pass was too strict to keep, not to
    invent a differently-shaped notion of "nearby."
    noise_min_samples: looser than the real pipeline's min_samples (10) by
    design -- default 3, i.e. a real vehicle needs only 3 nearby non-ground
    points, not 10, to earn a recoverable fragment identity. Still >1 so a
    single isolated point doesn't become its own fragment.
    """
    from sklearn.cluster import DBSCAN

    points_xy = np.asarray(points_xy)
    raw_cluster_id = np.asarray(raw_cluster_id)
    fragment_id = raw_cluster_id.copy()

    noise_mask = raw_cluster_id == -1
    if np.any(noise_mask):
        noise_points = points_xy[noise_mask]
        sub_labels = DBSCAN(eps=noise_eps_m, min_samples=noise_min_samples).fit_predict(noise_points)
        clustered_ids = raw_cluster_id[raw_cluster_id >= 0]
        offset = int(clustered_ids.max()) + 1 if clustered_ids.size else 0
        recovered = sub_labels >= 0
        new_ids = np.where(recovered, sub_labels + offset, -1)
        fragment_id[noise_mask] = new_ids

    return fragment_id


def fragment_majority_vehicle_id(fragment_id: np.ndarray, point_vehicle_id: np.ndarray) -> dict:
    """
    Majority-vote real vehicle id per fragment (see build_candidate_fragments),
    for scoring fragment-level pairwise_same_object_affinity -- pass the
    resulting {fragment_id: vehicle_id} back through pairwise_same_object_affinity
    by building a per-fragment id array and pairing fragment positions, the
    same way that function is used at the raw point level.

    A fragment with no non-background points at all maps to
    BACKGROUND_VEHICLE_ID. A fragment that straddles more than one real
    vehicle, or mixes a vehicle's edge points with background clutter (only
    possible for the noise-recovered fragments above, since raw DBSCAN
    clusters are what the whole audit is measuring the impurity of), is
    assigned by plurality among its NON-background points -- a deliberate
    simplification: an impure fragment is itself evidence of an upstream
    clustering problem, not something this labelling step tries to further
    disentangle. fragment_id == -1 (unfragmented stray points, see
    build_candidate_fragments) is skipped -- those were never candidate
    fragments to begin with.
    """
    fragment_id = np.asarray(fragment_id)
    point_vehicle_id = np.asarray(point_vehicle_id)
    result = {}
    for frag in np.unique(fragment_id):
        if frag == -1:
            continue
        members = point_vehicle_id[fragment_id == frag]
        non_background = members[members != BACKGROUND_VEHICLE_ID]
        if len(non_background) == 0:
            result[int(frag)] = BACKGROUND_VEHICLE_ID
        else:
            values, counts = np.unique(non_background, return_counts=True)
            result[int(frag)] = int(values[np.argmax(counts)])
    return result
