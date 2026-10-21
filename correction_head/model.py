"""Cluster-repair ("declustering") correction head -- architecture pinned
Thu 1 Oct 2026 (Oct 4 gate, execution_plan_daily.md Week 4').

WHAT THIS IS
------------
The federated object for Part 3 (DFL). Consumes, per candidate cluster
fragment produced by P4's RANSAC+DBSCAN pass:
  1. a cropped BEV feature map around the fragment's centroid, read from
     P3's cached `spatial_features_2d` (P3_perception/perception_cache.py,
     `crop_bev_feature` / `_pixel_from_xy`), and
  2. a handful of per-fragment / per-frame scalar auxiliary features drawn
     from P4/P5/P6's real cached tables (never learned, never raw pixels --
     see AUX_FEATURE_NAMES below for the exact source of each one),
and outputs a pairwise same-object affinity for every candidate fragment
pair within P4's labelling pipeline's candidate-pair radius
(P4_defence/run_cluster_repair_labelling.py, CANDIDATE_PAIR_RADIUS_M).
A lightweight greedy/graph post-process (not part of this module) turns
that pairwise affinity into a repaired cluster set: merging fragments that
belong together (repairs MERGED/SPLIT), and recovering a fragment that was
ground-swallowed into noise (repairs GROUND_SWALLOWED).

Supervised target: P5_labelling.labelling.pairwise_same_object_affinity /
the `affinity_label` column written to `<frame>_pairs.parquet` by
run_cluster_repair_labelling.py.

WHY THESE SHAPES (grounded in real config/code, not guessed)
--------------------------------------------------------------
- BEV crop channel count = 384. `pointpillar_late_fusion/config.yaml`'s
  `model.args.base_bev_backbone.num_upsample_filter` is [128, 128, 128];
  OpenCOOD's BaseBEVBackbone concatenates the three upsampled scales along
  the channel dim before returning `spatial_features_2d`, so the real
  checkpoint's feature map is (384, H, W) per CAV.
- BEV crop spatial size = 7x7. `P3_perception/perception_cache.py`'s
  `BEV_CROP_SIZE_PX = 7` (already implemented and unit-tested there).
- `lidar_range` = [-70.4, -40, -3, 70.4, 40, 1] from the same config.yaml,
  used below only to normalise the range/bearing auxiliary features into
  roughly [-1, 1] / [0, 1] -- it is not a free hyperparameter of this head.

PARAMETER COUNT
----------------
See `correction_head/count_params.py`, which computes the exact param
count from this same architecture spec using plain arithmetic (no torch
import, so it runs anywhere, including the cloud sandbox this file was
drafted in). At HEAD_CHANNELS=128 / EMB_DIM=64 (the defaults below) the
total is ~388K parameters (388,098 exactly, after the Wed 7 Oct 2026 frag_extent_norm
feature) -- inside the "hundreds of thousands to low
millions" ceiling from scope_and_timeline_oct19.md, and it lands in the
same ~300-500K band the project's own comms-budget estimate
(~1.2-2MB fp32 / ~300-500KB int8 / ~40-65KB 1-bit) was already written
against in task1_datasets_pipeline_dfl.md Sec 4 -- so those figures don't
need to be revised now that a real architecture backs them.

NORMALISATION CHOICE -- GroupNorm, not BatchNorm
--------------------------------------------------
BatchNorm's running mean/var are themselves parameters that would need to
be federated, and FL/DFL over small, non-IID local batches (a single
vehicle's ~200-frame buffer, itself skewed by that vehicle's own
range/density profile -- see P6) makes BatchNorm statistics unstable and
leaks cross-client information through them (the standard FedBN-style
objection). GroupNorm is computed per-sample, so it has no running
statistics to federate and behaves identically online and at gossip time.

STATUS -- NOT EXECUTION-VERIFIED
-----------------------------------
This file needs `torch`, which does not exist either in the cloud sandbox
or in the remote-device bridge's own Linux VM (checked directly: both
`import torch` and `import opencood` raise ModuleNotFoundError there).
Like P3's crop-caching code before it, this is sandbox-drafted against the
real config/source but has never been instantiated or forward-passed for
real. Treat the shapes/param-count as a paper design until it's run once
in the opencood conda env on the real machine -- see
`correction_head/verify_on_device.py` for the exact commands to do that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---- shapes fixed by upstream, real code/config (see module docstring) ----
BEV_CROP_CHANNELS = 384          # sum(num_upsample_filter) from config.yaml
BEV_CROP_SIZE_PX = 7             # P3_perception.perception_cache.BEV_CROP_SIZE_PX -- tried 11 on Mon 5 Oct 2026, regressed (see decisions.md), reverted to 7
LIDAR_RANGE = (-70.4, -40.0, -3.0, 70.4, 40.0, 1.0)  # config.yaml model.args.lidar_range

# ---- architecture hyperparameters (the actual design decision) ----
HEAD_CHANNELS = 128     # conv width after the 1x1 channel-reduction
FUSE_HIDDEN = 128       # fragment-MLP hidden width
EMB_DIM = 64            # fragment embedding size fed to the pairwise head
PAIR_HIDDEN = 128       # pairwise-MLP hidden width
GROUPNORM_GROUPS = 8

# Per-fragment auxiliary scalars, in fixed order. Every one of these is a
# real column already produced upstream -- nothing here is invented at
# train time:
#   0 log1p_n_points          <- fragments.parquet 'n_points' (P5/run_cluster_repair_labelling.py)
#   1 is_recovered_noise      <- fragments.parquet 'is_recovered_noise'
#   2 range_from_ego_norm     <- fragments.parquet centroid_x/y, normalised by LIDAR_RANGE
#   3 bearing_norm            <- atan2(centroid_y, centroid_x) / pi
#   4 range_density_bucket    <- P6_features.features.compute_sensing_profile_from_repair_labels
#   5 degradation_severity    <- P2 (type, severity) metadata; 0.0 for arm A (clean benign)
#   6 free_occ_conflict_area  <- P4's free/occupied violation scalars (consistency_layer.py), frame-level, broadcast to its fragments
#   7 tracking_affinity       <- P4's free/occupied violation scalars, frame-level, broadcast
#   8 frag_extent_norm        <- log1p(RMS radius from centroid, meters), computed from the
#                                 real per-point xyz in <frame>_<cav>_points.npz grouped by
#                                 fragment_id (Wed 7 Oct 2026 "missing metric" direction,
#                                 decisions.md): log1p_n_points alone can't tell a dense tiny
#                                 sliver from a genuinely small-but-sparse fragment; this adds
#                                 the spatial-spread signal that was missing.
AUX_FEATURE_NAMES = (
    "log1p_n_points",
    "is_recovered_noise",
    "range_from_ego_norm",
    "bearing_norm",
    "range_density_bucket",
    "degradation_severity",
    "free_occ_conflict_area",
    "tracking_affinity",
    "frag_extent_norm",
)
FRAGMENT_AUX_DIM = len(AUX_FEATURE_NAMES)

# Per-pair auxiliary scalars (order fixed):
#   0 centroid_distance_norm  <- pairs.parquet 'centroid_distance_m', normalised by CANDIDATE_PAIR_RADIUS_M
#   1 abs_delta_log_n_points  <- |log1p(n_points_i) - log1p(n_points_j)|
#   2 interposed_object_signal <- Thu 8 Oct 2026 physics feature: log1p(count of OTHER
#                                   fragments' real LiDAR points lying on the straight-line
#                                   segment between this pair's two centroids). A solid rigid
#                                   body can't be pierced by another solid object -- unlike
#                                   free_occ_conflict_area/tracking_affinity above (frame-level
#                                   aggregates broadcast to every fragment regardless of which
#                                   pair is scored, which regressed different-vehicle FPR when
#                                   tried Thu 8 Oct 2026), this is computed per-PAIR from the
#                                   exact two fragments being scored. See dataset.py's
#                                   _pairwise_interposed_signal().
PAIR_AUX_DIM = 3


class FragmentEncoder(nn.Module):
    """(B, 384, 7, 7) BEV crop + (B, FRAGMENT_AUX_DIM) scalars -> (B, EMB_DIM) embedding."""

    def __init__(
        self,
        in_channels: int = BEV_CROP_CHANNELS,
        head_channels: int = HEAD_CHANNELS,
        aux_dim: int = FRAGMENT_AUX_DIM,
        fuse_hidden: int = FUSE_HIDDEN,
        emb_dim: int = EMB_DIM,
        groups: int = GROUPNORM_GROUPS,
    ):
        super().__init__()
        # 1x1 channel reduction: this is where almost none of the model's
        # capacity should go, so it's a plain linear projection per pixel.
        self.reduce = nn.Conv2d(in_channels, head_channels, kernel_size=1)
        self.gn1 = nn.GroupNorm(groups, head_channels)
        # two 3x3 convs at fixed 7x7 spatial size (padding=1 keeps it 7x7) --
        # this is where the spatial pattern around the candidate fragment
        # (is there BEV texture that looks like one object or two?) is
        # actually learned.
        self.conv1 = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, head_channels)
        self.conv2 = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(groups, head_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.ReLU(inplace=True)

        self.fuse = nn.Sequential(
            nn.Linear(head_channels + aux_dim, fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fuse_hidden, emb_dim),
        )

    def forward(self, crop: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn1(self.reduce(crop)))
        x = self.act(self.gn2(self.conv1(x)))
        x = self.act(self.gn3(self.conv2(x)))
        x = self.pool(x).flatten(1)  # (B, head_channels)
        x = torch.cat([x, aux], dim=1)
        return self.fuse(x)  # (B, emb_dim)


GATE_HIDDEN = 16  # width of the learned distance/size-gap gate below


class PairwiseAffinityHead(nn.Module):
    """Two fragment embeddings + pair scalars -> same-object affinity logit.

    Combines the pair symmetrically (sum + abs-diff of the two embeddings)
    so the score doesn't depend on which fragment is passed as i vs j --
    the affinity between fragment A and B has no natural order.

    LEARNED DISTANCE GATE -- "option 2", added Tue 7 Oct 2026 (decisions.md)
    -------------------------------------------------------------------------
    pair_aux (dist_norm, abs_delta_log_n, interposed_object_signal) already
    feeds the main `mlp` above, same as every other input -- but Mon 5 Oct
    2026's diagnosis
    (diagnose_false_merges.py) found the head routinely ignoring dist_norm
    on far, size-mismatched different-vehicle pairs anyway: it's one of
    many inputs a shared MLP is free to underweight in favour of easier
    appearance cues. Option 1 (hard-negative-mining the loss, Tue 7 Oct
    2026) tried fixing this by showing the model more of exactly those
    pairs; it didn't help (different-vehicle FPR 12.09% -> 13.74%, worse).

    `self.gate` is a second, SEPARATE, narrower network that sees ONLY
    pair_aux (not the fragment embeddings) and outputs a penalty
    subtracted from the main mlp's logit. This gives distance/size-gap a
    structurally privileged, dedicated path to the final score instead of
    being drowned out inside the large shared MLP -- but the gate's own
    weights are learned like everything else here; there is no hardcoded
    distance cutoff anywhere (a fixed threshold was explicitly rejected
    Tue 7 Oct 2026 as not a learned fix and as tying the result to this
    project's own sensor/crop config). The network still decides, during
    training, how large that penalty should be and for which inputs.
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        pair_aux_dim: int = PAIR_AUX_DIM,
        hidden: int = PAIR_HIDDEN,
        gate_hidden: int = GATE_HIDDEN,
    ):
        super().__init__()
        # Input width is 2*emb_dim + pair_aux_dim, NOT emb_dim + pair_aux_dim:
        # forward() below concatenates TWO emb_dim-wide symmetric combinations
        # (emb_i+emb_j and |emb_i-emb_j|), not one. A real run of
        # verify_on_device.py (first execution of this module against real
        # torch, after the paper design was pinned) caught this as a shape
        # mismatch (RuntimeError: 600x130 and 66x128) -- the bug was here,
        # in __init__'s declared width, not in forward()'s actual computation.
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim + pair_aux_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Learned distance/size-gap gate -- see class docstring above.
        self.gate = nn.Sequential(
            nn.Linear(pair_aux_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor, pair_aux: torch.Tensor) -> torch.Tensor:
        symmetric = torch.cat([emb_i + emb_j, torch.abs(emb_i - emb_j)], dim=1)
        x = torch.cat([symmetric, pair_aux], dim=1)
        main_logit = self.mlp(x).squeeze(-1)
        gate_penalty = self.gate(pair_aux).squeeze(-1)
        # Subtracted in logit space -- a learned, pair_aux-conditioned
        # dampening of the main network's confidence (larger penalty ->
        # lower sigmoid probability of "same object"), not a hard cutoff.
        return main_logit - gate_penalty  # (B,) raw logit -- apply sigmoid / BCEWithLogitsLoss outside


class CorrectionHead(nn.Module):
    """Top-level module: shares one FragmentEncoder across all fragments in
    a batch, then scores requested pairs with PairwiseAffinityHead. This is
    the single nn.Module whose state_dict is what gets federated in Part 3
    (Δw per gossip round, aggregated per the axis-2 rule in force).
    """

    def __init__(self):
        super().__init__()
        self.encoder = FragmentEncoder()
        self.pair_head = PairwiseAffinityHead()

    def forward(
        self,
        crops: torch.Tensor,        # (N, 384, 7, 7) -- N candidate fragments in this batch
        frag_aux: torch.Tensor,     # (N, FRAGMENT_AUX_DIM)
        pair_index: torch.Tensor,   # (P, 2) long -- indices into crops/frag_aux for each requested pair
        pair_aux: torch.Tensor,     # (P, PAIR_AUX_DIM)
    ) -> torch.Tensor:
        emb = self.encoder(crops, frag_aux)               # (N, EMB_DIM)
        emb_i = emb[pair_index[:, 0]]
        emb_j = emb[pair_index[:, 1]]
        return self.pair_head(emb_i, emb_j, pair_aux)      # (P,) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
