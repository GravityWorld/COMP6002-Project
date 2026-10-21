"""Axis 2 -- aggregation rules.

Every rule has the same signature: given a vehicle's own delta, its
neighbours' deltas (already received + decoded over the comms channel --
see comms.py), and optional per-neighbour trust scores / sample counts,
return one aggregated delta to add to that vehicle's weights. This is
step (4) of the one-gossip-round spec in task1_datasets_pipeline_dfl.md
Sec 4: "w_i <- w_i + sum_j alpha_ij * Delta w_j".

All four rules operate on flat 1-D numpy arrays (a vehicle's full Delta w,
flattened) so the same code path serves both the synthetic test harness
here and, later, a real flattened state_dict from correction_head/model.py
(flatten/unflatten is the dataset-loader-adjacent glue, not this module's
job).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Plain average -- the FedAvg-style baseline, axis-2's non-robust arm.
# Deliberately has NO defence against a poisoned neighbour; that's the point
# -- it's what the poisoning curve (axis 3) is meant to show degrading.
# ---------------------------------------------------------------------------


def aggregate_average(
    own_delta: np.ndarray,
    neighbour_deltas: Sequence[np.ndarray],
    neighbour_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    all_deltas = [own_delta, *neighbour_deltas]
    if neighbour_weights is None:
        return np.mean(np.stack(all_deltas, axis=0), axis=0)
    weights = np.asarray([1.0, *neighbour_weights], dtype=np.float64)
    weights = weights / weights.sum()
    stacked = np.stack(all_deltas, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


# ---------------------------------------------------------------------------
# Coordinate-wise trimmed mean -- robust rule #1. Drops the top/bottom
# `trim_frac` fraction of values at EACH coordinate independently before
# averaging, so a single wildly-off neighbour can only pull a coordinate as
# far as the trim allows, not arbitrarily.
# ---------------------------------------------------------------------------


def aggregate_trimmed_mean(
    own_delta: np.ndarray,
    neighbour_deltas: Sequence[np.ndarray],
    trim_frac: float = 0.2,
) -> np.ndarray:
    all_deltas = np.stack([own_delta, *neighbour_deltas], axis=0)  # (K, D)
    k = all_deltas.shape[0]
    n_trim = int(np.floor(trim_frac * k))
    if 2 * n_trim >= k:
        # Too few peers to trim anything meaningfully -- fall back to a
        # plain coordinate-wise median rather than trimming the whole set.
        return np.median(all_deltas, axis=0)
    sorted_vals = np.sort(all_deltas, axis=0)
    kept = sorted_vals[n_trim : k - n_trim] if n_trim > 0 else sorted_vals
    return np.mean(kept, axis=0)


# ---------------------------------------------------------------------------
# Coordinate-wise median -- robust rule #2. Even more robust than trimmed
# mean to a minority of poisoned neighbours (breakdown point ~50%), at the
# cost of discarding more information from honest neighbours too.
# ---------------------------------------------------------------------------


def aggregate_median(
    own_delta: np.ndarray,
    neighbour_deltas: Sequence[np.ndarray],
) -> np.ndarray:
    all_deltas = np.stack([own_delta, *neighbour_deltas], axis=0)
    return np.median(all_deltas, axis=0)


# ---------------------------------------------------------------------------
# Trust-weighted -- robust rule #3. alpha_ij comes from the defence's own
# score (per Sec 4: "trust-weighted alpha_ij") -- here taken as an input
# dict rather than computed, since the real trust score is a downstream
# property of P4's occupancy-consistency check / P5's retrospective-
# consistency signal, not something this module should invent. A neighbour
# with trust 0 is fully excluded; trust is renormalised (including the
# vehicle's own always-trust-1.0 term) before use.
# ---------------------------------------------------------------------------


def aggregate_trust_weighted(
    own_delta: np.ndarray,
    neighbour_deltas: Sequence[np.ndarray],
    neighbour_trust: Sequence[float],
) -> np.ndarray:
    assert len(neighbour_deltas) == len(neighbour_trust), (
        "aggregate_trust_weighted: one trust score required per neighbour delta"
    )
    all_deltas = np.stack([own_delta, *neighbour_deltas], axis=0)
    trust = np.asarray([1.0, *neighbour_trust], dtype=np.float64)
    if trust.sum() <= 0:
        # Degenerate case: trust every neighbour at exactly zero -> fall
        # back to "trust only myself" rather than dividing by zero.
        trust = np.zeros_like(trust)
        trust[0] = 1.0
    trust = trust / trust.sum()
    return np.tensordot(trust, all_deltas, axes=(0, 0))


RULES = {
    "average": aggregate_average,
    "trimmed_mean": aggregate_trimmed_mean,
    "median": aggregate_median,
    "trust_weighted": aggregate_trust_weighted,
}

ROBUST_RULES = ("trimmed_mean", "median", "trust_weighted")
