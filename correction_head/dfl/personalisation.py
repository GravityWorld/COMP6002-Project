"""Axis 4 -- personalisation: shared-only weights vs shared trunk +
per-vehicle local calibration layer.

task1_datasets_pipeline_dfl.md Sec 4: "a vehicle's own dominant sensing-
range/density profile is exactly the kind of local structure this axis
is meant to exploit" -- the calibration layer is where that per-vehicle
structure is allowed to live instead of being averaged away by gossip.

This module works on a flat parameter vector + a boolean mask (True =
calibration, kept local; False = shared, federated) so it is agnostic to
whatever the real parameter layout turns out to be -- `calibration_mask_
from_param_shapes()` below is the one function that needs to know the
real head's actual layer names, and it is written so it can be called
with `correction_head.model.CorrectionHead().state_dict()`'s shapes
directly once this is pointed at the real model (needs torch; not
exercised in this cloud sandbox -- see module-level NOTE).

NOTE -- execution status: everything except `calibration_mask_from_param_
shapes()` is pure numpy and IS execution-verified here (run_synthetic_
scaffold.py runs it for real). `calibration_mask_from_param_shapes()`'s
logic is simple (pick the last named parameter group) but has only been
exercised against a synthetic fake shape list in the test harness, never
against the real torch state_dict -- flag as designed, not yet run
against the real model, per the project's standing practice on
torch-dependent code.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def split_params(flat_params: np.ndarray, calib_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """flat_params[calib_mask] = calibration (local), the rest = shared
    (federated). Returns (shared, calib) as two 1-D arrays, in the order
    `flat_params[~mask]` / `flat_params[mask]` -- use `merge_params` with
    the SAME mask to invert this."""
    assert flat_params.shape == calib_mask.shape
    return flat_params[~calib_mask], flat_params[calib_mask]


def merge_params(shared: np.ndarray, calib: np.ndarray, calib_mask: np.ndarray) -> np.ndarray:
    """Inverse of split_params: reassembles a full flat vector from its
    shared and calibration pieces, placed back at their original
    positions per calib_mask."""
    out = np.empty(calib_mask.shape, dtype=np.result_type(shared, calib))
    out[~calib_mask] = shared
    out[calib_mask] = calib
    return out


def calibration_mask_from_param_shapes(
    named_numels: Sequence[Tuple[str, int]], calibration_layer_names: Sequence[str]
) -> np.ndarray:
    """Build the boolean mask over a FLATTENED concatenation of named
    parameters (same order as `named_numels`), marking every parameter
    whose name is in `calibration_layer_names` as calibration (local).

    For the real head, the intended calibration layer is
    `PairwiseAffinityHead`'s final `nn.Linear` (the last layer that maps
    the combined-embedding + pair-aux vector to one logit) -- everything
    upstream of it (FragmentEncoder's conv/GroupNorm stack, the first
    PairwiseAffinityHead linear layer) stays shared and federated. This
    keeps federation on the expensive, genuinely-shared feature-extraction
    machinery while letting each vehicle's own final decision boundary
    calibrate to its own range/density profile, per Sec 4's framing.
    Pass the real final-layer's parameter name(s) (e.g. from
    `[n for n, _ in model.named_parameters()]`) as `calibration_layer_names`
    once this is wired to the real model -- not done yet, see module NOTE.
    """
    total = sum(numel for _, numel in named_numels)
    mask = np.zeros(total, dtype=bool)
    offset = 0
    for name, numel in named_numels:
        if name in calibration_layer_names:
            mask[offset : offset + numel] = True
        offset += numel
    assert offset == total
    return mask


class PersonalisedVehicleState:
    """Per-vehicle bookkeeping: one shared vector (gets gossip-aggregated)
    and one calibration vector (never leaves the vehicle, never gets
    aggregated -- only ever updated by that vehicle's own local SGD step)."""

    def __init__(self, shared: np.ndarray, calib: np.ndarray):
        self.shared = shared
        self.calib = calib

    def full_params(self, calib_mask: np.ndarray) -> np.ndarray:
        return merge_params(self.shared, self.calib, calib_mask)
