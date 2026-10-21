"""One gossip round, wiring axes 1/2/3/4/5 together.

Implements the 5-step spec from task1_datasets_pipeline_dfl.md Sec 4
verbatim:
  1. buffer ~200 frames -> local SGD steps -> Delta w_i, norm-clipped
  2. broadcast (Delta w_i, n_i, meta{...})
  3. w_i <- w_i + sum_j alpha_ij * Delta w_j   (aggregation.py, axis 2)
  4. aggregated per the axis-2 rule in force
  5. proximal anchor toward the centrally pre-trained head bounds drift

`local_update_fn` is injected rather than hardcoded, so this module has no
dependency on torch or the real dataset -- the synthetic test harness
passes a toy quadratic-loss local step; pointing this at the real head
later means passing a function that does `n` real local SGD steps on that
vehicle's buffered frames and returns the resulting flattened Delta w.
That swap is the ONLY change needed to move this module from synthetic
rows to real ones -- everything else (topology/aggregation/poisoning/
comms/personalisation wiring) is identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from . import comms as comms_mod
from .aggregation import RULES


@dataclass
class RoundDiagnostics:
    bytes_sent: Dict[str, int] = field(default_factory=dict)
    malicious_ids: List[str] = field(default_factory=list)
    attack_used: Optional[str] = None
    comms_mode: str = "fp32"
    aggregation_rule: str = "average"


def clip_norm(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm or norm == 0.0:
        return delta
    return delta * (max_norm / norm)


def gossip_round(
    weights: Dict[str, np.ndarray],
    adjacency: Dict[str, Set[str]],
    local_update_fn: Callable[[str, np.ndarray], np.ndarray],
    aggregation_rule: str = "average",
    comms_mode: str = "fp32",
    max_delta_norm: Optional[float] = None,
    malicious_ids: Sequence[str] = (),
    attack: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    neighbour_trust: Optional[Dict[str, Dict[str, float]]] = None,
    proximal_anchor: Optional[np.ndarray] = None,
    proximal_strength: float = 0.0,
) -> "tuple[Dict[str, np.ndarray], RoundDiagnostics]":
    """Runs one round for every vehicle in `weights`, returns (new_weights,
    diagnostics). `neighbour_trust[v][u]` is only required when
    aggregation_rule == 'trust_weighted'."""
    if rng is None:
        rng = np.random.default_rng()
    diag = RoundDiagnostics(
        malicious_ids=list(malicious_ids), attack_used=attack,
        comms_mode=comms_mode, aggregation_rule=aggregation_rule,
    )

    # Step 1: local SGD -> honest Delta w_i, norm-clipped.
    honest_deltas: Dict[str, np.ndarray] = {}
    for v, w in weights.items():
        delta = local_update_fn(v, w)
        if max_delta_norm is not None:
            delta = clip_norm(delta, max_delta_norm)
        honest_deltas[v] = delta

    # Poisoning (axis 3): malicious vehicles broadcast a corrupted delta
    # instead of their real one.
    if malicious_ids and attack is not None:
        from .poisoning import apply_poisoning
        broadcast_deltas = apply_poisoning(honest_deltas, malicious_ids, attack, rng)
    else:
        broadcast_deltas = dict(honest_deltas)

    # Step 2: broadcast over the comms channel (axis 5) -- every vehicle's
    # delta is encoded once, decoded once per neighbour that receives it.
    encoded: Dict[str, comms_mod.EncodedPayload] = {}
    for v, delta in broadcast_deltas.items():
        payload = comms_mod.encode(delta, comms_mode)
        encoded[v] = payload
        diag.bytes_sent[v] = payload.n_bytes

    def received_from(u: str) -> np.ndarray:
        n_params = broadcast_deltas[u].size
        return comms_mod.decode(encoded[u], n_params)

    # Steps 3-4: aggregate per the axis-2 rule in force. A vehicle's own
    # delta is used exact (never round-tripped through the comms channel
    # -- it never left the vehicle), neighbours' deltas are the decoded,
    # possibly-poisoned, possibly-quantised versions.
    new_weights: Dict[str, np.ndarray] = {}
    rule_fn = RULES[aggregation_rule]
    for v, w in weights.items():
        own_delta = honest_deltas[v]  # own update is always honest/exact
        neighbours = sorted(adjacency.get(v, set()))
        neighbour_deltas = [received_from(u) for u in neighbours]

        if aggregation_rule == "trust_weighted":
            trust_for_v = neighbour_trust.get(v, {}) if neighbour_trust else {}
            trust_scores = [trust_for_v.get(u, 1.0) for u in neighbours]
            aggregated = rule_fn(own_delta, neighbour_deltas, trust_scores)
        else:
            aggregated = rule_fn(own_delta, neighbour_deltas)

        updated = w + aggregated
        # Step 5: proximal anchor toward the centrally pre-trained head.
        if proximal_anchor is not None and proximal_strength > 0.0:
            updated = updated - proximal_strength * (updated - proximal_anchor)
        new_weights[v] = updated

    return new_weights, diag
