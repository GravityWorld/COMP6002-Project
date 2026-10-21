"""Axis 3 -- poisoning curve: f in {0, 10, 20, 30, 40}% adversarial vehicles.

task1_datasets_pipeline_dfl.md Sec 4 names the recursion directly: "you
are federating a repair model over an untrusted channel; a malicious car
can poison the model that decides what a 'correct' cluster looks like."
This module picks WHICH vehicles are malicious for a given f, and HOW a
malicious vehicle's delta gets corrupted before it's broadcast -- the
aggregation rules in aggregation.py are the measured DEFENCE, this is the
ATTACK half of that pairing.

Attacker model (documented, per execution_plan_daily.md's "attacker model
decided and documented"): a malicious vehicle sends a delta scaled to a
large, fixed-direction corruption rather than its real local update --
"large-norm, wrong-direction" is the worst case a norm-unaware rule like
plain average has no defence against, and it's exactly what trimmed-mean
/ median / trust-weighted are supposed to survive. A second, milder
"sign-flip" model is included too (send the negated real delta -- an
attacker that actively undoes consensus rather than a blunt-force outlier),
since a robust rule that only defends against large-norm outliers but not
a bounded-norm adversarial direction would be a weaker result than it
looks on the large-norm attack alone.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

F_VALUES = (0.0, 0.10, 0.20, 0.30, 0.40)


def select_malicious_vehicles(
    vehicle_ids: Sequence[str], f: float, rng: np.random.Generator
) -> List[str]:
    """Deterministic-given-rng choice of which vehicles are malicious at
    poisoning fraction f. Rounds to the nearest integer count; f=0 always
    returns an empty list regardless of rounding."""
    n = len(vehicle_ids)
    n_malicious = int(round(f * n))
    if f <= 0.0 or n_malicious <= 0:
        return []
    ids = list(vehicle_ids)
    rng.shuffle(ids)
    return ids[:n_malicious]


def attack_large_norm(
    honest_delta: np.ndarray, rng: np.random.Generator, scale: float = 25.0
) -> np.ndarray:
    """Blunt-force outlier: replace the real delta with a random direction
    scaled to `scale` times the honest delta's own norm -- large-norm,
    uncorrelated-direction corruption. Plain average has no defence
    against this; it's the headline failure case axis 3 is meant to show."""
    honest_norm = float(np.linalg.norm(honest_delta))
    if honest_norm == 0.0:
        honest_norm = 1.0
    direction = rng.normal(size=honest_delta.shape)
    direction_norm = np.linalg.norm(direction)
    direction = direction / direction_norm if direction_norm > 0 else direction
    return direction * (scale * honest_norm)


def attack_sign_flip(honest_delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bounded-norm, adversarial-direction corruption: negate the real
    update, so the attacker actively pulls consensus away from where
    honest vehicles are converging rather than just adding noise. A
    harder case for the robust rules than attack_large_norm, since it
    can't be caught by norm alone."""
    return -honest_delta


ATTACKS = {"large_norm": attack_large_norm, "sign_flip": attack_sign_flip}


def apply_poisoning(
    honest_deltas: Dict[str, np.ndarray],
    malicious_ids: Sequence[str],
    attack: str,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Returns a NEW dict, same keys as `honest_deltas`, with malicious
    vehicles' entries replaced by the attack's corrupted delta -- honest
    vehicles' deltas pass through untouched."""
    attack_fn = ATTACKS[attack]
    out = dict(honest_deltas)
    for vid in malicious_ids:
        if vid in out:
            out[vid] = attack_fn(out[vid], rng)
    return out
