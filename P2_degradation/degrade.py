"""
P2 — Degradation injection (fog + compression, 2 severities each).

Scheduled for Wed 9 / Thu 10 Sep (Week 2), not today — this is a placeholder
so the directory exists and W3's desk work has a stable import path to build
against, per the schema-freeze discipline in scope_and_timeline_oct19.md.

Calibrate fog params against Adver-City's PUBLISHED statistics — Adver-City
itself is not downloaded (cut from scope, parameter source only). Record
which table/figure each parameter came from; a reviewer will ask.
"""

import numpy as np


def inject_fog(points: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Range-dependent dropout + attenuation + near-sensor backscatter."""
    raise NotImplementedError("Week 2 — Wed 9 Sep task.")


def inject_compression(bev_tensor: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Voxel / BEV-tensor quantisation."""
    raise NotImplementedError("Week 2 — Thu 10 Sep task.")
