"""Axis 5 -- communication-budget encode/decode (fp32 / int8 / 1-bit).

task1_datasets_pipeline_dfl.md Sec 4: "fp32 / int8 / 1-bit sign-based
Delta w, swept over rounds -> bytes-vs-FPR Pareto curve." This module
only does the encode/decode + byte-sizing half; the "vs-FPR" half is
produced by actually training/evaluating the head under each mode, which
happens once this is pointed at the real head (not here).

Byte-size sanity check against the project's own prior estimate
(handover.md / task1_datasets_pipeline_dfl.md: "~300K-500K param head
runs roughly ~1.2-2MB fp32, ~300-500KB int8, ~40-65KB 1-bit") is in
run_synthetic_scaffold.py, using the real pinned param count (388,098)
from correction_head/model.py's count_params.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EncodedPayload:
    mode: str
    n_bytes: int
    payload: object  # mode-specific: raw bytes, or a (bytes, scale, zero) tuple, etc.


def encode_fp32(delta: np.ndarray) -> EncodedPayload:
    raw = delta.astype(np.float32).tobytes()
    return EncodedPayload("fp32", len(raw), raw)


def decode_fp32(encoded: EncodedPayload, n_params: int) -> np.ndarray:
    arr = np.frombuffer(encoded.payload, dtype=np.float32)
    assert arr.size == n_params
    return arr.astype(np.float64)


def encode_int8(delta: np.ndarray) -> EncodedPayload:
    """Uniform affine quantisation to signed int8, scale chosen per-tensor
    from the delta's own max absolute value (symmetric, zero-point = 0 --
    simplest scheme that still round-trips a weight-update delta, which is
    typically ~zero-centred)."""
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantised = np.clip(np.round(delta / scale), -127, 127).astype(np.int8)
    raw = quantised.tobytes()
    # scale stored as one extra fp32 -- negligible at this param count, but
    # counted honestly rather than assumed away.
    n_bytes = len(raw) + 4
    return EncodedPayload("int8", n_bytes, (raw, scale))


def decode_int8(encoded: EncodedPayload, n_params: int) -> np.ndarray:
    raw, scale = encoded.payload
    quantised = np.frombuffer(raw, dtype=np.int8)
    assert quantised.size == n_params
    return quantised.astype(np.float64) * scale


def encode_1bit(delta: np.ndarray) -> EncodedPayload:
    """Sign-based 1-bit compression: ship only sign(delta) packed 8-to-a-
    byte, plus one scalar magnitude (mean absolute value) so decode can
    reconstruct a magnitude-preserving approximation rather than a pure
    +-1 vector. This is the standard sign-SGD-style compressor, not a
    hardcoded project-specific rule -- it generalises to any delta shape."""
    signs = (delta >= 0).astype(np.uint8)  # 1 = non-negative, 0 = negative
    packed = np.packbits(signs)
    magnitude = float(np.mean(np.abs(delta))) if delta.size else 0.0
    n_bytes = len(packed.tobytes()) + 4
    return EncodedPayload("1bit", n_bytes, (packed, magnitude, delta.size))


def decode_1bit(encoded: EncodedPayload, n_params: int) -> np.ndarray:
    packed, magnitude, orig_size = encoded.payload
    assert orig_size == n_params
    signs = np.unpackbits(packed)[:n_params]
    return np.where(signs == 1, magnitude, -magnitude)


ENCODERS = {"fp32": encode_fp32, "int8": encode_int8, "1bit": encode_1bit}
DECODERS = {"fp32": decode_fp32, "int8": decode_int8, "1bit": decode_1bit}


def encode(delta: np.ndarray, mode: str) -> EncodedPayload:
    return ENCODERS[mode](delta)


def decode(encoded: EncodedPayload, n_params: int) -> np.ndarray:
    return DECODERS[encoded.mode](encoded, n_params)


def bytes_for_mode(n_params: int, mode: str) -> int:
    """Theoretical byte size for a delta of this length under `mode`,
    without needing an actual array -- used to sanity-check against the
    project's prior written estimate at the real 388,098 param count."""
    if mode == "fp32":
        return n_params * 4
    if mode == "int8":
        return n_params * 1 + 4
    if mode == "1bit":
        return -(-n_params // 8) + 4  # ceil(n_params / 8) + 4-byte magnitude
    raise ValueError(f"unknown comms mode: {mode}")


def reconstruction_error(original: np.ndarray, mode: str) -> Tuple[float, int]:
    """Round-trips `original` through encode/decode for `mode` and returns
    (relative L2 error, actual encoded byte size)."""
    encoded = encode(original, mode)
    recovered = decode(encoded, original.size)
    denom = np.linalg.norm(original)
    rel_err = float(np.linalg.norm(recovered - original) / denom) if denom > 0 else 0.0
    return rel_err, encoded.n_bytes
