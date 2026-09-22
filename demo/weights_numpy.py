"""NumPy port of boundary-aware Mean Teacher weights.

Matches ``semi-seg-ecg/src/utils/boundary_weights.py`` and the gates in
``algorithms/mean_teacher_boundary.py``. JS ``demo/js/weights.js`` is a
line-by-line port of these functions.
"""

from __future__ import annotations

import numpy as np


def max_pool1d_same(edge: np.ndarray, band: int) -> np.ndarray:
    """Zero-padded max-pool, kernel ``2 * band + 1``, stride 1.

    ``edge`` is ``(T,)`` or ``(B, T)``.
    """
    if band <= 0:
        return np.asarray(edge, dtype=np.float64)
    arr = np.asarray(edge, dtype=np.float64)
    k = 2 * band + 1
    if arr.ndim == 1:
        padded = np.pad(arr, (band, band), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, k)
        return windows.max(axis=-1)
    padded = np.pad(arr, ((0, 0), (band, band)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=-1)
    return windows.max(axis=-1)


def soft_boundary_weight(prob: np.ndarray, band: int = 4) -> np.ndarray:
    """Soft edge map from adjacent class-probability change.

    Args:
        prob: teacher probabilities ``(C, T)`` or ``(B, C, T)``.
        band: dilation half-width in samples.

    Returns:
        ``(T,)`` or ``(B, T)`` weights in ``[0, 1]``.
    """
    p = np.asarray(prob, dtype=np.float64)
    batched = p.ndim == 3
    if not batched:
        p = p[np.newaxis, ...]
    delta = 0.5 * np.abs(p[:, :, 1:] - p[:, :, :-1]).sum(axis=1)
    edge = np.pad(delta, ((0, 0), (1, 0)), mode="constant")
    edge = max_pool1d_same(edge, band)
    denom = np.maximum(edge.max(axis=-1, keepdims=True), 1e-6)
    out = np.clip(edge / denom, 0.0, 1.0)
    return out[0] if not batched else out


def region_boundary_weights(
    w_b: np.ndarray,
    conf: np.ndarray,
    conf_thresh: float = 0.80,
    sample_conf_thresh: float = 0.50,
    sample_keep: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Interior / boundary gates from teacher confidence.

    Returns ``(w_region, w_boundary, sample_keep)``.
    """
    w_b = np.asarray(w_b, dtype=np.float64).reshape(-1)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    if sample_keep is None:
        keep = bool(conf.mean() >= sample_conf_thresh)
    else:
        keep = bool(sample_keep)
    keep_f = 1.0 if keep else 0.0
    w_region = (1.0 - w_b) * (conf >= conf_thresh).astype(np.float64) * keep_f
    w_boundary = w_b * (0.5 + 0.5 * conf) * keep_f
    return w_region, w_boundary, keep


def ce_teacher(p_student: np.ndarray, p_teacher: np.ndarray) -> np.ndarray:
    """Per-timestep soft CE from two ``(C, T)`` probability arrays."""
    ps = np.clip(np.asarray(p_student, dtype=np.float64), 1e-8, 1.0)
    pt = np.asarray(p_teacher, dtype=np.float64)
    return -np.sum(pt * np.log(ps), axis=0)


def vanilla_mt_uniform(length: int) -> np.ndarray:
    """Vanilla Mean Teacher weighs every timestep equally (``1/T``)."""
    t = int(length)
    if t <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(t, 1.0 / t, dtype=np.float64)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    denom = float(w.sum())
    if denom < 1.0:
        denom = 1.0
    return float((v * w).sum() / denom)


def t_transition_indices(pred: np.ndarray) -> np.ndarray:
    """Sample indices where teacher argmax enters or leaves class T (3)."""
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    if pred.size < 2:
        return np.zeros(0, dtype=np.int64)
    onset = np.where((pred[1:] == 3) & (pred[:-1] != 3))[0] + 1
    offset = np.where((pred[:-1] == 3) & (pred[1:] != 3))[0]
    if onset.size == 0 and offset.size == 0:
        return onset.astype(np.int64)
    return np.concatenate([onset, offset]).astype(np.int64)


def best_t_index(pred: np.ndarray, w_b: np.ndarray) -> int:
    """T on/offset with highest ``w_b``; fallback to global ``w_b`` argmax."""
    w_b = np.asarray(w_b, dtype=np.float64).reshape(-1)
    idx = t_transition_indices(pred)
    if idx.size == 0:
        return int(np.argmax(w_b))
    return int(idx[np.argmax(w_b[idx])])
