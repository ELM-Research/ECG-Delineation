#!/usr/bin/env python3
"""Build demo/data/snapshots.js (synthetic) and demo/data/test.js (LUDB export).

The training-step fixture is pedagogically shaped (T-onset, gate, disagreement)
and marked ``source: synthetic_fixture``. Replace it with a real dump:

    python demo/dump_mt_demo.py --checkpoint path/to/best-MeanIoU.pth
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from js_export import round_array, write_js
from weights_numpy import (
    best_t_index,
    ce_teacher,
    region_boundary_weights,
    soft_boundary_weight,
    weighted_mean,
)

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "predictions_groundtruths"
ECG_DIR = REPO / "semi-seg-ecg" / "data" / "ludb" / "ecg"
T = 2500
# Five clean strips, plus two readable failures (missed P, missed T)
# mixed into the New example cycle. Skip old Typical (aVR) and Hard (no P).
DEMO_CASE_INDICES = [432, 1, 85, 338, 220, 109, 158]
FS = 250
WAVE_TO_CLASS = {"P": 1, "QRS": 2, "T": 3}


def load_waveform(path: Path, target_length: int = 2500) -> np.ndarray:
    with path.open("rb") as handle:
        waveform = np.asarray(pickle.load(handle), dtype=np.float64).ravel()
    if len(waveform) != target_length:
        source_t = np.linspace(0, 1, len(waveform))
        target_t = np.linspace(0, 1, target_length)
        waveform = np.interp(target_t, source_t, waveform)
    return (waveform - waveform.mean()) / (waveform.std() + 1e-8)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=0, keepdims=True), 1e-8, None)


def synth_labels() -> np.ndarray:
    """~10 beats of P / QRS / T on a 10 s, 250 Hz window."""
    labels = np.zeros(T, dtype=np.int64)
    rr = 240
    for beat, start in enumerate(range(80, T - 200, rr)):
        p0, p1 = start, start + 28
        q0, q1 = start + 40, start + 62
        t0, t1 = start + 78, start + 150
        labels[p0:p1] = 1
        labels[q0:q1] = 2
        labels[t0:t1] = 3
        if beat == 0:
            pass
    return labels


def synth_ecg(labels: np.ndarray, noise: float = 0.04, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = np.zeros(T, dtype=np.float64)
    i = 0
    while i < T:
        c = int(labels[i])
        if c == 0:
            i += 1
            continue
        j = i
        while j < T and int(labels[j]) == c:
            j += 1
        mid = 0.5 * (i + j - 1)
        width = max(3.0, 0.25 * (j - i))
        amp = {1: 0.25, 2: 1.6, 3: 0.45}[c]
        x += amp * np.exp(-0.5 * ((t - mid) / width) ** 2)
        i = j
    x += noise * rng.normal(size=T)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x


def logits_from_labels(
    labels: np.ndarray,
    *,
    peak: float,
    blur: int,
    t_shift: int = 0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shifted = labels.copy()
    if t_shift:
        shifted = np.zeros_like(labels)
        idx = np.arange(T)
        src = np.clip(idx - t_shift, 0, T - 1)
        shifted = labels[src]
        # keep non-T from original except around T
        non_t = labels != 3
        shifted[non_t] = labels[non_t]
    logits = np.full((4, T), -peak * 0.25, dtype=np.float64)
    for c in range(4):
        logits[c] += peak * (shifted == c).astype(np.float64)
    if blur > 0:
        k = np.ones(2 * blur + 1) / (2 * blur + 1)
        for c in range(4):
            logits[c] = np.convolve(logits[c], k, mode="same")
    logits += 0.15 * rng.normal(size=logits.shape)
    return logits


def pack_unlabeled(
    ecg_weak: np.ndarray,
    ecg_strong: np.ndarray,
    p_teacher: np.ndarray,
    p_student: np.ndarray,
    conf_thresh: float = 0.80,
    sample_conf_thresh: float = 0.50,
    force_keep: bool | None = None,
) -> tuple[dict, dict, int, np.ndarray]:
    conf = p_teacher.max(axis=0)
    w_b = soft_boundary_weight(p_teacher, band=4)
    w_region, w_boundary, sample_keep = region_boundary_weights(
        w_b,
        conf,
        conf_thresh=conf_thresh,
        sample_conf_thresh=sample_conf_thresh,
        sample_keep=force_keep,
    )
    ce = ce_teacher(p_student, p_teacher)
    teacher_pred = p_teacher.argmax(axis=0)
    t_idx = best_t_index(teacher_pred, w_b)
    scalars = {
        "loss_u_region": round(weighted_mean(ce, w_region), 4),
        "loss_u_boundary": round(weighted_mean(ce, w_boundary), 4),
        "mean_conf": round(float(conf.mean()), 4),
        "loss_x": 0.0,
        "loss": 0.0,
    }
    unlabeled = {
        "ecg_weak": round_array(ecg_weak),
        "ecg_strong": round_array(ecg_strong),
        "p_teacher": round_array(p_teacher),
        "p_student": round_array(p_student),
        "conf": round_array(conf),
        "w_b": round_array(w_b),
        "sample_keep": bool(sample_keep),
    }
    return unlabeled, scalars, t_idx, ce


def labeled_pack(labels: np.ndarray, seed: int) -> dict:
    ecg = synth_ecg(labels, noise=0.03, seed=seed)
    logits = logits_from_labels(labels, peak=4.2, blur=2, seed=seed + 3)
    p_student = _softmax(logits)
    ce = ce_teacher(p_student, np.eye(4)[labels].T)
    return {
        "ecg": round_array(ecg),
        "gt": labels.astype(int).tolist(),
        "p_student": round_array(p_student),
        "loss_x": round(float(ce.mean()), 4),
    }


def build_snapshots() -> dict:
    labels = synth_labels()
    captions = {
        "t_onset": (
            "T-wave onset/offset on the unlabeled crop. Seed-0 vs plain MT: "
            "T-onset MAE 16.8 → 15.2 ms (not the FCN→U-Net mIoU jump)."
        ),
        "closest_to_gate": (
            "Closest to the sample gate: mean teacher confidence is low, so "
            "most unlabeled consistency is dropped."
        ),
        "boundary_disagreement": (
            "Student T onset is shifted vs the teacher. Vanilla MT would "
            "spread that CE over the whole clip; w_boundary concentrates it."
        ),
        "typical": (
            "Typical kept crop: student ≈ teacher. At best-MeanIoU the CE "
            "curve is quiet; the weight map is the method."
        ),
        "rejected": (
            "Clip rejected: mean teacher confidence below 0.50. Unlabeled "
            "consistency is zeroed."
        ),
    }

    def step(
        role: str,
        *,
        peak_t: float,
        peak_s: float,
        t_shift: int,
        strong_noise: float,
        force_keep: bool | None,
        seed: int,
        unlabeled_file: str,
        labeled_file: str,
        patient_id: int,
        lead: str,
    ) -> dict:
        weak = synth_ecg(labels, noise=0.035, seed=seed)
        strong = synth_ecg(labels, noise=strong_noise, seed=seed + 1)
        p_teacher = _softmax(
            logits_from_labels(labels, peak=peak_t, blur=2, seed=seed + 2)
        )
        p_student = _softmax(
            logits_from_labels(
                labels, peak=peak_s, blur=3, t_shift=t_shift, seed=seed + 4
            )
        )
        unlabeled, scalars, t_idx, _ce = pack_unlabeled(
            weak,
            strong,
            p_teacher,
            p_student,
            force_keep=force_keep,
        )
        if role == "t_onset":
            onsets = np.where((labels[1:] == 3) & (labels[:-1] != 3))[0] + 1
            if onsets.size:
                t_idx = int(onsets[0])
        lab = labeled_pack(labels, seed=seed + 9)
        scalars["loss_x"] = lab["loss_x"]
        scalars["loss"] = round(0.5 * (scalars["loss_x"] + scalars["loss_u_region"] + scalars["loss_u_boundary"]), 4)
        pid, lead_name = patient_id, lead
        return {
            "id": role,
            "role": role,
            "caption": captions[role],
            "seed": 0,
            "source": "synthetic_fixture",
            "unlabeled_file": unlabeled_file,
            "labeled_file": labeled_file,
            "patient_id": pid,
            "lead": lead_name,
            "labeled_patient_id": 12,
            "labeled_lead": "II",
            "t_onset_index": int(t_idx),
            "unlabeled": unlabeled,
            "labeled": lab,
            "scalars": scalars,
        }

    steps = [
        step(
            "t_onset",
            peak_t=4.8,
            peak_s=4.2,
            t_shift=6,
            strong_noise=0.12,
            force_keep=True,
            seed=1,
            unlabeled_file="77_lead_V4.pkl",
            labeled_file="12_lead_II.pkl",
            patient_id=77,
            lead="V4",
        ),
        step(
            "boundary_disagreement",
            peak_t=4.5,
            peak_s=3.2,
            t_shift=14,
            strong_noise=0.18,
            force_keep=True,
            seed=2,
            unlabeled_file="150_lead_aVR.pkl",
            labeled_file="12_lead_II.pkl",
            patient_id=150,
            lead="aVR",
        ),
        step(
            "typical",
            peak_t=5.0,
            peak_s=4.9,
            t_shift=1,
            strong_noise=0.08,
            force_keep=True,
            seed=3,
            unlabeled_file="41_lead_II.pkl",
            labeled_file="12_lead_II.pkl",
            patient_id=41,
            lead="II",
        ),
        step(
            "closest_to_gate",
            peak_t=1.35,
            peak_s=1.1,
            t_shift=4,
            strong_noise=0.22,
            force_keep=None,
            seed=4,
            unlabeled_file="103_lead_III.pkl",
            labeled_file="12_lead_II.pkl",
            patient_id=103,
            lead="III",
        ),
        step(
            "rejected",
            peak_t=0.55,
            peak_s=0.5,
            t_shift=8,
            strong_noise=0.3,
            force_keep=False,
            seed=5,
            unlabeled_file="88_lead_V1.pkl",
            labeled_file="12_lead_II.pkl",
            patient_id=88,
            lead="V1",
        ),
    ]
    return {
        "source": "synthetic_fixture",
        "checkpoint": None,
        "hyps": {
            "boundary_band": 4,
            "conf_thresh": 0.80,
            "sample_conf_thresh": 0.50,
            "lambda_region": 1.0,
            "lambda_boundary": 1.0,
            "ema_decay": 0.99,
        },
        "note": (
            "Synthetic fixture for UI development. Replace with "
            "python demo/dump_mt_demo.py once best-MeanIoU.pth is available. "
            "Labeled and unlabeled panes are different recordings. "
            "Weak crop is the plotted window (not the raw LUDB file)."
        ),
        "steps": steps,
    }


def _sample_mean_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    class_ious = []
    for class_id in range(4):
        intersection = np.sum((pred == class_id) & (gt == class_id))
        union = np.sum((pred == class_id) | (gt == class_id))
        if union:
            class_ious.append(intersection / union)
    return float(np.mean(class_ious)) if class_ious else 0.0


def _pan_tompkins_intervals(ecg: np.ndarray, fs: int = FS) -> list[dict]:
    """Match demo/js/pan_tompkins.js: R peaks, then fixed P/QRS/T windows."""
    lead = np.asarray(ecg, dtype=np.float64).ravel()
    n = len(lead)
    if n == 0:
        return []

    def ms(t: float) -> int:
        return int(round(t * fs))

    d = np.zeros(n, dtype=np.float64)
    for i in range(2, n - 2):
        d[i] = (-lead[i - 2] - 2 * lead[i - 1] + 2 * lead[i + 1] + lead[i + 2]) / 8

    w = max(1, ms(0.15))
    mwi = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        acc += d[i] * d[i]
        if i >= w:
            acc -= d[i - w] * d[i - w]
        mwi[i] = acc / w

    thr = 0.3 * float(mwi.max()) if n else 0.0
    refr = ms(0.2)
    r_peaks: list[int] = []
    for i in range(1, n - 1):
        if (
            mwi[i] > thr
            and mwi[i] >= mwi[i - 1]
            and mwi[i] > mwi[i + 1]
            and (not r_peaks or i - r_peaks[-1] > refr)
        ):
            a = max(0, i - ms(0.13))
            b = min(n - 1, i + ms(0.02))
            r_peaks.append(int(a + np.argmax(np.abs(lead[a : b + 1]))))

    def clip(x: int) -> int:
        return max(0, min(n - 1, x))

    out: list[dict] = []
    for r in r_peaks:
        half_qrs = ms(0.045)
        out.append({"wave": "QRS", "start": clip(r - half_qrs), "end": clip(r + half_qrs)})
        if r - ms(0.08) > 0:
            a = r - ms(0.25)
            b = r - ms(0.08)
            c = int(a + np.argmax(np.abs(lead[a : b + 1])))
            half = ms(0.04)
            out.append({"wave": "P", "start": clip(c - half), "end": clip(c + half)})
        if r + ms(0.36) < n:
            a = r + ms(0.10)
            b = r + ms(0.36)
            c = int(a + np.argmax(np.abs(lead[a : b + 1])))
            half = ms(0.06)
            out.append({"wave": "T", "start": clip(c - half), "end": clip(c + half)})
    return out


def _intervals_to_labels(intervals: list[dict], length: int) -> np.ndarray:
    # Paint P, then T, then QRS so QRS wins overlaps (exclusive labels for mIoU).
    labels = np.zeros(length, dtype=np.int64)
    for wave in ("P", "T", "QRS"):
        class_id = WAVE_TO_CLASS[wave]
        for iv in intervals:
            if iv["wave"] != wave:
                continue
            a = min(iv["start"], iv["end"])
            b = max(iv["start"], iv["end"])
            labels[a : b + 1] = class_id
    return labels


def build_test_cases() -> dict:
    samples_path = EXPORT / "source_metadata" / "test_samples.csv"
    with samples_path.open(encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    gt_all = np.load(EXPORT / "ground_truth_labels.npy")
    preds = {
        "supervised_fcn": np.load(EXPORT / "predicted_labels" / "supervised_fcn.npy"),
        "supervised_unet": np.load(EXPORT / "predicted_labels" / "supervised_unet.npy"),
        "boundary_mt_unet": np.load(
            EXPORT / "predicted_labels" / "boundary_mt_unet.npy"
        ),
    }
    cases = []
    for index in DEMO_CASE_INDICES:
        sample = samples[index]
        waveform_name = sample["waveform"]
        gt = np.asarray(gt_all[index], dtype=np.int64)
        pred = {name: np.asarray(arr[index], dtype=np.int64) for name, arr in preds.items()}
        ecg = load_waveform(ECG_DIR / waveform_name)
        rules_labels = _intervals_to_labels(_pan_tompkins_intervals(ecg), len(ecg))
        sample_mean_iou = {name: _sample_mean_iou(pred[name], gt) for name in pred}
        sample_mean_iou["rules"] = _sample_mean_iou(rules_labels, gt)
        cases.append(
            {
                "id": Path(waveform_name).stem,
                "selection": "demo",
                "test_index": index,
                "waveform": waveform_name,
                "patient_id": int(sample["ID"]),
                "sample_mean_iou": sample_mean_iou,
                "ecg": round_array(ecg),
                "gt": gt.tolist(),
                "supervised_fcn": pred["supervised_fcn"].tolist(),
                "supervised_unet": pred["supervised_unet"].tolist(),
                "boundary_mt_unet": pred["boundary_mt_unet"].tolist(),
            }
        )
    return {
        "source": "predictions_groundtruths",
        "dataset": "LUDB",
        "label_fraction": "1/16",
        "models": {
            "supervised_fcn": {
                "label": "Supervised FCN",
                "test_mean_iou": 0.6661,
            },
            "supervised_unet": {
                "label": "Supervised U-Net",
                "test_mean_iou": 0.8204,
            },
            "boundary_mt_unet": {
                "label": "U-Net + boundary-MT",
                "test_mean_iou": 0.8449,
            },
        },
        "plain_mt_test_mean_iou": 0.8409,
        "claims": [
            "Supervised FCN vs supervised U-Net isolates the decoder.",
            "FCN vs boundary-MT U-Net mixes decoder and training.",
            "Boundary-MT vs plain MT is +0.004 mIoU (0.8449 vs 0.8409) and is not in this export.",
        ],
        "cases": cases,
    }


def main() -> None:
    snapshots = build_snapshots()
    write_js(DEMO / "data" / "snapshots.js", "MT_SNAPSHOTS", snapshots)
    test = build_test_cases()
    write_js(DEMO / "data" / "test.js", "TEST_CASES", test)
    print(f"Wrote {len(snapshots['steps'])} fixture steps")
    print(f"Wrote {len(test['cases'])} test cases")


if __name__ == "__main__":
    main()
