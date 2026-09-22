#!/usr/bin/env python3
"""Dump frozen boundary-aware Mean Teacher steps for the HTML demo.

Loads student (``model``) and EMA teacher (``model_ema``) from a checkpoint,
replays the unlabeled weak/strong + labeled CE step in eval mode, and writes
``demo/data/snapshots.js``.

Example (from repo root, after LUDB data and the checkpoint are available):

    python demo/dump_mt_demo.py \\
      --config semi-seg-ecg/configs/base/resnet18/mean_teacher_boundary_unet.yaml \\
      --override semi-seg-ecg/configs/bench/ludb/1over16.yaml \\
      --checkpoint baseline/exps/resnet18/mean_teacher_boundary_unet/ludb/1over16/best-MeanIoU.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "semi-seg-ecg" / "src"
sys.path.insert(0, str(REPO))

from demo.js_export import round_array, write_js  # noqa: E402
from demo.select_steps import parse_ludb_name, select_records  # noqa: E402
from demo.weights_numpy import (  # noqa: E402
    best_t_index,
    region_boundary_weights,
    weighted_mean,
)

# Filled by _import_torch_stack() after the checkpoint exists.
torch = None
F = None
init_model_from_cfg = None
soft_boundary_weight = None
build_seg_dataset = None

DEFAULT_CHECKPOINT = (
    REPO
    / "baseline"
    / "exps"
    / "resnet18"
    / "mean_teacher_boundary_unet"
    / "ludb"
    / "1over16"
    / "best-MeanIoU.pth"
)

MISSING_CKPT = """Checkpoint not found:
  {path}

LUDB waveforms are already on this machine. Copy the trained weights from gpu2
(no GPU needed after that; dump runs on CPU):

  scp gpu2:~/ECG-SEG/baseline/exps/resnet18/mean_teacher_boundary_unet/ludb/1over16/best-MeanIoU.pth \\
      baseline/exps/resnet18/mean_teacher_boundary_unet/ludb/1over16/best-MeanIoU.pth

Then, in the training env:

  conda activate semi_seg_ecg
  python demo/dump_mt_demo.py --checkpoint baseline/exps/resnet18/mean_teacher_boundary_unet/ludb/1over16/best-MeanIoU.pth --device cpu

Until then the HTML demo uses the synthetic fixture from:
  python demo/make_fixture.py
"""


def _shim_torch_six() -> None:
    """SemiSegECG was written against torch 1.11 (`torch._six.inf`)."""
    import types

    if "torch._six" not in sys.modules:
        six = types.ModuleType("torch._six")
        six.inf = float("inf")
        sys.modules["torch._six"] = six


def _init_model_from_cfg(config: dict, train: bool = False):
    import torch.nn as nn
    import models.backbones as backbones
    import models.decode_heads as decode_heads
    from models.encoder_decoder import EncoderDecoder

    backbone_name, backbone_kwargs = list(config["backbone"].items())[0]
    backbone = backbones.__dict__[backbone_name](**backbone_kwargs)
    decoder_name, decoder_kwargs = list(config["decode_head"].items())[0]
    decoder = decode_heads.__dict__[decoder_name](**decoder_kwargs)
    return EncoderDecoder(
        backbone=backbone,
        decode_head=decoder,
        decode_head_loss=nn.CrossEntropyLoss(),
        auxiliary_heads=None,
        auxiliary_head_losses=None,
    )


def _import_torch_stack() -> None:
    """Load training-stack imports only after the checkpoint is present."""
    global torch, F, init_model_from_cfg, soft_boundary_weight, build_seg_dataset
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    try:
        _shim_torch_six()
        import torch as _torch
        import torch.nn.functional as _F
        from utils.boundary_weights import soft_boundary_weight as _sbw
        from utils.semi_dataset import build_seg_dataset as _ds
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing training dependency ({exc}).\n"
            "Activate the env used to train:\n"
            "  conda activate semi_seg_ecg\n"
            "This Mac's base Python does not have tensorboard / the SemiSegECG stack."
        ) from exc
    torch = _torch
    F = _F
    init_model_from_cfg = _init_model_from_cfg
    soft_boundary_weight = _sbw
    build_seg_dataset = _ds


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_dataset_dirs(cfg: dict, semi_root: Path) -> None:
    """Resolve LUDB paths for this clone (data lives under semi-seg-ecg/)."""
    ds = cfg["dataset"]
    mapping = {
        "ecg_dir": "data/ludb/ecg",
        "label_dir": "data/ludb/label",
        "index_dir": "index/ludb",
    }
    for key, fallback in mapping.items():
        raw = Path(ds[key])
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((semi_root / raw).resolve())
            candidates.append((semi_root / fallback).resolve())
            candidates.append((semi_root.parent / raw).resolve())
        found = next((path for path in candidates if path.exists()), None)
        if found is None:
            tried = "\n  ".join(str(path) for path in candidates)
            raise SystemExit(f"Missing dataset.{key}. Tried:\n  {tried}")
        ds[key] = str(found)


def load_config(config_path: Path, override_path: Path | None) -> dict:
    with open(config_path) as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    if override_path is not None:
        with open(override_path) as handle:
            override = yaml.load(handle, Loader=yaml.FullLoader)
        cfg = deep_merge(cfg, override)
    return cfg


def strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if any(key.startswith("module.") for key in state):
        return {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def to_numpy_ecg(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    prob = logits.detach().float().softmax(dim=1)[0]
    return prob.cpu().numpy().astype(np.float64)


def run_step(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    labeled: dict,
    unlabeled: dict,
    device: torch.device,
    *,
    boundary_band: int,
    conf_thresh: float,
    sample_conf_thresh: float,
    lambda_region: float,
    lambda_boundary: float,
) -> dict:
    ecg_x = labeled["ecg"].unsqueeze(0).to(device)
    mask_x = labeled["target"].unsqueeze(0).to(device)
    if mask_x.dim() == 3:
        mask_x = mask_x.squeeze(1)
    ecg_u_w = unlabeled["ecg"].unsqueeze(0).to(device)
    ecg_u_s = unlabeled["ecg_aug"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_u_w = teacher(ecg_u_w, return_loss=False)["seg_logits"]
        prob_u_w = pred_u_w.softmax(dim=1)
        conf_u_w = prob_u_w.max(dim=1).values
        w_b_t = soft_boundary_weight(prob_u_w, band=boundary_band)
        pred_x = student(ecg_x, return_loss=False)["seg_logits"]
        pred_u_s = student(ecg_u_s, return_loss=False)["seg_logits"]
        loss_x = F.cross_entropy(pred_x, mask_x)
        ce = F.cross_entropy(pred_u_s, prob_u_w, reduction="none")[0]

    p_teacher = softmax_np(pred_u_w)
    p_student = softmax_np(pred_u_s)
    p_labeled = softmax_np(pred_x)
    conf = conf_u_w[0].detach().cpu().numpy().astype(np.float64)
    w_b = w_b_t[0].detach().cpu().numpy().astype(np.float64)
    ce_np = ce.detach().cpu().numpy().astype(np.float64)
    w_region, w_boundary, sample_keep = region_boundary_weights(
        w_b,
        conf,
        conf_thresh=conf_thresh,
        sample_conf_thresh=sample_conf_thresh,
    )
    loss_u_region = weighted_mean(ce_np, w_region)
    loss_u_boundary = weighted_mean(ce_np, w_boundary)
    loss_u = lambda_region * loss_u_region + lambda_boundary * loss_u_boundary
    loss = 0.5 * (float(loss_x.item()) + loss_u)

    gt = mask_x[0].detach().cpu().numpy().astype(np.int64)
    teacher_pred = p_teacher.argmax(axis=0).astype(np.int64)
    t_onset_index = best_t_index(teacher_pred, w_b)
    mean_ce_wb = float((ce_np * w_boundary).mean())

    return {
        "ecg_weak": to_numpy_ecg(unlabeled["ecg"]),
        "ecg_strong": to_numpy_ecg(unlabeled["ecg_aug"]),
        "ecg_labeled": to_numpy_ecg(labeled["ecg"]),
        "gt": gt,
        "p_teacher": p_teacher,
        "p_student": p_student,
        "p_labeled": p_labeled,
        "conf": conf,
        "w_b": w_b,
        "ce": ce_np,
        "sample_keep": sample_keep,
        "mean_conf": float(conf.mean()),
        "t_onset_index": t_onset_index,
        "t_score": float(w_b[t_onset_index]),
        "mean_ce_wb": mean_ce_wb,
        "loss_x": float(loss_x.item()),
        "loss_u_region": loss_u_region,
        "loss_u_boundary": loss_u_boundary,
        "loss": loss,
    }


def pack_step(rec: dict, unlabeled_file: str, labeled_file: str, seed: int) -> dict:
    pid, lead = parse_ludb_name(unlabeled_file)
    labeled_pid, labeled_lead = parse_ludb_name(labeled_file)
    role = rec.get("role", "diverse")
    captions = {
        "t_onset": (
            "T-wave onset/offset on the unlabeled crop. Seed-0 vs plain MT: "
            "T-onset MAE 16.8 → 15.2 ms (not the FCN→U-Net mIoU jump)."
        ),
        "rejected": (
            "Clip rejected: mean teacher confidence below sample_conf_thresh "
            "(0.50). Unlabeled consistency is zeroed."
        ),
        "closest_to_gate": (
            "Closest to the sample gate: no clip in this scan fell below "
            "mean conf 0.50, so this is the lowest-confidence unlabeled window."
        ),
        "boundary_disagreement": (
            "Highest mean(CE × w_boundary). At best-MeanIoU the student is "
            "close to the teacher; remaining error is concentrated on edges."
        ),
        "typical": "Typical kept unlabeled crop (median mean confidence).",
        "diverse": "Additional kept crop.",
    }
    return {
        "id": role,
        "role": role,
        "caption": captions.get(role, captions["diverse"]),
        "seed": seed,
        "source": "checkpoint",
        "unlabeled_file": unlabeled_file,
        "labeled_file": labeled_file,
        "patient_id": pid,
        "lead": lead,
        "labeled_patient_id": labeled_pid,
        "labeled_lead": labeled_lead,
        "t_onset_index": int(rec["t_onset_index"]),
        "unlabeled": {
            "ecg_weak": round_array(rec["ecg_weak"]),
            "ecg_strong": round_array(rec["ecg_strong"]),
            "p_teacher": round_array(rec["p_teacher"]),
            "p_student": round_array(rec["p_student"]),
            "conf": round_array(rec["conf"]),
            "w_b": round_array(rec["w_b"]),
            "sample_keep": bool(rec["sample_keep"]),
        },
        "labeled": {
            "ecg": round_array(rec["ecg_labeled"]),
            "gt": np.asarray(rec["gt"], dtype=np.int64).tolist(),
            "p_student": round_array(rec["p_labeled"]),
            "loss_x": round(float(rec["loss_x"]), 4),
        },
        "scalars": {
            "loss_u_region": round(float(rec["loss_u_region"]), 4),
            "loss_u_boundary": round(float(rec["loss_u_boundary"]), 4),
            "mean_conf": round(float(rec["mean_conf"]), 4),
            "loss": round(float(rec["loss"]), 4),
            "loss_x": round(float(rec["loss_x"]), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump MT demo snapshots")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO
        / "semi-seg-ecg"
        / "configs"
        / "base"
        / "resnet18"
        / "mean_teacher_boundary_unet.yaml",
    )
    parser.add_argument(
        "--override",
        type=Path,
        default=REPO / "semi-seg-ecg" / "configs" / "bench" / "ludb" / "1over16.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "demo" / "data" / "snapshots.js",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan", type=int, default=256)
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.override = args.override.resolve() if args.override else None
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()

    if not args.checkpoint.is_file():
        raise SystemExit(MISSING_CKPT.format(path=args.checkpoint))

    _import_torch_stack()

    cfg = load_config(args.config, args.override)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    semi_root = REPO / "semi-seg-ecg"
    resolve_dataset_dirs(cfg, semi_root)
    os.chdir(semi_root)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit("Expected a dict checkpoint with a 'model' key")
    student = init_model_from_cfg(cfg, train=False)
    teacher = init_model_from_cfg(cfg, train=False)
    student.load_state_dict(strip_module_prefix(ckpt["model"]), strict=True)
    ema = ckpt.get("model_ema")
    if ema is None:
        print("Warning: no model_ema in checkpoint; cloning student weights")
        ema = ckpt["model"]
    teacher.load_state_dict(strip_module_prefix(ema), strict=True)
    student.to(device).eval()
    teacher.to(device).eval()

    train_cfg = cfg.get("train", {})
    boundary_band = int(train_cfg.get("boundary_band", 4))
    conf_thresh = float(train_cfg.get("conf_thresh", 0.80))
    sample_conf_thresh = float(train_cfg.get("sample_conf_thresh", 0.50))
    lambda_region = float(train_cfg.get("lambda_region", 1.0))
    lambda_boundary = float(train_cfg.get("lambda_boundary", 1.0))

    labeled_ds = build_seg_dataset(cfg["dataset"], split="train_labeled", mode="train")
    unlabeled_ds = build_seg_dataset(
        cfg["dataset"], split="train_unlabeled", mode="train"
    )
    n = min(args.scan, len(labeled_ds), len(unlabeled_ds))
    print(f"Scanning {n} zipped train pairs on {device}")

    records: list[dict] = []
    for index in range(n):
        np.random.seed(args.seed * 100_000 + index)
        torch.manual_seed(args.seed * 100_000 + index)
        labeled = labeled_ds[index]
        unlabeled = unlabeled_ds[index]
        rec = run_step(
            student,
            teacher,
            labeled,
            unlabeled,
            device,
            boundary_band=boundary_band,
            conf_thresh=conf_thresh,
            sample_conf_thresh=sample_conf_thresh,
            lambda_region=lambda_region,
            lambda_boundary=lambda_boundary,
        )
        rec["unlabeled_file"] = unlabeled_ds.filenames[index]
        rec["labeled_file"] = labeled_ds.filenames[index]
        rec["_index"] = index
        records.append(rec)
        if (index + 1) % 32 == 0:
            print(f"  {index + 1}/{n}")

    selected = select_records(records, args.keep)
    steps = [
        pack_step(
            rec,
            rec["unlabeled_file"],
            rec["labeled_file"],
            args.seed,
        )
        for rec in selected
    ]
    payload = {
        "source": "checkpoint",
        "checkpoint": str(args.checkpoint),
        "hyps": {
            "boundary_band": boundary_band,
            "conf_thresh": conf_thresh,
            "sample_conf_thresh": sample_conf_thresh,
            "lambda_region": lambda_region,
            "lambda_boundary": lambda_boundary,
            "ema_decay": 0.99,
        },
        "note": (
            "Frozen eval-mode forward of student + EMA teacher. "
            "Labeled and unlabeled panes are different recordings. "
            "Weak random_resize_crop time-warps vs the raw LUDB file. "
            "At best-MeanIoU student ≈ teacher; the weight map is the method."
        ),
        "steps": steps,
    }
    write_js(args.out, "MT_SNAPSHOTS", payload)
    print(f"Wrote {len(steps)} steps to {args.out}")
    for step in steps:
        print(
            f"  {step['id']:24s} keep={step['unlabeled']['sample_keep']} "
            f"conf={step['scalars']['mean_conf']:.3f} "
            f"{step['unlabeled_file']}"
        )


if __name__ == "__main__":
    main()
