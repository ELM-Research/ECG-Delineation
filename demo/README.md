# Boundary-aware Mean Teacher demo

Offline HTML explorer for the ResNet-18 + U-Net boundary-aware Mean Teacher
(LUDB 1/16). Open [`index.html`](index.html) in a browser. No GPU, no Python,
and no network at the venue.

## Two tabs, two claims

**Training step** is the method. Vanilla Mean Teacher is the same
teacher/student pair with uniform `1/T` weights. Boundary-aware MT adds `w_b`,
an interior confidence gate (`conf_thresh=0.80`), a soft edge gate, and
`sample_keep` (`mean conf >= 0.50`). Default view: T-wave on/offset zoom.
Seed-0 vs plain MT: T-onset MAE 16.8 → 15.2 ms — not the FCN→U-Net mIoU jump.

**Test comparison** is the full system on three LUDB windows from
`predictions_groundtruths/`. Supervised FCN vs supervised U-Net isolates the
decoder. FCN vs boundary-MT mixes decoder and training. Plain MT (0.8409) vs
boundary-MT (0.8449) is **not** in that export.

## Open the page

From this folder:

```bash
open index.html
# or
python3 -m http.server 8000
# then http://127.0.0.1:8000/
```

Data is embedded as `<script src="data/*.js">` so `file://` works. Smoke-test
with Wi-Fi off. If a browser blocks local scripts, use the http.server fallback.

## Data

`data/snapshots.js` is generated. This repo ships a **synthetic fixture** so the
UI works without weights. Replace it with a real dump when
`best-MeanIoU.pth` and LUDB waveforms are available (they are gitignored;
default path `baseline/exps/resnet18/mean_teacher_boundary_unet/ludb/1over16/best-MeanIoU.pth`):

```bash
# UI fixture (no checkpoint)
python demo/make_fixture.py

# Real student + EMA teacher (run from repo root; cwd switches into semi-seg-ecg)
python demo/dump_mt_demo.py \
  --config semi-seg-ecg/configs/base/resnet18/mean_teacher_boundary_unet.yaml \
  --override semi-seg-ecg/configs/bench/ludb/1over16.yaml \
  --checkpoint baseline/exps/resnet18/mean_teacher_boundary_unet/ludb/1over16/best-MeanIoU.pth \
  --device cpu
```

The dump needs the training conda env (`conda activate semi_seg_ecg`) with LUDB
waveforms under `semi-seg-ecg/data/ludb`. This workspace does not include the
checkpoint or those pickles. Until you copy them, keep the synthetic fixture.

Plotly is vendored at `vendor/plotly.min.js` (~4.5 MB) so the page works offline.
If that file is missing:

```bash
curl -fsSL -o demo/vendor/plotly.min.js \
  https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js
```

The dump is an **eval-mode** forward (no dropout, BN running stats). Labeled and
unlabeled panes are different recordings, as in training. Weak
`random_resize_crop` time-warps vs the raw LUDB file; the plot is that crop.
At `best-MeanIoU` student ≈ teacher, so unlabeled CE is often quiet; the
weight map is the method.

Sliders (`band`, `conf_thresh`) recompute display curves only. The stacked
`L_x | L_region | L_boundary` bar is the frozen training loss from the dump.

## Venue

Laptop, browser, power outlet. Optional second monitor. No GPU or internet.
Backup: this folder plus the representative PNGs under
`predictions_groundtruths/representative_examples/`.
