<h2 align="center">
  Decoder Design Matters for ECG Delineation
</h2>

<div align="center">
  <img src="./assets/R-U-Net.png" alt="Our pipeline." width=600>
</div>


# Overview
Official Implementation of R-U-Net as proposed in [DECODER DESIGN MATTERS FOR ECG DELINEATION by Joseph Scharpf, William Han, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao](https://arxiv.org/abs/2609.16489).

This repository is largely a thin wrapper around the [SemiSegECG](https://dl.acm.org/doi/10.1145/3746252.3760790).
Currently, we plan to constrain the repository's features to only new models and SSL methods we develop within the SemiSegECG framework.

We provide a simple demonstration of R-U-Net's performance compared against a rules-based method (Pan Tompkins) and human delineation annotations. Feel free to check it out by opening the `demo/index.html` file or checking out [this link](https://ecg-seg-demo.vercel.app/)! All predictions are precomputed.

We plan to release model checkpoints after paper acceptance!

Please feel free to contribute to the repository! If there are any questions or bugs, please do not hesitate to reach out to wjhan{@}andrew{dot}cmu{edu} or submit an issue with corresponding details.

## Installation

1. Clone the repo, `cd` into it.

2. `git submodule update --init --recursive`

3. `uv sync`

4. `source .venv/bin/activate`

## Data Preparation

1. Download the in-domain and cross-domain waveforms and splits:

```bash
bash scripts/setup_data.sh
```

## Result Reproduction

Edit three values in `scripts/run_benchmark.sh`:

```bash
BASE_CONFIG="../configs/base/resnet18/mean_teacher_boundary_aware.yaml"
BENCH_CONFIG="../configs/bench/ludb/1over16.yaml"
MODE="single"  # single | distributed
```

Then run:

```bash
bash scripts/run_benchmark.sh
```

The base config selects boundary-aware Mean Teacher with ResNet-18 and a U-Net
decoder. The benchmark YAML overrides the base YAML. For cross-domain training,
set `BENCH_CONFIG="../configs/bench/cross_domain/merged.yaml"`.

Keep all other settings in YAML: data paths, seed, epochs, batch size, learning
rate, output directory, experiment name, and resume checkpoint. For distributed
training on one computer, set `ddp.world_size` to the number of GPUs to use.
All relative paths resolve from `semi-seg-ecg/src`.

The launcher preserves `CUDA_VISIBLE_DEVICES`. It selects which GPUs are visible;
`ddp.world_size` controls how many training processes the distributed launcher starts.
Visible GPUs are renumbered starting at zero, so keep `device: cuda` in both modes.
See [NVIDIA's device-selection documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-visible-devices).

Single-GPU example: add these settings to the benchmark YAML, keeping its existing
dataset block. The base config supplies all remaining settings.

```yaml
exp_name: ludb/1over16_single
device: cuda
dataloader:
  batch_size: 16
ddp:
  world_size: 1
```

Use these values in `scripts/run_benchmark.sh`:

```bash
BASE_CONFIG="../configs/base/resnet18/mean_teacher_boundary_aware.yaml"
BENCH_CONFIG="../configs/bench/ludb/1over16.yaml"
MODE="single"
```

Run on GPU 2 from the repository root:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_benchmark.sh
```

Two-GPU example: use these benchmark settings instead, keeping the same dataset block:

```yaml
exp_name: ludb/1over16_distributed
device: cuda
dataloader:
  batch_size: 8
ddp:
  world_size: 2
```

Keep the same config paths and set `MODE="distributed"` in the script. Run:

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_benchmark.sh
```

Batch size is per GPU. These examples keep 16 labeled and 16 unlabeled samples per
training step across GPUs, with the base config's `accum_iter: 1`. Match
`ddp.world_size` to the visible GPU count when using all selected GPUs. Training
initializes `ddp.rank`, `ddp.gpu`, and `ddp.distributed`; leave their base values alone.

Training runs testing once afterward using `test.target_metric` to select the
saved student checkpoint. Set `test: false` to skip testing. To test an existing
checkpoint separately, set `test.model_path` in YAML and run:

```bash
cd semi-seg-ecg/src
python test.py \
  -f ../configs/base/resnet18/mean_teacher_boundary_aware.yaml \
  -o ../configs/bench/ludb/1over16.yaml
```

## Citations
If this work has helped you please cite the following:

```
@misc{scharpf2026decoderdesignmattersecg,
      title={Decoder Design Matters for ECG Delineation}, 
      author={Joseph Scharpf and William Han and Chaojing Duan and Michael A. Rosenberg and Emerson Liu and Ding Zhao},
      year={2026},
      eprint={2609.16489},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.16489}, 
}

@inproceedings{10.1145/3746252.3760790,
  author = {Park, Minje and Lim, Jeonghwa and Yu, Taehyung and Joo, Sunghoon},
  title = {SemiSegECG: A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation},
  year = {2025},
  isbn = {9798400720406},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746252.3760790},
  doi = {10.1145/3746252.3760790},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages = {5099–5104},
  numpages = {6},
  location = {Seoul, Republic of Korea},
  series = {CIKM '25}
}
```

## License

MIT, except all third-party models and datasets used in the repository. Please refer to the third-party model and dataset's corresponding licenses.
Specifically, SemiSegECG abides by the Apache License Version 2.0.
