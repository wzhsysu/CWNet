# CWNet

Official implementation of the paper “Revisiting Multi-Illuminant White Balance: A Simpler and Better Baseline”, ACMMM2026.

> **Scope of this release.** This repository contains the **evaluation pipeline only**.
> The training pipeline is not included — `main.py` runs in `mode=test` and nothing else.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU. Set `device: 'cpu'` in `configs/default.yaml` to run on CPU.

## Dataset

Evaluation uses the **LSMI** (Large-Scale Multi-Illuminant) dataset.

<!-- TODO: add dataset download link -->
Dataset: **`<TODO: dataset link>`**

## Checkpoint

<!-- TODO: add pretrained checkpoint download link -->
Pretrained weights: **`<TODO: checkpoint link>`**

## Evaluation

1. Edit `scripts/test.sh` and set the dataset root and the checkpoint path:

   ```bash
   data.root=/PATH/TO/LSMI/galaxy_512/
   load.ckpt_path=/PATH/TO/ckpt.pt
   ```

   The dataset directory name must contain the camera keyword (`galaxy`, `sony`,
   `nikon`, or `gsn` for the merged Galaxy+Sony+Nikon set) — the loader infers the
   RAW bit depth from it.

2. Run:

   ```bash
   cd scripts
   bash test.sh
   ```

The script evaluates the test split and reports MAE.

## Notes on visualization

`test.visualize_result` is **off by default**. Turning it on re-renders predictions
through a camera RAW template (`datasets/<camera>.dng`), and **those templates are not
distributed with this release**. If you need the visualizations, place your own
`<camera>.dng` under `datasets/` and set `test.visualize_result=true`.

<!-- ## Citation

```bibtex
@inproceedings{cwnet2026,
  title     = {Revisiting Multi-Illuminant White Balance: A Simpler and Better Baseline},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (ACMMM)},
  year      = {2026}
}
``` -->
