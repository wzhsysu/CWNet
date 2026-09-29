# CWNet

Official implementation of the paper “Revisiting Multi-Illuminant White Balance: A Simpler and Better Baseline”, ACMMM2026.

**Note:** The synthesis method is based on a prior work [link](https://github.com/timothybrooks/unprocessing).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU. Set `device: 'cpu'` in `configs/default.yaml` to run on CPU.

## Dataset

- **LSMI** dataset — third-party, by Kim et al. (ICCV 2021). Not part of this project.
- Download and terms of use: [LSMI dataset repository](https://github.com/DY112/LSMI-dataset)

## Checkpoint

<!-- [Pretrained weights](https://drive.google.com/file/d/1z6GmSZyGdknXjb8XjnjAAjNvX9VjzJg2/view?usp=sharing) -->

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

<!-- ## Citation

```bibtex
@inproceedings{cwnet2026,
  title     = {Revisiting Multi-Illuminant White Balance: A Simpler and Better Baseline},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (ACMMM)},
  year      = {2026}
}
``` -->
