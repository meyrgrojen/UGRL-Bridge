# UGRL-Bridge MRI Reconstruction

Code for retrospective complex-valued single-image MRI reconstruction. The repository includes preprocessing, fixed Cartesian mask generation, staged training, evaluation, and inference.

## Installation

Use Python 3.10 or later. Install a PyTorch build compatible with your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The model, sampling, loss, and optimization settings are in `configs/ugrl_mri_singlecoil.yaml`.

## Data

Input samples are curated two-dimensional complex-valued single images in `.mat`, `.h5`, `.npy`, or `.npz` format. The default image key is `GT`. Obtain the datasets through their official access pages: [fastMRI](https://fastmri.med.nyu.edu/), [M4Raw](https://zenodo.org/records/7998442), and [CMRxRecon2023](https://cmrxrecon.github.io/Challenge.html). This code does not perform multi-coil combination.

The input manifest follows `splits/manifest_schema.csv`. Use `fastmri_knee`, `fastmri_brain_axflair`, `m4raw_t2`, or `cmrxrecon2023_cinesax` as the dataset name and `train`, `val`, or `test` as the split. Paths are relative to the input data root. Each sample ID must be unique within its dataset, and all samples from a subject must remain in one split.

## Preprocessing and fixed masks

Prepare a separate directory and manifest for each acceleration factor. For example, for fastMRI Brain at `R=8`:

```bash
python tools/prepare_data.py --input-root /path/to/curated --input-manifest /path/to/source_manifest.csv --output-root data/processed/R8 --output-manifest splits/brain_r8.csv --dataset fastmri_brain_axflair --image-size 320 --acceleration 8
python tools/generate_mask_bank.py --manifest splits/brain_r8.csv --dataset fastmri_brain_axflair --acceleration 8 --matrix-size 320 --out splits/brain_r8_masks.csv
```

## Training, evaluation, and inference

```bash
python train.py --dataset fastmri_brain_axflair --acceleration 8 --manifest splits/brain_r8.csv --data-root data/processed/R8
python evaluate.py --ckpt runs/ugrl_bridge/fastmri_brain_axflair/R8/best.pt --manifest splits/brain_r8.csv --data-root data/processed/R8 --split test --out runs/eval_brain_r8
python infer.py --ckpt runs/ugrl_bridge/fastmri_brain_axflair/R8/best.pt --input /path/to/measured_kspace.npz --out runs/single_reconstruction.npz
```
