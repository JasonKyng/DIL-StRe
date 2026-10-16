# Plug-and-Play Dynamic In-context Learning with Stochastic Regularization for Screen Content Image Super-Resolution

**Yuexin Wang, Xiaolei Wang, Guangliang Cheng, Huihui Bai, Tammam Tillo, and Jimin Xiao**  
*CVPR 2026 Findings*, pp. 8836-8845.

[Paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_Plug-and-Play_Dynamic_In-context_Learning_with_Stochastic_Regularization_for_Screen_Content_CVPRF_2026_paper.pdf) | [CVF publication page](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_Plug-and-Play_Dynamic_In-context_Learning_with_Stochastic_Regularization_for_Screen_Content_CVPRF_2026_paper.html) | [Project repository](https://github.com/JasonKyng/DIL-StRe)

This repository contains the **BTC-based implementation of Dynamic In-context Learning (DIL) and Stochastic Regularization (StRe)** for screen content image super-resolution. It extends [B-spline Texture Coefficients Estimator (BTC)](https://github.com/ByeongHyunPak/btc), rather than providing every backbone and experiment evaluated in the paper.

## Overview

Screen content images contain recurring structures such as characters and icons. DIL exploits these contextual relationships to improve reconstruction, while StRe addresses the frequency imbalance that can bias optimization toward less informative regions.

- **DIL:** models global relationships between pixel features using an efficient attention formulation. In the encoder-side implementation, it operates on RDN features before BTC estimates B-spline coefficients and knots.
- **StRe:** introduces a stochastic perturbation during training. In this implementation, Gaussian noise is added to the normalized LR image in the bilinear residual branch; the encoder still receives the original normalized input. Noise is disabled during evaluation.
- **BTC integration:** retains the coordinate-based B-spline reconstruction and local ensemble, supporting queries at the desired output coordinates.

<img width="1555" height="807" alt="image" src="https://github.com/user-attachments/assets/2c0c0f4a-1604-4b2d-9b98-9602c8188438" />

The attention implementation in `models/modules/galerkin.py` computes `Q(K^T V) / N` after normalizing keys and values. This avoids explicitly constructing an `N x N` pixel attention matrix. The main configuration uses 64-channel encoder features and eight attention heads.

## Repository structure

```text
btc+ours/
├── configs/
│   ├── train/                     # Three RDN-BTC training configurations
│   └── test/                      # SCI1K, SCID, SIQAD; integer scales x2-x10
├── datasets/
│   ├── image_folder.py            # HR-only and paired LR/HR loading
│   └── wrappers.py                # Downsampling, augmentation, coordinate sampling
├── models/
│   ├── methods/BTC.py             # BTC, encoder-side DIL, and StRe
│   └── modules/
│       ├── RDN.py                 # Feature encoder used by the supplied configs
│       ├── MLP.py                 # Standard and attention-equipped RGB decoders
│       └── galerkin.py            # DIL attention module
├── scripts/
│   ├── test.sh                    # Evaluate all three datasets
│   ├── test_SCI1K.sh
│   ├── test_SCID.sh
│   └── test_SIQAD.sh
├── train.py
├── test.py                        # PSNR/SSIM evaluation and image export
├── train.sh
├── test.sh                        # Local experiment examples with hard-coded paths
├── utils.py
└── LICENSE
```

## Installation

The paper reports experiments on a single NVIDIA RTX 3090 with 24 GB of memory. Use a CUDA-capable NVIDIA GPU. The current implementation calls `.cuda()` directly and does not provide a CPU/MPS execution option.

The upstream BTC README specifies PyTorch 1.13.0. The following is a suggested environment based on that version and the imports in this repository, not a validated environment lock for this snapshot:

```bash
conda create -n btc-dil python=3.10 -y
conda activate btc-dil

# Example for a CUDA 11.7-compatible NVIDIA driver.
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install "numpy<2" tensorboardX pyyaml tqdm imageio pillow einops
python -m pip check
```

See the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v1130) for other CUDA builds. Run all commands below from this repository's root directory. No `requirements.txt` is supplied in this snapshot.

## Datasets

The supplied configurations use **SCI1K** for training and **SCI1K-Test, SCID, and SIQAD** for evaluation. Download instructions are available in the [ITSRN data repository](https://github.com/codyshen0000/ITSRN/tree/main/Data), also referenced by upstream BTC.

Arrange the data as follows, or update the YAML paths to match your existing layout:

```text
/path/to/SCIData/
├── SCI1K-Train/
│   └── HR/
├── SCI1K-Test/
│   ├── X2/
│   │   ├── LR/
│   │   └── HR/
│   ├── X3/
│   └── ...                        # X4 through X10, each with LR/ and HR/
├── SCID/
│   ├── X2/
│   │   ├── LR/
│   │   └── HR/
│   └── ...                        # X3 through X10
└── SIQAD/
    ├── X2/
    │   ├── LR/
    │   └── HR/
    └── ...                        # X3 through X10
```

Replace the machine-specific `/gpfs/home/int/xxx/...` paths in the YAML files:

| Configuration field | Required location |
| --- | --- |
| `train_dataset.dataset.args.root_path` | `SCI1K-Train/HR` |
| `val_dataset.dataset.args.root_path` | `SCI1K-Test/X2/HR` in the supplied training configs |
| `test_dataset.dataset.args.root_path_1` | LR images for the selected dataset and scale |
| `test_dataset.dataset.args.root_path_2` | Matching HR images |
| `pre_train` / `resume` | An existing checkpoint, or `null` when unused |

Training generates LR patches from HR images using bicubic downsampling and samples scales uniformly from 1 to 4. Evaluation reads precomputed LR/HR pairs. The paired loader sorts each directory independently and matches images **by index**, so the two directories must have matching image counts and corresponding sorted filenames. Image directories should contain only image files; the loader does not filter unrelated files.

Training and validation default to `cache: in_memory`; change this to `none` if host RAM is limited.

## Configurations

| Training configuration | Model / decoder | Behavior in this snapshot |
| --- | --- | --- |
| `train-rdn+btc-attn-e.yaml` | `btc-attn` / `mlp` | **Main DIL + StRe configuration:** DIL after the RDN encoder, before coefficient/knot prediction |
| `train-rdn+btc-attn.yaml` | `btc` / `simple-att-mlp` | Alternative placement: attention inside the RGB decoder, with StRe |
| `train-rdn+btc-3rd.yaml` | `btc` / `mlp` | BTC with StRe and no DIL; this is not the unchanged upstream baseline |

Both `BTC` and `BTCAttn` currently inject `torch.randn_like(self.inp) * 0.1` into the residual branch during training. The standard deviation **0.1** is hard-coded in normalized input space; the YAML normalization maps `[0, 1]` to `[-1, 1]`. There is no command-line or YAML noise switch.

The decoder-side attention variant mixes the query points in each call. Its context therefore depends on the query batching used at evaluation. The encoder-side variant computes DIL over the LR feature map before query chunking.

## Training

### Main DIL + StRe configuration

First update the data and checkpoint paths in `configs/train/train-rdn+btc-attn-e.yaml`. After resolving the validation issue described below, launch:

```bash
python train.py \
  --config configs/train/train-rdn+btc-attn-e.yaml \
  --tag _dil_stre \
  --gpu 0
```

The resulting run directory is `save/rdn+btc-attn-e_dil_stre/`. The `--tag` string is appended literally; include a leading underscore if desired. Multiple training GPUs can be selected with `--gpu 0,1`, which enables `DataParallel`.

The root `train.sh` selects `train-rdn+btc-3rd.yaml`, so it runs the StRe-only configuration rather than the encoder-side DIL configuration above.

### Training settings

The main YAML contains:

| Setting | Value |
| --- | --- |
| Encoder | RDN, without its upsampling head |
| B-spline order | 3 |
| Decoder | Four hidden layers of width 256, ReLU, RGB output |
| LR patch size | 48 x 48 |
| Training scale range | 1-4, sampled continuously |
| Sampled HR query points | 2304 per patch |
| Batch size | 16 |
| Dataset repeat | 20 |
| Augmentation | Random horizontal/vertical flips and transpose |
| Loss | L1 reconstruction loss |
| Optimizer | Adam, initial learning rate `1e-4` |
| Schedule | 1000 epochs; halve LR at epochs 200, 400, 600, 800 |
| Validation / periodic save | Every 1 / 100 epochs |
| Random seed | 2021 in `train.py` |

The seed is fixed in code, but cuDNN benchmarking is enabled; exact deterministic reproduction is not guaranteed.

## Evaluation

Update the relevant test YAML paths and use a checkpoint trained with the matching model implementation.

Evaluate SCI1K at x2:

```bash
python test.py \
  --config configs/test/test-sci1k-02.yaml \
  --model save/rdn+btc-attn-e_dil_stre/epoch_best.pth \
  --gpu 0
```

Evaluate every supplied scale, x2 through x10, on one dataset:

```bash
bash scripts/test_SCI1K.sh save/rdn+btc-attn-e_dil_stre/epoch_best.pth 0
bash scripts/test_SCID.sh save/rdn+btc-attn-e_dil_stre/epoch_best.pth 0
bash scripts/test_SIQAD.sh save/rdn+btc-attn-e_dil_stre/epoch_best.pth 0
```

Or evaluate all three datasets:

```bash
bash scripts/test.sh save/rdn+btc-attn-e_dil_stre/epoch_best.pth 0
```

Use `scripts/test.sh` for this workflow; the root `test.sh` contains hard-coded paths to local experiments. Test configurations are named `test-{sci1k,scid,siqad}-{02,...,10}.yaml`. Evaluation uses batch size 1 and up to 50,000 query points per chunk (`eval_bsize`). Reduce the latter to lower query-decoding memory usage; this does not reduce the encoder's full-image feature memory.

The evaluator prints average **PSNR and SSIM**, and writes reconstructed images to:

```text
output/rdn+btc-attn-e_dil_stre/SCI1K/2x/test_001.png
```

The model directory name is taken from the checkpoint's parent directory. Preserve `btc` in that directory name: `test.py` uses it to enable BTC cell decoding, including the cell adjustment for scales above x4. Image filenames are sequential rather than copied from the input files.

For the three screen-content benchmarks, the current metric implementation crops a one-pixel border, computes PSNR from a luminance-weighted RGB difference, and computes SSIM on cropped RGB images using an 11 x 11 Gaussian window. These details describe this implementation and should be checked when comparing metrics from other repositories. The supplied configs cover integer scales x2-x10; this snapshot does not include a standalone arbitrary-scale `demo.py` or a scene-text-recognition evaluation pipeline.

## Citation

Please cite our paper and the BTC baseline when using this implementation:

```bibtex
@InProceedings{Wang_2026_CVPR,
  author    = {Wang, Yuexin and Wang, Xiaolei and Cheng, Guangliang and Bai, Huihui and Tillo, Tammam and Xiao, Jimin},
  title     = {Plug-and-Play Dynamic In-context Learning with Stochastic Regularization for Screen Content Image Super-Resolution},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026},
  pages     = {8836--8845}
}
```

## Acknowledgements and license

This implementation is based on [BTC](https://github.com/ByeongHyunPak/btc). We also acknowledge [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte), which are credited by the upstream project, and the dataset contributors.

The included [MIT license](LICENSE) retains the upstream BTC copyright notice. Please retain the applicable notices when redistributing the code; datasets and pretrained models are subject to their respective terms.
