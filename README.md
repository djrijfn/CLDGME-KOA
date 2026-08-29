# CLDGME-KOA

Cumulative Link and Direction-Guided Microstructure Enhancement Network for
Knee Osteoarthritis Grading.

This repository contains the official implementation of CLDGME-KOA for
Kellgren-Lawrence (KL) grade classification of knee radiographs (KL0-KL4),
corresponding to the manuscript *A Computer Vision Framework for Ordinal
Knee Osteoarthritis Grading with Cumulative Link and Direction-Guided
Microstructure Enhancement* (en826.tex).

## Method

All names below follow the manuscript.

- **Visual backbone**: ConvNeXt-Tiny initialized with ImageNet-1K weights,
  input `224x224` RGB with ImageNet normalization.
- **DGME** (Direction-Guided Microstructure Enhancement Module) is attached
  to the intermediate Stage2 feature map (`B x 192 x 28 x 28`) as a side
  branch. It consists of:
  - **OCAB** (Orthogonal Context Anchoring Branch): strip pooling along
    height/width, `7x1` / `1x7` depthwise convolutions, spatial anchoring.
  - **HRFAB** (Heterogeneous Receptive Field Aggregation Branch): parallel
    `3x3` / `5x5` / `7x7` depthwise convolutions fused by `1x1` + BN + GELU.
  - Outputs are fused with the backbone global feature (`768 + 768 -> 768 ->
    256`, ReLU MLP) to form the visual embedding `z` in R^256.
- **PS** (Semantic Prototype Regularization): a frozen
  `sentence-transformers/all-MiniLM-L6-v2` encoder maps 6 KL-grade prompts
  per grade (3 local + 3 global, manuscript Table 1, verbatim in
  `configs/kl_prompts_paper.yaml`) into 256-d prototypes via Prompt Ensemble
  (PE). The visual embedding is aligned with the grade prototype by
  class-weighted cross-entropy during training.
- **OCLR** (Ordinal Cumulative Link Regression) head: shared latent severity
  score `eta = w^T z + b`, strictly ordered thresholds parameterized by an
  anchor plus Softplus-transformed positive increments, cumulative
  probabilities `q_j = sigmoid(eta - theta_j)`, and decoding by counting
  `q_j > 0.5`. No decoding threshold is optimized on the validation set.
- **HFC** (Horizontal Flip Consistency): Jensen-Shannon divergence between
  the semantic distributions of the original and horizontally flipped views.
  Dual-view inference averages the cumulative probabilities of the two views
  before ordinal decoding.

Training objective:
`L = lambda_PS * L_PS + lambda_OCLR * L_OCLR + lambda_HFC * L_HFC`
with `lambda_PS = 0.5`, `lambda_OCLR = 1.0`, `lambda_HFC = 0.73`.

## Main results (paper protocol)

Model checkpoints are selected by validation Accuracy; the test set is used
only for final reporting. Decoding uses the fixed threshold 0.5.

| Split | Accuracy | Macro-AUC | Macro-F1 | QWK | MAE |
|---|---:|---:|---:|---:|---:|
| Validation (best) | 0.6901 | 0.8678 | 0.6844 | 0.8280 | - |
| Test (dual-view) | 0.7150 | 0.8921 | 0.7082 | 0.8596 | 0.3255 |
| Test (single-view) | 0.7095 | 0.8905 | 0.7044 | 0.8562 | 0.3315 |

## Data

- Public knee radiographs derived from the Osteoarthritis Initiative (OAI).
  Each image is assigned a KL grade 0-4.
- Predefined split used in this study: 5,778 train / 826 validation /
  1,656 test images, organized as `image_root/{train,val,test}/{0,1,2,3,4}/`.
- Images are 224x224 RGB PNG with ImageNet mean/std normalization.
- The dataset itself is not redistributed. Obtain it from the OAI official
  channels and arrange the folders as above. Then set `image_root` in
  `configs/cldgme_paper_main.yaml` to the local data directory
  (default `./data/oai-xray`). The train/val/test file lists used in this
  study can be reconstructed by listing each class folder.

## Environment

The implementation was developed with Python 3.11.4, PyTorch 2.6.0+cu118,
CUDA 11.8 on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).

```
pip install torch torchvision transformers sentence-transformers pyyaml numpy scikit-learn tqdm
```

If the HuggingFace Hub is unreachable, the text encoder loads from cache:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Reproduce

Main experiment (paper protocol):

```bash
python -m src.train --config configs/cldgme_paper_main.yaml
```

## Naming conventions

Config keys and code identifiers follow the manuscript:

| Manuscript | Code |
|---|---|
| CLDGME-KOA | `CLDGMEKOA` (`src/model.py`) |
| DGME | `DGME` module, `use_dgme`, `dgme_stage` |
| OCAB | `OrthogonalContextAnchoringBranch`, `dgme_use_ocab` |
| HRFAB | `HeterogeneousReceptiveFieldAggregationBranch`, `dgme_use_hrfab` |
| OCLR | `OCLRHead`, `ordinal_head_type: oclr` (or `cumulative_link`) |
| PS | text branch: `use_text_branch`, `loss_alpha_text` |
| HFC | `use_consistency_loss`, `loss_gamma_consistency` |
| PE | enabled by `configs/kl_prompts_paper.yaml` (6 prompts per grade) |

Backward-compatible aliases (`ClipKoaMinimal`, `CumulativeLinkOrdinalHead`,
legacy `sleb_*` config keys) are kept so pre-rename checkpoints and configs
remain loadable.

## Baselines

Seven literature baselines used in the manuscript comparison are implemented
in `src/baselines.py` and trained through `src/train_baseline.py`. The
corresponding `baseline_*.yaml` configs are not part of this minimal release.

## Implementation notes

- Class weights are applied to the semantic prototype loss only; the OCLR
  loss is unweighted binary cross-entropy over the K-1 cumulative tasks.
- Prompt prototypes are projected, L2-normalized per prompt, and then
  averaged (project -> normalize -> mean).
- The text loss averages the cross-entropy of the original and flipped
  views; label smoothing is 0.05.
- The DGME side branch uses BatchNorm2d, bias-free depthwise convolutions,
  and padding matched to the kernel size.

## Repository layout

```
src/            training, model, losses, metrics, data loading
configs/        paper main config and prompt table
```
