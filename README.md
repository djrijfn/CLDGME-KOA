# CLDGME-KOA

Cumulative Link and Direction-Guided Microstructure Enhancement Network for
Knee Osteoarthritis Grading.

This repository contains the official implementation of CLDGME-KOA for
Kellgren-Lawrence (KL) grade classification of knee radiographs (KL0-KL4),
corresponding to the associated manuscript.

## Method Overview

This repository implements a deep learning framework for knee osteoarthritis
grading.

The framework consists of:

- a convolutional backbone for feature extraction;
- feature enhancement modules for improving representation learning;
- prompt-based semantic guidance;
- an ordinal learning strategy for severity prediction.

The complete architectural details and experimental analysis are described in
the associated manuscript.

## Experimental Results

The repository provides the implementation and evaluation pipeline of the
proposed framework. Detailed quantitative comparisons and experimental
analyses will be reported in the corresponding publication.

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

Main experiment:

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
in `src/baselines.py` and trained through `src/train_baseline.py`.

## Repository layout

```
src/            training, model, losses, metrics, data loading
configs/        paper main config and prompt table
```
