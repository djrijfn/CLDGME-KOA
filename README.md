# CLDGME-KOA

Official PyTorch implementation of **CLDGME-KOA (Cumulative Link and
Direction-Guided Microstructure Enhancement Network)** for five-grade knee
osteoarthritis classification from radiographs. The model predicts
Kellgren-Lawrence (KL) grades 0--4 and corresponds to the accompanying
manuscript.

> This repository contains source code and experiment configurations. It does
> **not** redistribute OAI images, clinical records, trained checkpoints, or
> manuscript results.

## Contents

- [Method overview](#method-overview)
- [Dataset information](#dataset-information)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs and evaluation](#outputs-and-evaluation)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)
- [Contributing and support](#contributing-and-support)

## Method overview

CLDGME-KOA combines four components:

1. A ConvNeXt-Tiny image backbone extracts global visual features.
2. The Direction-Guided Microstructure Enhancement (DGME) module processes
   intermediate spatial features. It contains an Orthogonal Context Anchoring
   Branch (OCAB) and a Heterogeneous Receptive Field Aggregation Branch
   (HRFAB).
3. A prompt-semantic (PS) branch aligns image features with text prototypes
   constructed from six clinical descriptions for each KL grade.
4. An Ordered Cumulative Link Regression (OCLR) head models the ordinal
   relationship between KL grades. Horizontal-flip consistency (HFC) is used
   during training, and optional dual-view averaging is used during evaluation.

For the main configuration, the total training objective combines the prompt
classification loss, ordinal loss, and consistency loss:

$$
\mathcal{L} = \lambda_{\mathrm{PS}}\mathcal{L}_{\mathrm{PS}}
+ \lambda_{\mathrm{OCLR}}\mathcal{L}_{\mathrm{OCLR}}
+ \lambda_{\mathrm{HFC}}\mathcal{L}_{\mathrm{HFC}}.
$$

The default weights in `configs/cldgme_paper_main.yaml` are
$\lambda_{\mathrm{PS}}=0.5$, $\lambda_{\mathrm{OCLR}}=1.0$, and
$\lambda_{\mathrm{HFC}}=0.73$.

## Dataset information

### Source and access

The experiments use knee radiographs derived from the Osteoarthritis
Initiative (OAI). Access to the original data must be requested through the
[official OAI portal](https://nda.nih.gov/oai/). Users are responsible for
complying with the OAI data-use agreement and any institutional requirements.
The OAI data are not covered by this repository's software license.

### Input expected by this repository

This release does not include a raw-OAI preprocessing script. Before training,
prepare the radiographs as 224 x 224 RGB images and assign each image one KL
grade from 0 to 4. Before resampling, the split used in the accompanying study
contains 5,778 training images, 826 validation images, and 1,656 test images.

### Training-set resampling

Only the training partition is resampled. During each training epoch, a
class-aware weighted sampler draws 8,273 samples with replacement from the
5,778 original training images. KL1 and KL3 use a sampling factor of 2, while
KL4 uses a factor of 5; KL0 and KL2 use a factor of 1.

| KL grade | Original training images | Target samples per epoch | Sampling factor | Target increase |
|---|---:|---:|---:|---:|
| KL0 | 2,286 | 2,286 | x1 | 0 |
| KL1 | 1,046 | 2,092 | x2 | 1,046 |
| KL2 | 1,516 | 1,516 | x1 | 0 |
| KL3 | 757 | 1,514 | x2 | 757 |
| KL4 | 173 | 865 | x5 | 692 |
| **Total** | **5,778** | **8,273** | -- | **2,495** |

The values in the third column are the target/expected class composition
implied by the sampling factors. Because `WeightedRandomSampler` samples with
replacement, the realized class count in an individual epoch can differ
slightly from these values. The total number of draws remains 8,273. The
validation and test partitions are not resampled.

The released main configuration performs this step dynamically with
`use_weighted_sampler: true`, `sampler_class_factors: [1, 2, 1, 2, 5]`, and
`sampler_num_samples: 8273`. Keep only the 5,778 original training images in
the `train` directory; do not materialize additional copies there. Restricting
resampling to the training loader prevents repeated observations from entering
model selection or final evaluation.

Store the images in class-specific folders:

```text
data/oai-xray/
|-- train/
|   |-- 0/
|   |-- 1/
|   |-- 2/
|   |-- 3/
|   `-- 4/
|-- val/
|   |-- 0/
|   |-- 1/
|   |-- 2/
|   |-- 3/
|   `-- 4/
`-- test/
    |-- 0/
    |-- 1/
    |-- 2/
    |-- 3/
    `-- 4/
```

The loader scans class folders recursively. Supported formats are PNG, JPEG,
BMP, and TIFF. In addition to numeric class names, the loader recognizes
`normal`, `doubtful`, `minimal`, `mild`, `moderate`, and `severe`; numeric
folders are recommended because they map unambiguously to KL grades 0--4.
The validation folder may alternatively be named `valid` or `validation`.

To use a different location, change `image_root` in
`configs/cldgme_paper_main.yaml`. Keep the original patient-level split when
reconstructing the study dataset; regenerating a random image-level split can
cause data leakage and will not reproduce the reported experimental setting.

## Repository structure

```text
.
|-- configs/
|   |-- cldgme_paper_main.yaml   # main experiment configuration
|   `-- kl_prompts_paper.yaml    # six text prompts for each KL grade
|-- src/
|   |-- baselines.py             # comparison-model implementations
|   |-- dataset.py               # folder-based image dataset
|   |-- losses.py                # multi-branch and ordinal losses
|   |-- metrics.py               # classification and ordinal metrics
|   |-- model.py                 # CLDGME-KOA architecture
|   |-- prompts.py               # fallback prompt definitions
|   |-- train.py                 # main training/evaluation entry point
|   `-- train_baseline.py        # baseline training/evaluation entry point
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Requirements

The reference environment used during development was:

- Python 3.11.4
- PyTorch 2.6.0 with CUDA 11.8
- NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB memory

The code also selects CPU automatically when CUDA is unavailable, but training
on CPU is expected to be substantially slower. Package requirements are listed
in `requirements.txt`. Internet access is needed on the first run when
`pretrained_backbone: true` or when the Hugging Face text encoder is not
already cached.

The runtime dependencies are PyTorch, torchvision, Transformers,
Sentence-Transformers, NumPy, PyYAML, scikit-learn, SciPy, and tqdm. Pillow is
installed through torchvision and is used to load the radiographs.

## Installation

Run all commands from the repository root.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the generic PyTorch package does not match the local CUDA installation,
install the appropriate PyTorch build first and then install the remaining
requirements. Confirm the environment with:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

## Configuration

The main experiment is defined by `configs/cldgme_paper_main.yaml`. At minimum,
review the following fields before running it:

| Key | Purpose | Main setting |
|---|---|---|
| `image_root` | Root containing `train`, `val`, and `test` | `./data/oai-xray` |
| `output_dir` | Checkpoints and metrics directory | `runs/cldgme_paper_main` |
| `batch_size` | Images per optimization step | `6` |
| `num_workers` | Data-loading worker processes | `4` |
| `epochs` | Number of training epochs | `20` |
| `pretrained_backbone` | Load ImageNet ConvNeXt-Tiny weights | `true` |
| `use_weighted_sampler` | Enable dynamic training-set resampling | `true` |
| `sampler_class_factors` | KL0--KL4 per-sample weighting factors | `[1, 2, 1, 2, 5]` |
| `sampler_num_samples` | Number of training draws per epoch | `8273` |
| `hf_text_model_name` | Hugging Face text encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| `selection_metric` | Metric used to select `best.pt` | `accuracy` |
| `prediction_source` | Branch used for final predictions | `ordinal` |

Paths are interpreted relative to the directory from which the command is
executed. The prompt path can additionally be resolved relative to the config
file. For a GPU with less memory, reduce `batch_size`. On Windows, set
`num_workers: 0` if multiprocessing data loading causes worker-start errors.

## Usage

### 1. Check the data layout

The following command should print nonzero image counts for all three splits:

```bash
python -c "from src.dataset import KoaFolderDataset; root='./data/oai-xray'; print({s: len(KoaFolderDataset(root, s)) for s in ('train','val','test')})"
```

The dataset counts should be `train: 5778`, `val: 826`, and `test: 1656` because
the training directory contains only the original images. The dynamic sampler
then produces 8,273 draws per training epoch. Verify both quantities with:

```bash
python -c "import yaml; from src.train import build_loaders; cfg=yaml.safe_load(open('configs/cldgme_paper_main.yaml', encoding='utf-8')); train,_,_,_,counts=build_loaders(cfg); print('original class counts:', counts); print('dataset images:', len(train.dataset)); print('samples per epoch:', len(train.sampler))"
```

The expected original class counts are `[2286, 1046, 1516, 757, 173]`.
`class_counts` in `metrics.json` records these original counts rather than the
random class counts realized by the sampler in each epoch.

### 2. Train and evaluate CLDGME-KOA

```bash
python -m src.train --config configs/cldgme_paper_main.yaml
```

The script trains on `train`, selects the best checkpoint using the validation
metric specified by `selection_metric`, reloads that checkpoint, and evaluates
it once on `test`. There is currently no separate inference-only command in
this release.

### 3. Run a baseline

Baseline implementations are provided in `src/baselines.py` and use the same
folder-based data loader and evaluation metrics. Create a separate YAML config
for each comparison so that its method-specific hyperparameters and output
directory are recorded. A minimal example can be derived from the main config:

```yaml
# configs/example_baseline.yaml
seed: 42
image_root: ./data/oai-xray
output_dir: runs/example_baseline
image_size: 224
batch_size: 6
num_workers: 4
epochs: 20
lr: 1.0e-4
weight_decay: 1.0e-4
num_classes: 5
method: deep_siamese_cnn
backbone_type: resnet18
pretrained_backbone: true
use_weighted_sampler: false
use_weighted_loss: true
label_smoothing: 0.05
selection_metric: accuracy
use_flip_ensemble_eval: true
save_confusion_matrix: true
save_eval_predictions: true
```

Run it with:

```bash
python -m src.train_baseline --config configs/example_baseline.yaml
```

Do not treat the illustrative baseline settings above as manuscript
hyperparameters. Exact comparison settings must be taken from the accompanying
manuscript or the experiment record used for the reported table.

## Outputs and evaluation

The main configuration writes the following files under
`runs/cldgme_paper_main/`:

- `best.pt`: checkpoint with the best validation value of `selection_metric`;
- `last.pt`: final saved model state after reloading the best checkpoint;
- `metrics.json`: class counts, per-epoch validation history, best-validation
  metrics, and final test metrics.

Reported metrics are accuracy, macro precision, macro recall, macro F1,
one-vs-rest macro AUC (when all required classes are present), quadratic
weighted kappa (QWK), and mean absolute error (MAE). When enabled in the YAML
file, `metrics.json` also includes the test confusion matrix and the true and
predicted test labels.

Example inspection command:

```bash
python -c "import json; result=json.load(open('runs/cldgme_paper_main/metrics.json', encoding='utf-8')); print(json.dumps(result['test'], indent=2))"
```

## Reproducibility notes

- The supplied main configuration uses seed 42 and deterministic dataset
  splits, but exact GPU results can still vary across PyTorch, CUDA, cuDNN, and
  hardware versions.
- The main configuration performs class-aware sampling dynamically. Although
  the random seed is fixed, each epoch draws with replacement; the effective
  epoch length is 8,273 while the on-disk training set remains 5,778 images.
- Input images are resized to 224 x 224 and normalized with ImageNet mean and
  standard deviation. Training applies horizontal flipping and small random
  rotations; evaluation uses deterministic resizing and normalization.
- The initial run may download ImageNet backbone weights and
  `sentence-transformers/all-MiniLM-L6-v2`. To run from an existing local cache
  without network access, set `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` before launching training.
- `pretrained_backbone: false` disables the ImageNet weight download, but this
  changes the experimental setting.
- Checkpoints can be large and are excluded by `.gitignore`. Preserve the YAML
  config, environment versions, split definition, and `metrics.json` together
  when archiving a run.

## Citation

If this code contributes to a publication, cite the accompanying manuscript:

> *Cumulative Link and Direction-Guided Microstructure Enhancement Network for
> Knee Osteoarthritis Grading*.

Journal, year, DOI, and final author metadata are not included here because
final bibliographic information was not available when this code package was
prepared. This section should be updated with the publisher-formatted citation
after publication. Data users should also follow the acknowledgement and
citation requirements specified by the OAI data-use agreement.

## License

The source code is released under the [MIT License](LICENSE). This license
applies only to the software in this repository. OAI images and metadata remain
subject to the OAI terms and are not redistributed here.

## Contributing and support

Bug reports and focused pull requests are welcome. When reporting a problem,
include the operating system, Python/PyTorch/CUDA versions, configuration file,
full command, and traceback. Contributions should preserve the existing module
interfaces and include a minimal command or test that verifies the change.
