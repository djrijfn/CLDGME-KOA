import argparse
import csv
import itertools
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

try:
    from .dataset import KoaFolderDataset
    from .losses import compute_multibranch_loss, decode_coral_logits
    from .metrics import compute_metrics
    from .model import CLDGMEKOA
    from .prompts import DEFAULT_PROMPTS
except ImportError:
    from dataset import KoaFolderDataset
    from losses import compute_multibranch_loss, decode_coral_logits
    from metrics import compute_metrics
    from model import CLDGMEKOA
    from prompts import DEFAULT_PROMPTS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_path(path, config_dir=None):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    if config_dir is not None:
        for base in (config_dir, os.path.dirname(config_dir)):
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return candidate
    return path


def _normalize_prompt_data(data, num_classes=5):
    if isinstance(data, dict) and 'prompts' in data:
        data = data['prompts']

    if isinstance(data, dict):
        out = []
        for grade in range(int(num_classes)):
            value = data.get(grade, data.get(str(grade), None))
            if value is None:
                raise ValueError(f'missing prompts for grade {grade}')
            out.append(value if isinstance(value, list) else [value])
        return out

    if isinstance(data, list):
        if len(data) == int(num_classes) and all(isinstance(x, (str, list)) for x in data):
            return [x if isinstance(x, list) else [x] for x in data]
        if all(isinstance(x, dict) for x in data):
            by_grade = {}
            for item in data:
                grade = int(item['grade'])
                value = item.get('prompts', item.get('prompt', item.get('description')))
                by_grade[grade] = value if isinstance(value, list) else [value]
            return [by_grade[i] for i in range(int(num_classes))]

    raise ValueError('prompt_file must be a grade->prompts mapping or a list with one entry per class')


def load_prompts(clinical_root, prompt_file=None, config_dir=None, num_classes=5):
    prompt_path = _resolve_path(prompt_file, config_dir=config_dir)
    if prompt_path is not None:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return _normalize_prompt_data(data, num_classes=num_classes)

    csv_path = os.path.join(clinical_root, 'grade_descriptions.csv')
    if not os.path.isfile(csv_path):
        return DEFAULT_PROMPTS
    out = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row['grade'])] = row['description'].strip()
    if len(out) != 5:
        return DEFAULT_PROMPTS
    return [out[i] for i in range(5)]


def is_better_metric(current_metrics, best_score, selection_metric):
    value = current_metrics.get(selection_metric)
    if value is None:
        raise KeyError(f'selection_metric={selection_metric} not found in validation metrics')
    value = float(value)
    if best_score is None:
        return True, value
    if str(selection_metric).lower() == 'mae':
        return value < best_score, value
    return value > best_score, value


def compute_class_stats(labels, num_classes):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(num_classes)).astype(np.float32)
    counts[counts == 0.0] = 1.0
    inv = 1.0 / counts
    class_weights = inv / inv.sum() * float(num_classes)
    sample_weights = [float(inv[int(y)]) for y in labels]
    return class_weights, sample_weights, counts.tolist()


class RandomGamma:
    def __init__(self, gamma_range=(0.8, 1.25), p=0.3):
        self.gamma_range = gamma_range
        self.p = float(p)

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        gamma = random.uniform(float(self.gamma_range[0]), float(self.gamma_range[1]))
        return transforms.functional.adjust_gamma(img, gamma=gamma)


class AddGaussianNoise:
    def __init__(self, std_range=(0.005, 0.02), p=0.2):
        self.std_range = std_range
        self.p = float(p)

    def __call__(self, tensor):
        if random.random() >= self.p:
            return tensor
        std = random.uniform(float(self.std_range[0]), float(self.std_range[1]))
        return tensor + torch.randn_like(tensor) * std


def build_train_transform(cfg, size):
    policy = str(cfg.get('augmentation_policy', 'default')).lower()
    if policy != 'radiograph_preserving':
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(8),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.RandomAutocontrast(p=0.2),
            RandomGamma(
                gamma_range=cfg.get('gamma_range', (0.8, 1.25)),
                p=cfg.get('gamma_p', 0.3),
            ),
            transforms.ToTensor(),
            AddGaussianNoise(
                std_range=cfg.get('gaussian_noise_std', (0.005, 0.02)),
                p=cfg.get('gaussian_noise_p', 0.2),
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def override_sample_weights(labels, default_sample_weights, cfg, num_classes):
    sampler_class_weights = cfg.get('sampler_class_weights', None)
    sampler_class_factors = cfg.get('sampler_class_factors', None)
    if sampler_class_weights is not None and sampler_class_factors is not None:
        raise ValueError('sampler_class_weights and sampler_class_factors cannot be set together')
    if sampler_class_factors is not None:
        if len(sampler_class_factors) != int(num_classes):
            raise ValueError(
                f'sampler_class_factors must have {int(num_classes)} values, got {len(sampler_class_factors)}'
            )
        factors = [float(v) for v in sampler_class_factors]
        return [factors[int(y)] for y in labels]
    if sampler_class_weights is None:
        return default_sample_weights
    if len(sampler_class_weights) != int(num_classes):
        raise ValueError(
            f'sampler_class_weights must have {int(num_classes)} values, got {len(sampler_class_weights)}'
        )
    weights = [float(v) for v in sampler_class_weights]
    return [weights[int(y)] for y in labels]


def compute_sampler_num_samples(labels, cfg, num_classes):
    explicit_num_samples = cfg.get('sampler_num_samples', None)
    if explicit_num_samples is not None:
        return int(explicit_num_samples)

    sampler_class_factors = cfg.get('sampler_class_factors', None)
    if sampler_class_factors is None:
        return len(labels)
    if len(sampler_class_factors) != int(num_classes):
        raise ValueError(
            f'sampler_class_factors must have {int(num_classes)} values, got {len(sampler_class_factors)}'
        )
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(num_classes)).astype(np.float32)
    factors = np.asarray([float(v) for v in sampler_class_factors], dtype=np.float32)
    return int(np.rint(counts * factors).sum())


def build_loaders(cfg):
    size = int(cfg['image_size'])
    num_classes = int(cfg.get('num_classes', 5))
    train_tf = build_train_transform(cfg, size)
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds_train = KoaFolderDataset(cfg['image_root'], 'train', transform=train_tf)
    ds_val = KoaFolderDataset(cfg['image_root'], 'val', transform=eval_tf)
    ds_test = KoaFolderDataset(cfg['image_root'], 'test', transform=eval_tf)
    class_weights_np, sample_weights, class_counts = compute_class_stats(ds_train.get_labels(), num_classes)
    sample_weights = override_sample_weights(ds_train.get_labels(), sample_weights, cfg, num_classes)
    sampler_num_samples = compute_sampler_num_samples(ds_train.get_labels(), cfg, num_classes)
    sampler = None
    if bool(cfg.get('use_weighted_sampler', True)):
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=sampler_num_samples,
            replacement=True,
        )
    bs = int(cfg['batch_size'])
    nw = int(cfg['num_workers'])
    train_loader = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32)
    return train_loader, val_loader, test_loader, class_weights, class_counts


def select_prediction(outputs, prediction_source='ordinal'):
    source = str(prediction_source).lower()
    ord_logits = outputs.get('ordinal_logits', None)
    text_logits = outputs.get('text_logits', None)
    if source == 'ordinal':
        if ord_logits is not None:
            pred, _ = decode_coral_logits(ord_logits)
            return pred
        if text_logits is not None:
            return torch.argmax(text_logits, dim=1)
    elif source == 'text':
        if text_logits is not None:
            return torch.argmax(text_logits, dim=1)
        if ord_logits is not None:
            pred, _ = decode_coral_logits(ord_logits)
            return pred
    else:
        raise ValueError(f'unsupported prediction_source: {prediction_source}')
    raise RuntimeError('no available prediction branch; enable text or ordinal branch')


def _ordinal_class_probs_from_logits(ord_logits, num_classes):
    num_classes = int(num_classes)
    p_gt = torch.sigmoid(ord_logits)
    probs = torch.zeros((ord_logits.shape[0], num_classes), dtype=ord_logits.dtype, device=ord_logits.device)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for c in range(1, num_classes - 1):
        probs[:, c] = p_gt[:, c - 1] - p_gt[:, c]
    probs[:, num_classes - 1] = p_gt[:, num_classes - 2]
    probs = torch.clamp(probs, min=0.0)
    denom = probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs / denom


def _corn_class_probs_from_logits(ord_logits, num_classes):
    cond = torch.sigmoid(ord_logits)
    greater = torch.cumprod(cond, dim=1)
    probs = torch.zeros((ord_logits.shape[0], num_classes), dtype=ord_logits.dtype, device=ord_logits.device)
    probs[:, 0] = 1.0 - greater[:, 0]
    for c in range(1, num_classes - 1):
        probs[:, c] = greater[:, c - 1] * (1.0 - cond[:, c])
    probs[:, num_classes - 1] = greater[:, num_classes - 2]
    probs = torch.clamp(probs, min=0.0)
    denom = probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs / denom


def _softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)


def _sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _ordinal_class_probs_from_logits_np(ord_logits, num_classes):
    p_gt = _sigmoid_np(ord_logits)
    probs = np.zeros((p_gt.shape[0], int(num_classes)), dtype=np.float64)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for c in range(1, int(num_classes) - 1):
        probs[:, c] = p_gt[:, c - 1] - p_gt[:, c]
    probs[:, int(num_classes) - 1] = p_gt[:, int(num_classes) - 2]
    probs = np.clip(probs, a_min=0.0, a_max=None)
    denom = np.clip(probs.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return probs / denom


def _corn_class_probs_from_logits_np(ord_logits, num_classes):
    cond = _sigmoid_np(ord_logits)
    greater = np.cumprod(cond, axis=1)
    probs = np.zeros((greater.shape[0], int(num_classes)), dtype=np.float64)
    probs[:, 0] = 1.0 - greater[:, 0]
    for c in range(1, num_classes - 1):
        probs[:, c] = greater[:, c - 1] * (1.0 - cond[:, c])
    probs[:, num_classes - 1] = greater[:, num_classes - 2]
    probs = np.clip(probs, a_min=0.0, a_max=None)
    denom = np.clip(probs.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return probs / denom


def _ordinal_class_probs_np(ord_logits, num_classes, ordinal_type='coral'):
    if str(ordinal_type).lower() == 'corn':
        return _corn_class_probs_from_logits_np(ord_logits, num_classes)
    return _ordinal_class_probs_from_logits_np(ord_logits, num_classes)


def _ordinal_gt_probs_np(ord_logits, ordinal_type='coral'):
    probs = _sigmoid_np(ord_logits)
    if str(ordinal_type).lower() == 'corn':
        return np.cumprod(probs, axis=1)
    return probs


def _ordinal_pred_with_thresholds(ord_logits, thresholds, ordinal_type='coral'):
    probs = _ordinal_gt_probs_np(ord_logits, ordinal_type=ordinal_type)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
    return (probs > thr).sum(axis=1).astype(np.int64)


def _accuracy_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return float(np.mean(y_true == y_pred))


def _search_threshold_grid(ord_logits, y_true, threshold_grid, ordinal_type='coral'):
    best_thresholds = None
    best_acc = -1.0
    best_pred = None
    for thresholds in itertools.product(threshold_grid, repeat=ord_logits.shape[1]):
        pred = _ordinal_pred_with_thresholds(ord_logits, thresholds, ordinal_type=ordinal_type)
        acc = _accuracy_score_np(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = list(map(float, thresholds))
            best_pred = pred
    return best_thresholds, best_pred, best_acc


def _build_fine_threshold_grid(coarse_thresholds, fine_step=0.025, fine_radius=0.05):
    fine_grid = []
    for value in coarse_thresholds:
        low = max(0.0, float(value) - float(fine_radius))
        high = min(1.0, float(value) + float(fine_radius))
        num_steps = int(round((high - low) / float(fine_step))) + 1
        values = [round(low + i * float(fine_step), 6) for i in range(num_steps)]
        values = [min(1.0, max(0.0, v)) for v in values]
        fine_grid.append(sorted(set(values + [float(value)])))
    return fine_grid


def _search_best_ordinal_thresholds_two_stage(
    ord_logits,
    y_true,
    coarse_candidates=None,
    fine_step=0.025,
    fine_radius=0.05,
    ordinal_type='coral',
):
    coarse_grid = coarse_candidates or [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    coarse_thresholds, coarse_pred, coarse_acc = _search_threshold_grid(
        ord_logits,
        y_true,
        coarse_grid,
        ordinal_type=ordinal_type,
    )

    fine_grid_per_threshold = _build_fine_threshold_grid(
        coarse_thresholds=coarse_thresholds,
        fine_step=fine_step,
        fine_radius=fine_radius,
    )
    best_thresholds = None
    best_acc = coarse_acc
    best_pred = coarse_pred
    for thresholds in itertools.product(*fine_grid_per_threshold):
        pred = _ordinal_pred_with_thresholds(ord_logits, thresholds, ordinal_type=ordinal_type)
        acc = _accuracy_score_np(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = list(map(float, thresholds))
            best_pred = pred

    if best_thresholds is None:
        best_thresholds = coarse_thresholds
    return best_thresholds, best_pred, best_acc


def _collect_eval_outputs(model, loader, device, num_classes=5, use_flip_ensemble=False):
    model.eval()
    y_true = []
    text_logits_all = []
    ordinal_logits_all = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images, return_dict=True)
        if use_flip_ensemble:
            outputs_h = model(torch.flip(images, dims=[3]), return_dict=True)
            if outputs.get('text_logits', None) is not None and outputs_h.get('text_logits', None) is not None:
                outputs['text_logits'] = 0.5 * (outputs['text_logits'] + outputs_h['text_logits'])
            if outputs.get('ordinal_logits', None) is not None and outputs_h.get('ordinal_logits', None) is not None:
                # HFlip-TTA for ordinal regression is defined on cumulative
                # probabilities q, not on raw logits.
                ord_probs = 0.5 * (
                    torch.sigmoid(outputs['ordinal_logits'])
                    + torch.sigmoid(outputs_h['ordinal_logits'])
                )
                outputs['ordinal_logits'] = torch.logit(ord_probs.clamp(1e-6, 1.0 - 1e-6))

        y_true.extend(labels.cpu().tolist())
        if outputs.get('text_logits', None) is not None:
            text_logits_all.append(outputs['text_logits'].detach().cpu())
        if outputs.get('ordinal_logits', None) is not None:
            ordinal_logits_all.append(outputs['ordinal_logits'].detach().cpu())

    text_logits = torch.cat(text_logits_all, dim=0).numpy() if len(text_logits_all) > 0 else None
    ordinal_logits = torch.cat(ordinal_logits_all, dim=0).numpy() if len(ordinal_logits_all) > 0 else None
    return {
        'y_true': np.asarray(y_true, dtype=np.int64),
        'text_logits': text_logits,
        'ordinal_logits': ordinal_logits,
        'num_classes': int(num_classes),
    }


def _build_strategy_candidates(
    eval_outputs,
    prediction_source='ordinal',
    search_thresholds=False,
    threshold_candidates=None,
    fine_threshold_step=0.025,
    fine_threshold_radius=0.05,
    search_fusion=False,
    fusion_alpha_candidates=None,
    ordinal_type='coral',
):
    y_true = eval_outputs['y_true']
    num_classes = int(eval_outputs['num_classes'])
    text_logits = eval_outputs['text_logits']
    ordinal_logits = eval_outputs['ordinal_logits']

    candidates = []
    text_probs = _softmax_np(text_logits) if text_logits is not None else None
    ordinal_probs = _ordinal_class_probs_np(ordinal_logits, num_classes, ordinal_type=ordinal_type) if ordinal_logits is not None else None

    if text_probs is not None:
        candidates.append(
            {
                'name': 'text',
                'y_pred': np.argmax(text_probs, axis=1).astype(np.int64),
                'y_score': text_probs,
                'meta': {'source': 'text'},
            }
        )

    if ordinal_logits is not None:
        pred_default = _ordinal_pred_with_thresholds(
            ordinal_logits,
            [0.5] * (num_classes - 1),
            ordinal_type=ordinal_type,
        )
        candidates.append(
            {
                'name': 'ordinal',
                'y_pred': pred_default,
                'y_score': ordinal_probs,
                'meta': {'source': 'ordinal', 'thresholds': [0.5] * (num_classes - 1)},
            }
        )

        if bool(search_thresholds):
            best_thresholds, best_pred, _ = _search_best_ordinal_thresholds_two_stage(
                ord_logits=ordinal_logits,
                y_true=y_true,
                coarse_candidates=threshold_candidates,
                fine_step=fine_threshold_step,
                fine_radius=fine_threshold_radius,
                ordinal_type=ordinal_type,
            )
            candidates.append(
                {
                    'name': 'ordinal_tuned',
                    'y_pred': best_pred,
                    'y_score': ordinal_probs,
                    'meta': {'source': 'ordinal', 'thresholds': best_thresholds},
                }
            )

    if text_probs is not None and ordinal_probs is not None and bool(search_fusion):
        alphas = fusion_alpha_candidates or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for alpha in alphas:
            fused = float(alpha) * text_probs + (1.0 - float(alpha)) * ordinal_probs
            candidates.append(
                {
                    'name': f'fusion_{float(alpha):.2f}',
                    'y_pred': np.argmax(fused, axis=1).astype(np.int64),
                    'y_score': fused,
                    'meta': {'source': 'fusion', 'alpha': float(alpha)},
                }
            )

    if len(candidates) == 0:
        raise RuntimeError('no prediction candidate available for evaluation')
    return candidates


def _select_best_candidate(y_true, candidates, num_classes, include_confusion=False, include_predictions=False):
    best = None
    for candidate in candidates:
        metrics = compute_metrics(
            y_true.tolist(),
            candidate['y_pred'].tolist(),
            num_classes=num_classes,
            include_confusion=include_confusion,
            include_predictions=include_predictions,
            y_score=candidate['y_score'].tolist(),
        )
        metrics['strategy'] = candidate['name']
        metrics['strategy_meta'] = candidate['meta']
        score_key = (
            float(metrics['accuracy']),
            float(metrics.get('qwk', -1.0)),
            float(metrics.get('f1', -1.0)),
        )
        if best is None or score_key > best['score_key']:
            best = {'metrics': metrics, 'score_key': score_key}
    return best['metrics']


def _evaluate_with_strategy(
    eval_outputs,
    prediction_source='ordinal',
    num_classes=5,
    include_confusion=False,
    include_predictions=False,
    search_thresholds=False,
    threshold_candidates=None,
    fine_threshold_step=0.025,
    fine_threshold_radius=0.05,
    search_fusion=False,
    fusion_alpha_candidates=None,
    fixed_strategy=None,
    ordinal_type='coral',
):
    y_true = eval_outputs['y_true']
    num_classes = int(num_classes)
    text_logits = eval_outputs['text_logits']
    ordinal_logits = eval_outputs['ordinal_logits']

    if fixed_strategy is not None:
        source = str(fixed_strategy.get('source', prediction_source)).lower()
        candidates = []
        if source == 'text':
            text_probs = _softmax_np(text_logits)
            candidates.append(
                {
                    'name': 'text',
                    'y_pred': np.argmax(text_probs, axis=1).astype(np.int64),
                    'y_score': text_probs,
                    'meta': dict(fixed_strategy),
                }
            )
        elif source == 'ordinal':
            thresholds = fixed_strategy.get('thresholds', [0.5] * (num_classes - 1))
            ordinal_probs = _ordinal_class_probs_np(ordinal_logits, num_classes, ordinal_type=ordinal_type)
            candidates.append(
                {
                    'name': 'ordinal_fixed',
                    'y_pred': _ordinal_pred_with_thresholds(
                        ordinal_logits,
                        thresholds,
                        ordinal_type=ordinal_type,
                    ),
                    'y_score': ordinal_probs,
                    'meta': dict(fixed_strategy),
                }
            )
        elif source == 'fusion':
            alpha = float(fixed_strategy.get('alpha', 0.5))
            text_probs = _softmax_np(text_logits)
            ordinal_probs = _ordinal_class_probs_np(ordinal_logits, num_classes, ordinal_type=ordinal_type)
            fused = alpha * text_probs + (1.0 - alpha) * ordinal_probs
            candidates.append(
                {
                    'name': 'fusion_fixed',
                    'y_pred': np.argmax(fused, axis=1).astype(np.int64),
                    'y_score': fused,
                    'meta': dict(fixed_strategy),
                }
            )
        else:
            raise ValueError(f'unsupported fixed evaluation strategy source: {source}')
        return _select_best_candidate(
            y_true=y_true,
            candidates=candidates,
            num_classes=num_classes,
            include_confusion=include_confusion,
            include_predictions=include_predictions,
        )

    if str(prediction_source).lower() in ('search_ordinal', 'ordinal_search'):
        candidates = _build_strategy_candidates(
            eval_outputs=eval_outputs,
            prediction_source='search',
            search_thresholds=search_thresholds,
            threshold_candidates=threshold_candidates,
            fine_threshold_step=fine_threshold_step,
            fine_threshold_radius=fine_threshold_radius,
            search_fusion=False,
            fusion_alpha_candidates=fusion_alpha_candidates,
            ordinal_type=ordinal_type,
        )
        candidates = [c for c in candidates if c['meta'].get('source') == 'ordinal']
    elif str(prediction_source).lower() == 'search':
        candidates = _build_strategy_candidates(
            eval_outputs=eval_outputs,
            prediction_source=prediction_source,
            search_thresholds=search_thresholds,
            threshold_candidates=threshold_candidates,
            fine_threshold_step=fine_threshold_step,
            fine_threshold_radius=fine_threshold_radius,
            search_fusion=search_fusion,
            fusion_alpha_candidates=fusion_alpha_candidates,
            ordinal_type=ordinal_type,
        )
    else:
        candidates = _build_strategy_candidates(
            eval_outputs=eval_outputs,
            prediction_source=prediction_source,
            search_thresholds=False,
            threshold_candidates=threshold_candidates,
            fine_threshold_step=fine_threshold_step,
            fine_threshold_radius=fine_threshold_radius,
            search_fusion=False,
            fusion_alpha_candidates=fusion_alpha_candidates,
            ordinal_type=ordinal_type,
        )
        source = str(prediction_source).lower()
        if source == 'text':
            candidates = [c for c in candidates if c['meta'].get('source') == 'text']
        elif source == 'ordinal':
            candidates = [c for c in candidates if c['meta'].get('source') == 'ordinal' and c['name'] == 'ordinal']
        elif source == 'class':
            candidates = [c for c in candidates if c['meta'].get('source') == 'class']
        else:
            raise ValueError(f'unsupported prediction_source: {prediction_source}')

    return _select_best_candidate(
        y_true=y_true,
        candidates=candidates,
        num_classes=num_classes,
        include_confusion=include_confusion,
        include_predictions=include_predictions,
    )


def select_scores(outputs, prediction_source='ordinal', num_classes=5, ordinal_type='coral'):
    source = str(prediction_source).lower()
    ord_logits = outputs.get('ordinal_logits', None)
    text_logits = outputs.get('text_logits', None)

    if source == 'ordinal':
        if ord_logits is not None:
            if str(ordinal_type).lower() == 'corn':
                return _corn_class_probs_from_logits(ord_logits, num_classes=num_classes)
            return _ordinal_class_probs_from_logits(ord_logits, num_classes=num_classes)
        if text_logits is not None:
            return torch.softmax(text_logits, dim=1)
    elif source == 'text':
        if text_logits is not None:
            return torch.softmax(text_logits, dim=1)
        if ord_logits is not None:
            return _ordinal_class_probs_from_logits(ord_logits, num_classes=num_classes)
    else:
        raise ValueError(f'unsupported prediction_source: {prediction_source}')
    raise RuntimeError('no available score branch; enable text or ordinal branch')


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes=5,
    use_flip_ensemble=False,
    include_confusion=False,
    include_predictions=False,
    prediction_source='ordinal',
    search_thresholds=False,
    threshold_candidates=None,
    fine_threshold_step=0.025,
    fine_threshold_radius=0.05,
    search_fusion=False,
    fusion_alpha_candidates=None,
    fixed_strategy=None,
    ordinal_type='coral',
):
    eval_outputs = _collect_eval_outputs(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
        use_flip_ensemble=use_flip_ensemble,
    )
    return _evaluate_with_strategy(
        eval_outputs=eval_outputs,
        prediction_source=prediction_source,
        num_classes=num_classes,
        include_confusion=include_confusion,
        include_predictions=include_predictions,
        search_thresholds=search_thresholds,
        threshold_candidates=threshold_candidates,
        fine_threshold_step=fine_threshold_step,
        fine_threshold_radius=fine_threshold_radius,
        search_fusion=search_fusion,
        fusion_alpha_candidates=fusion_alpha_candidates,
        fixed_strategy=fixed_strategy,
        ordinal_type=ordinal_type,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    seed_everything(int(cfg['seed']))
    num_classes = int(cfg.get('num_classes', 5))
    prompts = load_prompts(
        cfg.get('clinical_root', ''),
        prompt_file=cfg.get('prompt_file', None),
        config_dir=config_dir,
        num_classes=num_classes,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_weights, class_counts = build_loaders(cfg)

    model = CLDGMEKOA(
        prompts,
        int(cfg['embed_dim']),
        bool(cfg['pretrained_backbone']),
        text_encoder_type=cfg.get('text_encoder_type', 'hf'),
        hf_text_model_name=cfg.get('hf_text_model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
        freeze_hf_text_encoder=bool(cfg.get('freeze_hf_text_encoder', False)),
        num_classes=num_classes,
        backbone_type=cfg.get('backbone_type', 'convnext_tiny'),
        use_ordinal_branch=bool(cfg.get('use_ordinal_branch', True)),
        use_text_branch=bool(cfg.get('use_text_branch', True)),
        use_dgme=bool(cfg.get('use_dgme', cfg.get('use_sleb', False))),
        dgme_stage=cfg.get('dgme_stage', cfg.get('sleb_stage', 'stage2')),
        dgme_dropout=float(cfg.get('dgme_dropout', cfg.get('sleb_dropout', 0.0))),
        dgme_use_ocab=bool(cfg.get('dgme_use_ocab', cfg.get('sleb_use_dssgb', True))),
        dgme_use_hrfab=bool(cfg.get('dgme_use_hrfab', cfg.get('sleb_use_mlcb', True))),
        ordinal_head_type=cfg.get('ordinal_head_type', 'oclr'),
    ).to(device)

    manual_class_weights = cfg.get('manual_class_weights', None)
    if manual_class_weights is not None:
        if len(manual_class_weights) != num_classes:
            raise ValueError(
                f'manual_class_weights must have {num_classes} values, got {len(manual_class_weights)}'
            )
        class_weights = torch.as_tensor(manual_class_weights, dtype=torch.float32, device=device)
    elif bool(cfg.get('use_weighted_loss', True)):
        class_weights = class_weights.to(device)
    else:
        class_weights = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg['epochs']))
    best_score = None
    best_metrics = {}
    history = []

    alpha = float(cfg.get('loss_alpha_text', 1.0))
    beta = float(cfg.get('loss_beta_ordinal', 1.0))
    gamma = float(cfg.get('loss_gamma_consistency', cfg.get('lambda_consistency', 10.0)))
    use_text_branch = bool(cfg.get('use_text_branch', True))
    use_ordinal_branch = bool(cfg.get('use_ordinal_branch', True))
    use_consistency_loss = bool(cfg.get('use_consistency_loss', True))
    consistency_target = cfg.get('consistency_target', 'text')
    coral_threshold_weights = cfg.get('coral_threshold_weights', None)
    prediction_source = cfg.get('prediction_source', 'ordinal')
    search_thresholds = bool(cfg.get('search_ordinal_thresholds', False))
    threshold_candidates = cfg.get('ordinal_threshold_coarse_candidates', cfg.get('ordinal_threshold_candidates', None))
    fine_threshold_step = float(cfg.get('ordinal_threshold_fine_step', 0.025))
    fine_threshold_radius = float(cfg.get('ordinal_threshold_fine_radius', 0.05))
    search_fusion = bool(cfg.get('search_text_ordinal_fusion', False))
    fusion_alpha_candidates = cfg.get('fusion_alpha_candidates', None)
    best_strategy = None
    selection_metric = str(cfg.get('selection_metric', 'accuracy')).lower()
    best_ckpt_path = os.path.join(cfg['output_dir'], 'best.pt')

    for epoch in range(int(cfg['epochs'])):
        model.train()
        running_total = 0.0
        running_text = 0.0
        running_ord = 0.0
        running_cons = 0.0

        for images, labels, _ in tqdm(train_loader, desc=f'epoch {epoch + 1}'):
            images = images.to(device)
            labels = labels.to(device)

            outputs_o = model(images, return_dict=True)
            outputs_h = model(torch.flip(images, dims=[3]), return_dict=True)
            loss_dict = compute_multibranch_loss(
                outputs_o,
                outputs_h,
                labels,
                num_classes=num_classes,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                class_weights=class_weights,
                label_smoothing=float(cfg.get('label_smoothing', 0.0)),
                use_text_branch=use_text_branch,
                use_ordinal_branch=use_ordinal_branch,
                use_consistency_loss=use_consistency_loss,
                coral_threshold_weights=coral_threshold_weights,
            )
            loss = loss_dict['total']
            if not loss.requires_grad:
                raise RuntimeError('all losses are disabled; at least one branch loss must be enabled')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total += float(loss_dict['total'].item())
            running_text += float(loss_dict['text'].item())
            running_ord += float(loss_dict['ordinal'].item())
            running_cons += float(loss_dict['consistency'].item())

        scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=num_classes,
            use_flip_ensemble=bool(cfg.get('use_flip_ensemble_eval', False)),
            include_confusion=False,
            include_predictions=False,
            prediction_source=prediction_source,
            search_thresholds=search_thresholds,
            threshold_candidates=threshold_candidates,
            fine_threshold_step=fine_threshold_step,
            fine_threshold_radius=fine_threshold_radius,
            search_fusion=search_fusion,
            fusion_alpha_candidates=fusion_alpha_candidates,
        )
        val_metrics['epoch'] = epoch + 1
        denom = max(1, len(train_loader))
        val_metrics['train_loss_total'] = running_total / denom
        val_metrics['train_loss_text'] = running_text / denom
        val_metrics['train_loss_ordinal'] = running_ord / denom
        val_metrics['train_loss_consistency'] = running_cons / denom
        history.append(val_metrics)
        auc_str = f"{val_metrics['auc']:.4f}" if val_metrics['auc'] is not None else 'nan'

        print(
            f"epoch={epoch + 1} "
            f"text={val_metrics['train_loss_text']:.4f} "
            f"ord={val_metrics['train_loss_ordinal']:.4f} "
            f"cons={val_metrics['train_loss_consistency']:.4f} "
            f"total={val_metrics['train_loss_total']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={auc_str} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_qwk={val_metrics['qwk']:.4f} "
            f"strategy={val_metrics.get('strategy', prediction_source)}"
        )

        improved, current_score = is_better_metric(val_metrics, best_score, selection_metric)
        if improved:
            best_score = current_score
            best_metrics = val_metrics
            best_strategy = val_metrics.get('strategy_meta', None)
            torch.save(
                {
                    'model': model.state_dict(),
                    'cfg': cfg,
                    'prompts': prompts,
                    'class_counts': class_counts,
                    'best_strategy': best_strategy,
                    'selection_metric': selection_metric,
                    'selection_score': best_score,
                },
                best_ckpt_path,
            )

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint['model'])
    best_strategy = best_checkpoint.get('best_strategy', best_strategy)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes=num_classes,
        use_flip_ensemble=bool(cfg.get('use_flip_ensemble_eval', False)),
        include_confusion=bool(cfg.get('save_confusion_matrix', True)),
        include_predictions=bool(cfg.get('save_eval_predictions', False)),
        prediction_source=prediction_source,
        search_thresholds=search_thresholds,
        threshold_candidates=threshold_candidates,
        fine_threshold_step=fine_threshold_step,
        fine_threshold_radius=fine_threshold_radius,
        search_fusion=search_fusion,
        fusion_alpha_candidates=fusion_alpha_candidates,
        fixed_strategy=best_strategy,
    )
    torch.save(
        {
            'model': model.state_dict(),
            'cfg': cfg,
            'prompts': prompts,
            'class_counts': class_counts,
            'best_strategy': best_strategy,
            'selection_metric': selection_metric,
            'selection_score': best_score,
        },
        os.path.join(cfg['output_dir'], 'last.pt'),
    )
    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'class_counts': class_counts,
                'val_history': history,
                'best_val': best_metrics,
                'selection_metric': selection_metric,
                'test': test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f'best_val_{selection_metric}=', best_score)
    print('best_val_qwk=', best_metrics.get('qwk', None))
    print('test_metrics=', test_metrics)


if __name__ == '__main__':
    main()
