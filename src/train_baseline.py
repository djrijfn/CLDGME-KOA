import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from .baselines import AdjustableOrdinalLoss, build_baseline_model, semixup_loss
    from .losses import coral_loss, decode_coral_logits
    from .metrics import compute_metrics
    from .train import build_loaders, is_better_metric
except ImportError:
    from baselines import AdjustableOrdinalLoss, build_baseline_model, semixup_loss
    from losses import coral_loss, decode_coral_logits
    from metrics import compute_metrics
    from train import build_loaders, is_better_metric


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _method_name(cfg):
    return str(cfg.get('method', cfg.get('baseline_method', 'deep_siamese_cnn'))).lower()


def _softmax_scores(logits):
    return torch.softmax(logits, dim=1)


def _is_coral_method(method):
    return method in (
        'ordinal_regression_module',
        'coral_ordinal',
        'coral',
        'ordinal_vit',
        'vit_ordinal',
        'vit_coral',
    )


def _ordinal_class_probs_from_logits(ord_logits, num_classes):
    num_classes = int(num_classes)
    p_gt = torch.sigmoid(ord_logits)
    probs = torch.zeros((ord_logits.shape[0], num_classes), dtype=ord_logits.dtype, device=ord_logits.device)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for c in range(1, num_classes - 1):
        probs[:, c] = p_gt[:, c - 1] - p_gt[:, c]
    probs[:, num_classes - 1] = p_gt[:, num_classes - 2]
    probs = torch.clamp(probs, min=0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def evaluate_classifier(
    model,
    loader,
    device,
    num_classes=5,
    include_confusion=False,
    include_predictions=False,
    use_flip_ensemble=False,
    max_batches=None,
    method='classifier',
):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    for batch_idx, (images, labels, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        if bool(use_flip_ensemble):
            logits_flip = model(torch.flip(images, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        if _is_coral_method(str(method).lower()):
            pred, _ = decode_coral_logits(logits)
            scores = _ordinal_class_probs_from_logits(logits, num_classes=num_classes)
        else:
            scores = _softmax_scores(logits)
            pred = torch.argmax(scores, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_score.extend(scores.cpu().tolist())

    return compute_metrics(
        y_true,
        y_pred,
        num_classes=num_classes,
        include_confusion=include_confusion,
        include_predictions=include_predictions,
        y_score=y_score,
    )


def build_loss_objects(cfg, device, class_weights):
    method = _method_name(cfg)
    use_weighted_loss = bool(cfg.get('use_weighted_loss', True))
    weights = class_weights.to(device) if use_weighted_loss and class_weights is not None else None

    ce_loss = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(cfg.get('label_smoothing', 0.0)),
    )
    ordinal_loss = AdjustableOrdinalLoss(
        num_classes=int(cfg.get('num_classes', 5)),
        distance_power=float(cfg.get('ordinal_distance_power', 1.0)),
        offset=float(cfg.get('ordinal_offset', 1.0)),
        squared=bool(cfg.get('ordinal_squared', True)),
    ).to(device)
    return {
        'method': method,
        'class_weights': weights,
        'ce': ce_loss,
        'ordinal': ordinal_loss,
    }


def train_one_batch(model, images, labels, loss_objects, cfg):
    method = loss_objects['method']
    if method in ('deep_siamese_cnn', 'siamese', 'deep_siamese'):
        logits = model(images)
        total = loss_objects['ce'](logits, labels)
        return {
            'total': total,
            'supervised': total.detach(),
            'ordinal': labels.new_tensor(0.0, dtype=torch.float32),
            'in_consistency': labels.new_tensor(0.0, dtype=torch.float32),
            'out_consistency': labels.new_tensor(0.0, dtype=torch.float32),
            'interpolation': labels.new_tensor(0.0, dtype=torch.float32),
        }

    if method in ('vgg19_ordinal', 'vgg19_adjustable_ordinal', 'adjustable_ordinal'):
        logits = model(images)
        ordinal = loss_objects['ordinal'](logits, labels)
        ce_weight = float(cfg.get('ordinal_ce_weight', 0.0))
        ce = loss_objects['ce'](logits, labels) if ce_weight > 0 else labels.new_tensor(0.0, dtype=torch.float32)
        total = ordinal + ce_weight * ce
        return {
            'total': total,
            'supervised': ce.detach(),
            'ordinal': ordinal.detach(),
            'in_consistency': labels.new_tensor(0.0, dtype=torch.float32),
            'out_consistency': labels.new_tensor(0.0, dtype=torch.float32),
            'interpolation': labels.new_tensor(0.0, dtype=torch.float32),
        }

    if method in ('semixup', 'semi_mixup'):
        return semixup_loss(
            model=model,
            images=images,
            labels=labels,
            class_weights=loss_objects['class_weights'],
            label_smoothing=float(cfg.get('label_smoothing', 0.0)),
            alpha=float(cfg.get('semixup_alpha', 0.75)),
            lambda_in=float(cfg.get('semixup_lambda_in', 1.0)),
            lambda_out=float(cfg.get('semixup_lambda_out', 1.0)),
            lambda_interp=float(cfg.get('semixup_lambda_interp', 1.0)),
            noise_std=float(cfg.get('semixup_noise_std', 0.02)),
        )

    if _is_coral_method(method):
        logits = model(images)
        total = coral_loss(logits, labels, num_classes=int(cfg.get('num_classes', 5)))
        return {
            'total': total,
            'supervised': labels.new_tensor(0.0, dtype=torch.float32),
            'ordinal': total.detach(),
            'in_consistency': labels.new_tensor(0.0, dtype=torch.float32),
            'out_consistency': labels.new_tensor(0.0, dtype=torch.float32),
            'interpolation': labels.new_tensor(0.0, dtype=torch.float32),
        }

    logits = model(images)
    total = loss_objects['ce'](logits, labels)
    return {
        'total': total,
        'supervised': total.detach(),
        'ordinal': labels.new_tensor(0.0, dtype=torch.float32),
        'in_consistency': labels.new_tensor(0.0, dtype=torch.float32),
        'out_consistency': labels.new_tensor(0.0, dtype=torch.float32),
        'interpolation': labels.new_tensor(0.0, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_dir'], exist_ok=True)
    seed_everything(int(cfg.get('seed', 42)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(cfg.get('num_classes', 5))
    train_loader, val_loader, test_loader, class_weights, class_counts = build_loaders(cfg)
    model = build_baseline_model(cfg).to(device)
    loss_objects = build_loss_objects(cfg, device, class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get('lr', 1e-4)),
        weight_decay=float(cfg.get('weight_decay', 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get('epochs', 20)))

    selection_metric = str(cfg.get('selection_metric', 'accuracy')).lower()
    best_score = None
    best_metrics = {}
    history = []
    best_ckpt_path = os.path.join(cfg['output_dir'], 'best.pt')
    max_train_batches = cfg.get('max_train_batches', None)
    max_eval_batches = cfg.get('max_eval_batches', None)
    method = _method_name(cfg)

    for epoch in range(int(cfg.get('epochs', 20))):
        model.train()
        running = {
            'total': 0.0,
            'supervised': 0.0,
            'ordinal': 0.0,
            'in_consistency': 0.0,
            'out_consistency': 0.0,
            'interpolation': 0.0,
        }
        num_batches = 0

        for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader, desc=f'epoch {epoch + 1}')):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            images = images.to(device)
            labels = labels.to(device)

            loss_dict = train_one_batch(model, images, labels, loss_objects, cfg)
            loss = loss_dict['total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1
            for key in running:
                value = loss_dict.get(key)
                if value is not None:
                    running[key] += float(value.item())

        scheduler.step()

        val_metrics = evaluate_classifier(
            model,
            val_loader,
            device,
            num_classes=num_classes,
            include_confusion=False,
            include_predictions=False,
            use_flip_ensemble=bool(cfg.get('use_flip_ensemble_eval', False)),
            max_batches=max_eval_batches,
            method=method,
        )
        denom = max(1, num_batches)
        val_metrics['epoch'] = epoch + 1
        for key, value in running.items():
            val_metrics[f'train_loss_{key}'] = value / denom
        history.append(val_metrics)

        auc_str = f"{val_metrics['auc']:.4f}" if val_metrics['auc'] is not None else 'nan'
        print(
            f"epoch={epoch + 1} "
            f"total={val_metrics['train_loss_total']:.4f} "
            f"sup={val_metrics['train_loss_supervised']:.4f} "
            f"ord={val_metrics['train_loss_ordinal']:.4f} "
            f"in={val_metrics['train_loss_in_consistency']:.4f} "
            f"out={val_metrics['train_loss_out_consistency']:.4f} "
            f"interp={val_metrics['train_loss_interpolation']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={auc_str} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_qwk={val_metrics['qwk']:.4f}"
        )

        improved, current_score = is_better_metric(val_metrics, best_score, selection_metric)
        if improved:
            best_score = current_score
            best_metrics = val_metrics
            torch.save(
                {
                    'model': model.state_dict(),
                    'cfg': cfg,
                    'class_counts': class_counts,
                    'selection_metric': selection_metric,
                    'selection_score': best_score,
                },
                best_ckpt_path,
            )

    if not os.path.isfile(best_ckpt_path):
        raise RuntimeError('no checkpoint was saved; check training loop and validation data')

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint['model'])
    test_metrics = evaluate_classifier(
        model,
        test_loader,
        device,
        num_classes=num_classes,
        include_confusion=bool(cfg.get('save_confusion_matrix', True)),
        include_predictions=bool(cfg.get('save_eval_predictions', False)),
        use_flip_ensemble=bool(cfg.get('use_flip_ensemble_eval', False)),
        max_batches=max_eval_batches,
        method=method,
    )
    torch.save(
        {
            'model': model.state_dict(),
            'cfg': cfg,
            'class_counts': class_counts,
            'selection_metric': selection_metric,
            'selection_score': best_score,
        },
        os.path.join(cfg['output_dir'], 'last.pt'),
    )
    with open(os.path.join(cfg['output_dir'], 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'method': _method_name(cfg),
                'class_counts': class_counts,
                'val_history': history,
                'best_val': best_metrics,
                'selection_metric': selection_metric,
                'test': test_metrics,
                'notes': cfg.get('notes', ''),
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
