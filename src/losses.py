import torch
import torch.nn.functional as F


def jsd_loss(logits_a, logits_b, eps=1e-8):
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(p.clamp_min(eps)), m.clamp_min(eps), reduction='batchmean')
    kl_qm = F.kl_div(torch.log(q.clamp_min(eps)), m.clamp_min(eps), reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)


def coral_targets(labels, num_classes):
    thresholds = torch.arange(int(num_classes) - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).to(dtype=torch.float32)


def coral_loss(logits, labels, num_classes, threshold_weights=None):
    """Binary cross-entropy over the K-1 cumulative ordinal tasks (OCLR objective)."""
    targets = coral_targets(labels, num_classes)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if threshold_weights is not None:
        weights = torch.as_tensor(threshold_weights, dtype=loss.dtype, device=loss.device)
        if weights.numel() != logits.shape[1]:
            raise ValueError(f'threshold_weights must have {logits.shape[1]} values, got {weights.numel()}')
        loss = loss * weights.view(1, -1)
    return loss.mean()


def decode_coral_logits(logits):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).sum(dim=1).to(dtype=torch.long)
    return pred, probs


def compute_multibranch_loss(
    outputs_orig,
    outputs_flip,
    labels,
    num_classes,
    alpha=1.0,
    beta=1.0,
    gamma=10.0,
    class_weights=None,
    label_smoothing=0.0,
    use_text_branch=True,
    use_ordinal_branch=True,
    use_consistency_loss=True,
    coral_threshold_weights=None,
):
    device = labels.device
    zero = labels.new_tensor(0.0, dtype=torch.float32, device=device)

    text_loss = zero
    if bool(use_text_branch):
        logits_orig = outputs_orig.get('text_logits', None)
        logits_flip = outputs_flip.get('text_logits', None)
        if logits_orig is not None and logits_flip is not None:
            ce_orig = F.cross_entropy(
                logits_orig,
                labels,
                weight=class_weights,
                label_smoothing=float(label_smoothing),
            )
            ce_flip = F.cross_entropy(
                logits_flip,
                labels,
                weight=class_weights,
                label_smoothing=float(label_smoothing),
            )
            text_loss = 0.5 * (ce_orig + ce_flip)

    ordinal_loss = zero
    if bool(use_ordinal_branch):
        ord_orig = outputs_orig.get('ordinal_logits', None)
        ord_flip = outputs_flip.get('ordinal_logits', None)
        if ord_orig is not None and ord_flip is not None:
            ord_a = coral_loss(
                ord_orig,
                labels,
                num_classes=num_classes,
                threshold_weights=coral_threshold_weights,
            )
            ord_b = coral_loss(
                ord_flip,
                labels,
                num_classes=num_classes,
                threshold_weights=coral_threshold_weights,
            )
            ordinal_loss = 0.5 * (ord_a + ord_b)

    consistency_loss = zero
    if bool(use_consistency_loss):
        logits_orig = outputs_orig.get('text_logits', None)
        logits_flip = outputs_flip.get('text_logits', None)
        if logits_orig is not None and logits_flip is not None:
            consistency_loss = jsd_loss(logits_orig, logits_flip)

    total = (
        float(alpha) * text_loss
        + float(beta) * ordinal_loss
        + float(gamma) * consistency_loss
    )
    return {
        'total': total,
        'text': text_loss,
        'ordinal': ordinal_loss,
        'consistency': consistency_loss,
    }
