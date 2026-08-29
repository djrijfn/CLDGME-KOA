from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    num_classes=5,
    include_confusion=False,
    include_predictions=False,
    y_score=None,
):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    auc = None
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='macro'))
        except ValueError:
            auc = None
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'auc': auc,
        'f1': float(f1),
        'macro_precision': float(p),
        'macro_recall': float(r),
        'macro_f1': float(f1),
        'qwk': float(cohen_kappa_score(y_true, y_pred, weights='quadratic')),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }
    if include_confusion:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes))))
        out['confusion_matrix'] = cm.tolist()
    if include_predictions:
        out['y_true'] = list(map(int, y_true))
        out['y_pred'] = list(map(int, y_pred))
    return out
