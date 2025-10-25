from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


def batch_dice(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute mean Dice over batch and classes.

    logits: (N,C,H,W); targets: (N,H,W)
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_oh, dims)
    cardinality = torch.sum(probs + targets_oh, dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean()


def batch_iou(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_oh, dims)
    union = torch.sum(probs + targets_oh - probs * targets_oh, dims)
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def _to_binary_masks(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    # pred: (N,C,H,W) logits; target: (N,H,W)
    pred_label = pred.softmax(1).argmax(1)
    pred_oh = F.one_hot(pred_label, num_classes=num_classes).permute(0, 3, 1, 2).cpu().numpy().astype(np.bool_)
    targ_oh = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).cpu().numpy().astype(np.bool_)
    return pred_oh, targ_oh


def hausdorff95_and_assd(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Compute HD95 and ASSD averaged across batch and classes.
    Uses binary surface distances per class. Assumes 2D slices with unit spacing.
    """
    n, c, h, w = logits.shape
    pred_oh, targ_oh = _to_binary_masks(logits, targets, c)

    def surface_distances(a: np.ndarray, b: np.ndarray):
        # a,b: (H,W) boolean masks
        if not a.any() and not b.any():
            return np.array([0.0]), np.array([0.0])
        if not a.any():
            a, b = b, a  # swap; empty mask distances measured to b's surface
        # boundaries: pixels where mask minus eroded mask
        from scipy.ndimage import binary_erosion

        a_border = a ^ binary_erosion(a)
        b_border = b ^ binary_erosion(b)
        # distance transform on complement of borders gives distance to nearest border
        dt_a = distance_transform_edt(~a_border)
        dt_b = distance_transform_edt(~b_border)
        # distances from a border to b border
        d_ab = dt_b[a_border]
        d_ba = dt_a[b_border]
        return d_ab, d_ba

    hd95_all = []
    assd_all = []
    for i in range(n):
        for k in range(1, c):  # skip background class 0 in boundary metrics
            d_ab, d_ba = surface_distances(pred_oh[i, k], targ_oh[i, k])
            dists = np.concatenate([d_ab, d_ba])
            if dists.size == 0:
                continue
            hd95 = np.percentile(dists, 95)
            assd = dists.mean()
            hd95_all.append(hd95)
            assd_all.append(assd)
    hd95_mean = float(np.mean(hd95_all)) if hd95_all else 0.0
    assd_mean = float(np.mean(assd_all)) if assd_all else 0.0
    return hd95_mean, assd_mean


def sensitivity_specificity(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Compute sensitivity (recall) and specificity averaged across classes and batch."""
    n, c, h, w = logits.shape
    pred_label = logits.softmax(1).argmax(1)
    sens_all = []
    spec_all = []
    for k in range(1, c):  # ignore background for class metrics
        p = (pred_label == k)
        t = (targets == k)
        tp = (p & t).sum().item()
        fn = (~p & t).sum().item()
        tn = (~p & ~t).sum().item()
        fp = (p & ~t).sum().item()
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        sens_all.append(sens)
        spec_all.append(spec)
    return float(np.mean(sens_all) if sens_all else 0.0), float(np.mean(spec_all) if spec_all else 0.0)

