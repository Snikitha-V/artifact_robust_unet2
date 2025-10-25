import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-class Dice loss. logits: (N,C,H,W); targets: (N,H,W) long labels."""
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        cardinality = torch.sum(probs + targets_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # average across classes (ignore background optional)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = ce_weight
        self.w_dice = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_ce * self.ce(logits, targets) + self.w_dice * self.dice(logits, targets)
