from __future__ import annotations
import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Simple additive attention gate for UNet skip connections.

    Given a gating signal g (decoder) and skip feature x (encoder), compute an attention map
    and modulate x before concatenation.
    """

    def __init__(self, in_ch_skip: int, in_ch_gating: int, inter_ch: int | None = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(1, in_ch_skip // 2)

        self.theta_x = nn.Sequential(
            nn.Conv2d(in_ch_skip, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.phi_g = nn.Sequential(
            nn.Conv2d(in_ch_gating, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta = self.theta_x(x)
        phi = self.phi_g(g)
        # upsample phi to theta size if needed
        if phi.shape[-2:] != theta.shape[-2:]:
            phi = nn.functional.interpolate(phi, size=theta.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.act(theta + phi))
        return x * attn
