import torch
import torch.nn as nn
from .blocks import DoubleConv, Down, Up, OutConv
from .attention_gate import AttentionGate


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_ch: int = 64, bilinear: bool = True):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        # Encoder
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, (base_ch * 16) // factor)

        # Attention gates for skip connections (skip gets attended by gating from decoder)
        self.gate3 = AttentionGate(in_ch_skip=base_ch * 8, in_ch_gating=base_ch * 16 // factor)
        self.gate2 = AttentionGate(in_ch_skip=base_ch * 4, in_ch_gating=base_ch * 8 // factor)
        self.gate1 = AttentionGate(in_ch_skip=base_ch * 2, in_ch_gating=base_ch * 4 // factor)
        self.gate0 = AttentionGate(in_ch_skip=base_ch, in_ch_gating=base_ch * 2 // factor)

        # Decoder
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)
        self.outc = OutConv(base_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # attention on skip features
        x3_att = self.gate3(x3, x4)
        u1 = self.up1(x4, x3_att)

        x2_att = self.gate2(x2, u1)
        u2 = self.up2(u1, x2_att)

        x1_att = self.gate1(x1, u2)
        u3 = self.up3(u2, x1_att)

        x0_att = self.gate0(x0, u3)
        u4 = self.up4(u3, x0_att)

        logits = self.outc(u4)
        return logits
