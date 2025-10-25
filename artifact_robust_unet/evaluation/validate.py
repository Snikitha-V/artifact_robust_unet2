from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models import UNet, AttentionUNet
from ..data.acdc_dataset import ACDCSliceDataset
from .metrics import batch_dice


def load_model(ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model", "unet")
    in_ch = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 4))
    if model_name == "unet":
        model = UNet(in_channels=in_ch, num_classes=num_classes)
    else:
        model = AttentionUNet(in_channels=in_ch, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds_dir = Path(cfg.get("data", {}).get("dataset_dir", "./data"))
    img_size = tuple(cfg.get("data", {}).get("image_size", [256, 256]))
    num_classes = int(cfg.get("model", {}).get("num_classes", 4))

    ds = ACDCSliceDataset(dataset_dir=ds_dir, image_size=img_size, num_classes=num_classes)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint)).to(device)
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            dices.append(batch_dice(logits, labels))
    mean_dice = float(torch.stack(dices).mean().item()) if dices else 0.0
    print({"mean_dice": mean_dice})


if __name__ == "__main__":
    main()
