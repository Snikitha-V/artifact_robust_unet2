from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import os

from ..models import UNet, AttentionUNet
from ..data.acdc_dataset import ACDCSliceDataset
from .loss_functions import CombinedLoss
from ..evaluation.metrics import batch_dice


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def build_model(name: str, in_ch: int, num_classes: int) -> nn.Module:
    if name.lower() == "unet":
        return UNet(in_channels=in_ch, num_classes=num_classes)
    if name.lower() in {"attention_unet", "attnunet", "attn_unet"}:
        return AttentionUNet(in_channels=in_ch, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "attention_unet"])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output_dir", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_batches", type=int, default=0, help="Limit training batches per epoch (0=no limit)")
    ap.add_argument("--max_val_batches", type=int, default=0, help="Limit validation batches per epoch (0=no limit)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds_dir = Path(cfg.get("data", {}).get("dataset_dir", "./data"))
    img_size = tuple(cfg.get("data", {}).get("image_size", [256, 256]))
    in_ch = int(cfg.get("model", {}).get("in_channels", 1))
    num_classes = int(cfg.get("model", {}).get("num_classes", 4))

    ds = ACDCSliceDataset(dataset_dir=ds_dir, image_size=img_size, num_classes=num_classes)
    n_val = max(1, int(len(ds) * float(args.val_split)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=bool(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=bool(args.num_workers))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if device.type == "cuda":
        th.backends.cudnn.benchmark = True
    model = build_model(args.model, in_ch, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CombinedLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    # Optional: compile the model for better GPU utilization (PyTorch 2.x)
    # Disabled by default on Windows due to Triton/Inductor requirements; enable by setting ENABLE_TORCH_COMPILE=1
    enable_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
    if enable_compile and getattr(th, "compile", None):
        try:
            model = th.compile(model)
        except Exception:
            # Fall back to eager if compilation fails
            pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    best_ckpt = out_dir / "best.ckpt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for b_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            if args.max_batches and b_idx >= args.max_batches:
                break

        # validation
        model.eval()
        dices = []
        with th.no_grad():
            for v_idx, (imgs, labels) in enumerate(val_loader, start=1):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                dices.append(batch_dice(logits, labels))
                if args.max_val_batches and v_idx >= args.max_val_batches:
                    break
        mean_dice = float(th.stack(dices).mean().item()) if dices else 0.0
        epoch_loss = running_loss / max(1, len(train_loader.dataset))
        print(f"Epoch {epoch}: loss={epoch_loss:.4f} val_dice={mean_dice:.4f}")

        # save best
        if mean_dice > best_dice:
            best_dice = mean_dice
            th.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "in_channels": in_ch,
                "num_classes": num_classes,
            }, best_ckpt)
            print(f"Saved best checkpoint -> {best_ckpt}")


if __name__ == "__main__":
    main()
