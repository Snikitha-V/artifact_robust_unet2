from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from ..evaluation.validate import load_model
from ..data.acdc_dataset import ACDCSliceDataset
from ..artifacts.artifact_augment import ArtifactAugmentDataset, ArtifactConfig
from .metrics import batch_dice, batch_iou, hausdorff95_and_assd, sensitivity_specificity


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, limit: int | None = None) -> dict:
    model.eval()
    dices = []
    ious = []
    hd95s = []
    assds = []
    sens_all = []
    spec_all = []
    with torch.no_grad():
        for b_idx, (imgs, labels) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            dices.append(batch_dice(logits, labels).cpu())
            ious.append(batch_iou(logits, labels).cpu())
            hd95, assd = hausdorff95_and_assd(logits.cpu(), labels.cpu())
            sens, spec = sensitivity_specificity(logits, labels)
            hd95s.append(torch.tensor(hd95))
            assds.append(torch.tensor(assd))
            sens_all.append(torch.tensor(sens))
            spec_all.append(torch.tensor(spec))
            if limit and b_idx >= limit:
                break
    def mean(ts):
        return float(torch.stack(ts).mean().item()) if ts else 0.0
    return {
        "dice": mean(dices),
        "iou": mean(ious),
        "hd95": mean(hd95s),
        "assd": mean(assds),
        "sensitivity": mean(sens_all),
        "specificity": mean(spec_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit_batches", type=int, default=10, help="Limit batches to speed up evaluation")
    ap.add_argument("--artifact_prob", type=float, default=0.8)
    ap.add_argument("--artifact_severity", type=float, default=0.3)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds_dir = Path(cfg["data"]["dataset_dir"])  # type: ignore
    img_size = tuple(cfg["data"].get("image_size", [256, 256]))  # type: ignore
    num_classes = int(cfg.get("model", {}).get("num_classes", 4))

    base_ds = ACDCSliceDataset(ds_dir, image_size=img_size, num_classes=num_classes)
    clean_loader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    art_cfg = ArtifactConfig(
        prob=args.artifact_prob,
        gaussian_noise_std=args.artifact_severity * 0.1,
        rician_noise_std=args.artifact_severity * 0.05,
        bias_strength=args.artifact_severity * 0.3,
        rot_deg=5.0 * args.artifact_severity * 3,
        trans_pix=5.0 * args.artifact_severity * 3,
    )
    art_ds = ArtifactAugmentDataset(base_ds, art_cfg)
    art_loader = DataLoader(art_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint)).to(device)

    results = {
        "clean": evaluate(model, clean_loader, device, limit=args.limit_batches),
        "artifact": evaluate(model, art_loader, device, limit=args.limit_batches),
    }

    out_dir = Path("runs/comprehensive")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print({"saved": str(out_path)})


if __name__ == "__main__":
    main()
