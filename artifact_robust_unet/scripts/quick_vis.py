from __future__ import annotations

from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt

from ..evaluation.validate import load_model
from ..data.acdc_dataset import ACDCSliceDataset


def main():
    cfg = yaml.safe_load(Path("artifact_robust_unet/configs/config.yaml").read_text(encoding="utf-8"))
    ds = ACDCSliceDataset(cfg["data"]["dataset_dir"], tuple(cfg["data"]["image_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path("runs/unet_clean/best.ckpt")).to(device).eval()

    img, lbl = ds[0]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pm = pred.softmax(1).argmax(1).squeeze(0).cpu()

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(img.squeeze(0).numpy(), cmap="gray")
    axs[0].set_title("Image")
    axs[1].imshow(lbl.numpy(), vmin=0, vmax=3)
    axs[1].set_title("Label")
    axs[2].imshow(pm.numpy(), vmin=0, vmax=3)
    axs[2].set_title("Pred")
    for a in axs:
        a.axis("off")
    out = Path("runs/unet_clean/sample_pred.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    print({"saved": str(out)})


if __name__ == "__main__":
    main()
