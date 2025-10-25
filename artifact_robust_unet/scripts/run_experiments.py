from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # Adjust paths if needed
    cfg = "configs/config.yaml"
    runs = Path("runs")
    runs.mkdir(parents=True, exist_ok=True)

    # Train UNet
    run(["python", "-m", "artifact_robust_unet.training.train", "--model", "unet", "--config", cfg, "--output_dir", str(runs / "unet_clean")])
    # Train Attention UNet
    run(["python", "-m", "artifact_robust_unet.training.train", "--model", "attention_unet", "--config", cfg, "--output_dir", str(runs / "attn_clean")])


if __name__ == "__main__":
    main()
