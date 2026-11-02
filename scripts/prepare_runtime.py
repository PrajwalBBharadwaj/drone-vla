#!/usr/bin/env python3
"""Validate the chosen PyTorch runtime and optionally download/preload model weights."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_torch(profile):
    import torch
    import torchvision
    if torch.__version__.split("+")[0] != "2.8.0" or torchvision.__version__.split("+")[0] != "0.23.0":
        raise RuntimeError("Expected torch 2.8.0 and torchvision 0.23.0")
    gpu = profile != "cpu"
    if gpu:
        if torch.version.cuda != "12.6" or not torch.cuda.is_available():
            raise RuntimeError(f"Expected working CUDA 12.6; got {torch.version.cuda}, available={torch.cuda.is_available()}")
        torch.ones(1, device="cuda").sum().item()  # Verify an actual GPU operation, not just device discovery.
    elif torch.version.cuda is not None:
        # A CUDA build left over from a prior install is not the requested CPU profile.
        raise RuntimeError("CPU profile requires the CPU PyTorch wheels")
    print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if gpu else None}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda126", "jetson-cu126"), required=True)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--navigation", action="store_true", help="Preload the small metric outdoor depth model")
    args = parser.parse_args()
    check_torch(args.profile)
    if args.preload:
        os.environ["VLA_DEPTH_DEVICE"] = "cpu" if args.profile == "cpu" else "cuda:0"
        if args.navigation:
            from navigation_perception import MetricDepth
            from PIL import Image
            MetricDepth()(Image.new("RGB", (448, 252)))
        else:
            from vision_test import _get_depth_pipeline
            _get_depth_pipeline()
        print("Depth model ready", flush=True)


if __name__ == "__main__":
    main()
