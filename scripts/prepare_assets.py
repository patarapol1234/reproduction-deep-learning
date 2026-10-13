#!/usr/bin/env python3
"""Use RobustBench to reuse or download the official example's assets."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    from robustbench.data import load_cifar10c
    from robustbench.model_zoo.enums import ThreatModel
    from robustbench.utils import load_model
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    load_model("Standard", args.model_dir, "cifar10", ThreatModel.corruptions)
    load_cifar10c(1, 5, args.data_dir, False, ["gaussian_noise"])
    print("Standard checkpoint and CIFAR-10-C available")


if __name__ == "__main__":
    main()
