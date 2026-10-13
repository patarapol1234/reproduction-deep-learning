#!/usr/bin/env python3
"""Record environment and check inputs before invoking the official program."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import PLAN, ROOT, sha256, verify_official, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--batch128", action="store_true")
    args = parser.parse_args()
    provenance = verify_official()
    import numpy as np
    import torch
    import yaml
    from autoattack import AutoAttack  # RobustBench's import dependency.
    sys.path.insert(0, str(ROOT / "official"))
    import cifar10c  # Check the actual official entrypoint imports successfully.

    if args.require_cuda:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("The unchanged official cifar10c.py requires a CUDA GPU")
            # Check an actual CUDA allocation and kernel before the full experiment.
            probe = torch.ones(8, device="cuda")
            if (probe * 2).sum().item() != 16:
                raise RuntimeError("Unexpected CUDA computation result")
            torch.cuda.synchronize()
            del probe
        except Exception as error:
            write_json(args.output, {
                "status": "failed", "stage": "cuda_preflight",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_build": torch.version.cuda,
                "official_commit": provenance["commit"],
                "official_files_verified": True,
                "error_type": type(error).__name__, "error": str(error),
            })
            raise
    for method in PLAN["methods"]:
        cfg = yaml.safe_load((ROOT / "official/cfgs" / f"{method}.yaml").read_text())
        if cfg["MODEL"] != {"ADAPTATION": method, "ARCH": "Standard"}:
            raise RuntimeError(f"Unexpected official model configuration: {method}")
        if cfg["CORRUPTION"]["TYPE"] != PLAN["corruptions"]:
            raise RuntimeError("Corruption order differs from the official YAML")
    files = []
    for name in PLAN["corruptions"]:
        path = args.data_dir / "CIFAR-10-C" / f"{name}.npy"
        images = np.load(path, mmap_mode="r")
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise RuntimeError(f"Unexpected CIFAR-10-C shape/dtype: {path}")
        files.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                      "shape": list(images.shape), "dtype": str(images.dtype)})
    labels = args.data_dir / "CIFAR-10-C/labels.npy"
    y = np.load(labels, mmap_mode="r")
    if y.ndim != 1 or len(y) < 10000 or y[:10000].min() < 0 or y[:10000].max() > 9:
        raise RuntimeError("Unexpected CIFAR-10-C labels")
    checkpoint = args.model_dir / "cifar10/corruptions/Standard.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint missing: {checkpoint}; run slurm/prepare_assets.sh")
    packages = {}
    for name in ("torch", "torchvision", "robustbench", "numpy", "yacs", "iopath",
                 "setuptools", "timm", "gdown", "matplotlib", "PyYAML"):
        packages[name] = importlib.metadata.version(name)
    if packages["robustbench"] != "1.1":
        raise RuntimeError("Use robustbench==1.1 for the declared v2 software stack")
    payload = {
        "status": "prepared", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "packages": packages,
        "official_commit": provenance["commit"], "official_files_verified": True,
        "official_entrypoint": str(Path(cifar10c.__file__).resolve()),
        "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
        "cuda_compute_check": "passed" if args.require_cuda else "not_requested",
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "batch_sizes": [200, 128] if args.batch128 else [200],
        "corruption_files": files,
        "labels": {"path": str(labels.resolve()), "sha256": sha256(labels)},
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
    }
    write_json(args.output, payload)
    print(f"Inputs and official imports ready; CUDA={payload['cuda_available']}")


if __name__ == "__main__":
    main()
