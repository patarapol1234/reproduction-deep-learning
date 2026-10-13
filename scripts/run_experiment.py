#!/usr/bin/env python3
"""Launch unchanged official cifar10c.py and retain its command and logs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import PLAN, ROOT, parse_errors, sha256, verify_official, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=PLAN["methods"])
    parser.add_argument("--batch-size", type=int, choices=(200, 128), default=200)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verify_official()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    official_output = run_dir / "official-output"
    command = [
        sys.executable, str(ROOT / "official/cifar10c.py"),
        "--cfg", str(ROOT / "official/cfgs" / f"{args.method}.yaml"),
        "DATA_DIR", str(args.data_dir.resolve()),
        "CKPT_DIR", str(args.model_dir.resolve()),
        "SAVE_DIR", str(official_output),
        "CORRUPTION.SEVERITY", "[5]", "CORRUPTION.NUM_EX", "10000",
        "TEST.BATCH_SIZE", str(args.batch_size), "RNG_SEED", "1",
        "CUDNN.BENCHMARK", "True",
    ]
    record = {
        "status": "running", "official_commit": PLAN["official_commit"],
        "method": args.method, "batch_size": args.batch_size,
        "seed": PLAN["seed"], "shuffle": False, "severity": 5,
        "num_examples_per_corruption": 10000, "corruptions": PLAN["corruptions"],
        "command": command, "cwd": str(ROOT / "official"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "console_log": "console.log", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(run_dir / "run.json", record)
    print(f"Starting official {args.method}, B={args.batch_size}: {run_dir}", flush=True)
    started = time.monotonic()
    try:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        with (run_dir / "console.log").open("w") as console:
            result = subprocess.run(command, cwd=ROOT / "official", env=environment,
                                    stdout=console, stderr=subprocess.STDOUT, check=False)
        record["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Official program failed; inspect {run_dir / 'console.log'}")
        rows = parse_errors(run_dir / "console.log")
        verify_official()
        record.update(status="completed", corruption_rows=len(rows),
                      mean_error_percent=sum(r["error_percent"] for r in rows) / len(rows))
    except BaseException as error:
        record.update(status="failed", failure=str(error))
        raise
    finally:
        record["process_seconds"] = time.monotonic() - started
        record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["console_sha256"] = sha256(run_dir / "console.log") if (run_dir / "console.log").exists() else None
        write_json(run_dir / "run.json", record)
    print(f"Completed: mean error={record['mean_error_percent']:.3f}%", flush=True)


if __name__ == "__main__":
    main()
