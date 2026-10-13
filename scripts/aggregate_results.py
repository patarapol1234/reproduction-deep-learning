#!/usr/bin/env python3
"""Summarize official log percentages after all planned runs complete."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import PLAN, ROOT, parse_errors, sha256, verify_official, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    verify_official()
    directory = args.results_dir.resolve()
    environment = json.loads((directory / "environment.json").read_text())
    if not environment["cuda_available"] or not environment["official_files_verified"]:
        raise RuntimeError("A CUDA preflight record is required for experimental results")
    if environment["official_commit"] != PLAN["official_commit"]:
        raise RuntimeError("Unexpected official commit in the run environment")
    expected = {(batch, method) for batch in environment["batch_sizes"] for method in PLAN["methods"]}
    if environment["batch_sizes"] not in ([200], [200, 128]):
        raise RuntimeError("Unexpected batch-size matrix")
    observed, corruption_rows, run_rows = set(), [], []
    for path in sorted((directory / "runs").rglob("run.json")):
        record = json.loads(path.read_text())
        key = (record["batch_size"], record["method"])
        if key not in expected or key in observed or record["status"] != "completed":
            raise RuntimeError(f"Unexpected, duplicate, or incomplete run: {path}")
        if (record["official_commit"] != PLAN["official_commit"] or record["seed"] != 1
                or record["shuffle"] or record["severity"] != 5
                or record["num_examples_per_corruption"] != 10000
                or record["corruptions"] != PLAN["corruptions"]):
            raise RuntimeError(f"Protocol mismatch: {path}")
        if record["slurm_job_id"] != environment["slurm_job_id"]:
            raise RuntimeError(f"Run belongs to a different Slurm job: {path}")
        log = path.parent / record["console_log"]
        if sha256(log) != record["console_sha256"]:
            raise RuntimeError(f"Log changed after experiment completion: {log}")
        rows = parse_errors(log)
        mean = sum(row["error_percent"] for row in rows) / len(rows)
        if abs(mean - record["mean_error_percent"]) > 1e-9:
            raise RuntimeError(f"Recorded mean differs from official log: {path}")
        observed.add(key)
        for row in rows:
            corruption_rows.append({"batch_size": key[0], "method": key[1], **row,
                                    "source_log": str(log.relative_to(directory))})
        reference = PLAN["reference_mean_error_percent"][key[1]] if key[0] == 200 else None
        run_rows.append({
            "batch_size": key[0], "method": key[1], "corruptions": len(rows),
            "mean_error_percent": mean, "reference_error_percent": reference,
            "difference_pp": mean - reference if reference is not None else None,
            "process_seconds": record["process_seconds"],
        })
    if observed != expected:
        raise RuntimeError(f"Missing planned runs: {expected - observed}")
    summary = directory / "summary"
    summary.mkdir(exist_ok=True)
    write_csv(summary / "per_corruption.csv", corruption_rows)
    write_csv(summary / "summary.csv", run_rows)
    primary = {r["method"]: r["mean_error_percent"] for r in run_rows if r["batch_size"] == 200}
    write_json(summary / "audit.json", {
        "status": "completed", "official_files_verified": True,
        "official_commit": PLAN["official_commit"], "slurm_job_id": environment["slurm_job_id"],
        "runs": len(run_rows), "corruption_rows": len(corruption_rows),
        "planned_image_evaluations": len(corruption_rows) * 10000,
        "log_precision_percentage_points": 0.01,
        "primary_error_percent": primary,
        "source_to_tent_gain_pp": primary["source"] - primary["tent"],
        "norm_to_tent_gain_pp": primary["norm"] - primary["tent"],
        "summary_sha256": sha256(summary / "summary.csv"),
        "per_corruption_sha256": sha256(summary / "per_corruption.csv"),
        "measurement_note": "Errors are read from unchanged official logs (two decimal places). Process time includes Python startup, model/data loading, transfers and evaluation; no per-loop timing or GPU-memory measurement is collected.",
    })
    plot(corruption_rows, directory / "figures")
    print(f"Complete official results: {len(run_rows)} runs, {len(corruption_rows)} corruption rows")
    for row in run_rows:
        print(f"B={row['batch_size']} {row['method']}: {row['mean_error_percent']:.3f}%")


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = ("#768596", "#3b82a0", "#207864")
    for index, (method, color) in enumerate(zip(PLAN["methods"], colors)):
        lookup = {r["corruption"]: r["error_percent"] for r in rows
                  if r["batch_size"] == 200 and r["method"] == method}
        x = [n + (index - 1) * .25 for n in range(15)]
        ax.bar(x, [lookup[name] for name in PLAN["corruptions"]], width=.25,
               label=method.title(), color=color)
    ax.set_xticks(range(15))
    ax.set_xticklabels([name.replace("_", " ") for name in PLAN["corruptions"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Classification error (%)")
    ax.set_title("Official Tent example: CIFAR-10-C, severity 5, B=200")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "per_corruption.png", dpi=180)
    fig.savefig(output / "per_corruption.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
