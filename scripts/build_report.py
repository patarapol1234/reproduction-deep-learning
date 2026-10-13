#!/usr/bin/env python3
"""Build a preparation, interrupted-attempt, or completed official-job report."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from common import PLAN, ROOT, sha256, verify_official, write_json


class TextContent(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self.skip += 1
        if tag in ("p", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip -= 1
        if tag in ("p", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip() + " ")


def table(headers, rows):
    return "<table><tr>" + "".join(f"<th>{escape(str(x))}</th>" for x in headers) + "</tr>" + \
        "".join("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</table>"


def interrupted_values(directory, output, references):
    """Document the observed NVML interruption without inventing measurements."""
    match = re.fullmatch(r"job-(\d+)", directory.name)
    if not directory.is_dir() or not match:
        raise RuntimeError("An interrupted attempt requires an existing job-JOB_ID directory")
    job_id = match[1]
    if (directory / "environment.json").exists() or list(directory.glob("runs/**/run.json")):
        raise RuntimeError("This interruption report supports attempts stopped before preflight only")
    log_name = f"official-v2-{job_id}"
    logs = directory.parents[1] / "logs"
    stdout = logs / f"{log_name}.out"
    stderr = logs / f"{log_name}.err"
    console = stdout.read_text()
    if f"Job: {job_id}; results:" not in console or \
            "Failed to initialize NVML: Driver/library version mismatch" not in console:
        raise RuntimeError("The log does not establish the expected preflight interruption")
    if "error % [" in console:
        raise RuntimeError("Evaluation output is present; inspect the full result set")
    preparation_file = directory.parent / "preparation.json"
    preparation = json.loads(preparation_file.read_text())
    if preparation["status"] != "prepared" or not preparation["official_files_verified"] or \
            preparation["official_commit"] != PLAN["official_commit"]:
        raise RuntimeError("Preparation evidence does not match this official experiment")
    if {Path(item["path"]).stem for item in preparation["corruption_files"]} != set(PLAN["corruptions"]):
        raise RuntimeError("Preparation evidence does not cover all 15 corruptions")
    started = re.search(r"^Started: (.+)$", console, re.MULTILINE)
    library = re.search(r"NVML library version: (.+)", console)
    observed = {
        "slurm_job_id": job_id,
        "started_at": started[1] if started else None,
        "report_status": "interrupted",
        "observed_message": "Failed to initialize NVML: Driver/library version mismatch",
        "nvml_library_version": library[1].strip() if library else None,
        "evaluation_started": False,
        "measured_runs": 0,
        "planned_runs": 3,
        "compute_node_cuda_execution": "not_checked",
        "slurm_accounting_state": "not_available_in_download",
        "preparation_host": preparation["hostname"],
        "preparation_cuda_available": preparation["cuda_available"],
        "official_commit": PLAN["official_commit"],
        "evidence_sha256": {name: sha256(path) for name, path in (
            (stdout.name, stdout), (stderr.name, stderr),
            ("preparation.json", preparation_file))},
    }
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    for path in (stdout, stderr, preparation_file):
        target = evidence / path.name
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
    write_json(output / "attempt-evidence.json", observed)
    packages = preparation["packages"]
    values = {
        "status": f"Interrupted before evaluation — Slurm job {job_id}",
        "discussion_break": 'style="page-break-before: always;"',
        "status_detail": "The downloaded job log reports an NVML driver/library mismatch. The initial GPU diagnostic stopped the submitted script before CUDA preflight or any official method ran. No new accuracy or error measurements are available.",
        "environment": f"Asset preparation was recorded on {escape(preparation['hostname'])} at {escape(preparation['captured_at_utc'])}: Python {escape(preparation['python'])}, PyTorch {escape(packages['torch'])}, torchvision {escape(packages['torchvision'])}, RobustBench {escape(packages['robustbench'])}, NumPy {escape(packages['numpy'])}, YACS {escape(packages['yacs'])}, iopath {escape(packages['iopath'])}, and PyTorch CUDA build {escape(str(preparation['cuda_build']))}. All 15 corruption arrays passed shape and dtype checks, and labels/checkpoint hashes were recorded. This is preparation-host evidence, not the allocated GPU’s environment. CUDA availability was false on that host; CUDA computation on the allocated GPU was not checked. The requested allocation was one GPU in partition gpu4090; its actual device model and compute hostname were not recorded.",
        "results_table": table(["Method", "Public reference (%)", "New v2 error (%)", "Delta (pp)"],
                               [[m.title(), f"{references[m]:.1f}", "Not measured", "Not available"] for m in PLAN["methods"]]),
        "results_discussion": "The planned primary experiment comprises three method runs and 45 per-corruption results, totaling 450,000 image evaluations across repeated corruptions of the same 10,000 source test images. This attempt produced no method logs, no per-corruption errors and no aggregate CSV. The reference values in Table 1 are published comparison targets, not results from job " + job_id + ". Neither an error reduction nor agreement with the public example can be evaluated from this attempt.",
        "attempt_section": "<h3>Observed execution evidence</h3>" + table(["Stage", "Evidence / outcome"], [
            ["Asset preparation", "15 arrays checked; official hashes verified; checkpoint present"],
            ["Job start", f"Job {job_id}, {observed['started_at']}"],
            ["GPU diagnostic", "NVML driver/library version mismatch; library " + str(observed["nvml_library_version"])],
            ["CUDA preflight", "Not reached; no job environment.json"],
            ["Source / Norm / Tent", "Not started; downloaded job directory is empty"],
            ["Mean errors / comparison", "Unavailable; no evaluation measurements"],
        ]) + "<p>The stdout file records the mismatch, while the downloaded stderr file is empty. The supplied Slurm script used <code>set -euo pipefail</code> and called <code>nvidia-smi</code> as a required command before Python preflight. The control flow and missing outputs support a stop at this diagnostic. Scheduler accounting, the compute-node name and a loaded kernel-driver version were not downloaded, so the exact driver state and final Slurm accounting status are not established.</p>",
        "recovery": "<h3>Recovery prepared for the next attempt</h3><p>The project’s Slurm wrapper now records an <code>nvidia-smi</code> failure as a warning and continues to a required PyTorch CUDA check. Preflight requires GPU availability plus an actual tensor allocation and computation on CUDA; a failure is recorded in <code>environment.json</code> and stops evaluation. This change addresses the diagnostic gate and preserves the official implementation. It does not repair the cluster driver, and its behavior on the allocated GPU remains to be verified in a new job.</p><p>If the CUDA check fails, use a working GPU node or have the cluster administrator inspect the driver/library installation before retrying. If it passes, run the three official methods, verify all 45 corruption records, and regenerate the measured-results report. The local recovery commands are documented in <code>RECOVERY-21057.md</code>.</p>",
        "conclusion": "The official-code workflow and input preparation are documented, but job " + job_id + " stopped before evaluation because its initial NVML diagnostic failed. This attempt does not demonstrate TENT’s performance or a successful numerical reproduction. A revised GPU check is ready for a new cluster run; a completed official run is required before reporting new Source, Norm and Tent errors.",
    }
    metadata = {"status": "interrupted", "results_dir": str(directory),
                "slurm_job_id": job_id, "official_commit": PLAN["official_commit"],
                "measured_runs": 0, "evidence_file": "attempt-evidence.json"}
    return values, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report")
    parser.add_argument("--interrupted", action="store_true",
                        help="Document a preflight NVML interruption using downloaded logs")
    args = parser.parse_args()
    if args.interrupted and not args.results_dir:
        parser.error("--interrupted requires --results-dir")
    verify_official()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    references = PLAN["reference_mean_error_percent"]
    values = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "status": "Prepared — new GPU results pending",
        "status_detail": "Official code, cluster workflow and analysis are prepared. No new GPU experiment has been executed, and no previous v1 measurements are presented as v2 results.",
        "environment": "Planned cluster allocation: one RTX 4090, four CPU cores and 16 GB host memory. The actual Python, package versions, CUDA build, GPU and Slurm job ID will be recorded during the new run.",
        "results_table": table(["Method", "Public reference (%)", "New v2 error (%)", "Delta (pp)"],
                               [[m.title(), f"{references[m]:.1f}", "Pending", "Pending"] for m in PLAN["methods"]]),
        "results_discussion": "The three primary runs will produce 45 corruption results. The planned protocol entails 450,000 image evaluations across repeated corruptions of the same 10,000 source test images. Agreement with the public reference and the ranking of methods remain to be measured.",
        "figure": "", "batch_section": "", "attempt_section": "", "recovery": "", "discussion_break": "",
        "conclusion": "The v2 package follows the official implementation directly and provides a reproducible execution and reporting workflow. Experimental success remains unverified until a new CUDA job completes and its logs are analyzed.",
    }
    metadata = {"status": "prepared", "results_dir": None, "official_commit": PLAN["official_commit"]}
    if args.interrupted:
        changes, metadata = interrupted_values(args.results_dir.resolve(), output, references)
        values.update(changes)
    elif args.results_dir:
        directory = args.results_dir.resolve()
        audit = json.loads((directory / "summary/audit.json").read_text())
        if audit["status"] != "completed" or audit["official_commit"] != PLAN["official_commit"]:
            raise RuntimeError("Only a completed official v2 result set can populate this report")
        if sha256(directory / "summary/summary.csv") != audit["summary_sha256"]:
            raise RuntimeError("Summary changed after aggregation; aggregate again")
        environment = json.loads((directory / "environment.json").read_text())
        with (directory / "summary/summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        primary = {r["method"]: r for r in rows if int(r["batch_size"]) == 200}
        if set(primary) != set(PLAN["methods"]):
            raise RuntimeError("Primary Source/Norm/Tent results are incomplete")
        errors = {m: float(primary[m]["mean_error_percent"]) for m in PLAN["methods"]}
        packages = environment["packages"]
        values.update({
            "status": "Completed — official v2 measurements",
            "status_detail": f"Slurm job {escape(str(environment['slurm_job_id']))}: {audit['runs']} official runs and {audit['corruption_rows']} corruption results. Log completeness, hashes and aggregate means were checked.",
            "environment": f"Recorded environment: Python {escape(environment['python'])}, PyTorch {escape(packages['torch'])}, RobustBench {escape(packages['robustbench'])}, CUDA build {escape(str(environment['cuda_build']))}, cuDNN {environment['cudnn']}, GPU {escape(str(environment['gpu']))}. The complete environment and package list accompany the results.",
            "results_table": table(["Method", "Public reference (%)", "New v2 error (%)", "Delta (pp)"],
                                   [[m.title(), f"{references[m]:.1f}", f"{errors[m]:.3f}", f"{errors[m]-references[m]:+.3f}"] for m in PLAN["methods"]]),
            "results_discussion": f"Tent’s error difference relative to Source is {errors['source']-errors['tent']:+.3f} percentage points, and relative to Norm is {errors['norm']-errors['tent']:+.3f} percentage points; positive values indicate improvement. The maximum absolute mean-error difference from the public reference is {max(abs(errors[m]-references[m]) for m in PLAN['methods']):.3f} pp. These values are measured outcomes rather than assumed matches. Process durations were " + "; ".join(f"{m.title()} {float(primary[m]['process_seconds']):.2f} s" for m in PLAN["methods"]) + ".",
            "conclusion": f"Using the official implementation, Source, Norm and Tent obtained {errors['source']:.3f}%, {errors['norm']:.3f}% and {errors['tent']:.3f}% mean error, respectively. " + ("Tent had the lowest primary mean error." if errors['tent'] < min(errors['source'], errors['norm']) else "The new run did not establish Tent as the lowest-error primary method.") + " This conclusion applies to the declared public protocol and does not establish an exact R-26 reproduction.",
        })
        figure = directory / "figures/per_corruption.png"
        if figure.exists():
            assets = output / "assets"
            assets.mkdir(exist_ok=True)
            shutil.copyfile(figure, assets / figure.name)
            values["figure"] = '<p><img src="assets/per_corruption.png" alt="Measured errors for the three methods across 15 corruptions"></p><p class="caption">Figure 1. New official-run per-corruption errors at severity 5.</p>'
        batch = {r["method"]: r for r in rows if int(r["batch_size"]) == 128}
        if batch:
            values["batch_section"] = "<h2>Optional batch-size comparison</h2>" + table(["Method", "B=200 error (%)", "B=128 error (%)"],
                [[m.title(), f"{errors[m]:.3f}", f"{float(batch[m]['mean_error_percent']):.3f}"] for m in PLAN["methods"]]) + "<p>Batch 128 changes both normalization statistics and update frequency: 79 batches per corruption instead of 50, including a final 16-image batch. A single run does not establish that either batch size is generally better.</p>"
        metadata.update(status="completed", results_dir=str(directory), slurm_job_id=environment["slurm_job_id"])
    template_file = output / "template.html"
    if not template_file.exists():
        template_file = ROOT / "report/template.html"
    template = template_file.read_text()
    report = re.sub(r"\{\{([a-z_]+)\}\}", lambda match: values[match[1]], template)
    if "{{" in report:
        raise RuntimeError("Unresolved report placeholders")
    (output / "official-tent-report.html").write_text(report)
    text = TextContent()
    text.feed(report)
    (output / "official-tent-report.txt").write_text("".join(text.parts).strip() + "\n")
    article = report.split('<article id="report-content">', 1)[1].split("</article>", 1)[0]
    content = TextContent()
    content.feed(article)
    words = len("".join(content.parts).split())
    if words > 2000:
        raise RuntimeError(f"Report exceeds 2,000 words: {words}")
    metadata.update(word_count=words, word_limit=2000, references_excluded=True)
    write_json(output / "report-status.json", metadata)
    print(f"Report: {output / 'official-tent-report.html'} ({words} words; {metadata['status']})")


if __name__ == "__main__":
    main()
