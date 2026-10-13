#!/usr/bin/env python3
"""Build analysis/official-tent-report.html from the outputs of Slurm job 21071 only.

Every number is computed from files that job produced (console.log, run.json,
environment.json, audit.json) and from the reference table in official/README.md.
No other job's results and no other report file are read.
Run:  STUDENT_ID=<your id> python3 analysis/make_report.py && python3 scripts/render_pdf.py --report-dir analysis
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import pathlib
import re
import shutil
from datetime import datetime
from html.parser import HTMLParser

V2 = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent
JOB = "21071"
STUDENT_ID = os.environ.get("STUDENT_ID", "<student id>")
METHODS = ("source", "norm", "tent")
PLAN = json.loads((V2 / "experiment-plan.json").read_text())
COR = PLAN["corruptions"]
COMMIT = PLAN["official_commit"]
ERR = re.compile(r"^\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\] .*?error % \[([a-z_]+)5\]: ([0-9.]+)%", re.M)


# ---------- load data produced by the job ----------
def run_dir(method):
    return V2 / f"results/job-{JOB}/runs/b200/{method}"


def load_run(method):
    text = (run_dir(method) / "console.log").read_text(errors="replace")
    rows = ERR.findall(text)
    assert [n for _, n, _ in rows] == COR, f"{method}: corruption order/count mismatch"
    stamps = [datetime.strptime(t, "%y/%m/%d %H:%M:%S") for t, _, _ in rows]
    first = datetime.strptime(re.search(r"^\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\]", text, re.M).group(1), "%y/%m/%d %H:%M:%S")
    secs = [(b - a).total_seconds() for a, b in zip([first] + stamps[:-1], stamps)]
    rec = json.loads((run_dir(method) / "run.json").read_text())
    assert rec["status"] == "completed" and rec["returncode"] == 0 and rec["official_commit"] == COMMIT
    assert rec["slurm_job_id"] == JOB
    assert hashlib.sha256((run_dir(method) / "console.log").read_bytes()).hexdigest() == rec["console_sha256"], f"{method}: log hash mismatch"
    return {"err": {n: float(e) for _, n, e in rows}, "secs": secs, "rec": rec, "text": text}


RUNS = {m: load_run(m) for m in METHODS}
ENV = json.loads((V2 / f"results/job-{JOB}/environment.json").read_text())
AUDIT = json.loads((V2 / f"results/job-{JOB}/summary/audit.json").read_text())
assert AUDIT["status"] == "completed" and AUDIT["corruption_rows"] == 45 and AUDIT["slurm_job_id"] == JOB
assert ENV["status"] == "prepared" and ENV["cuda_compute_check"] == "passed" and ENV["slurm_job_id"] == JOB
assert ENV["official_files_verified"] and ENV["official_commit"] == COMMIT

# reference table from the authors' README (severity 5, Standard WRN-28-10)
lines = (V2 / "official/README.md").read_text(encoding="utf-8").splitlines()
head = next(i for i, l in enumerate(lines) if "| mean | gauss_noise" in l)
cells = lambda s: [x.strip() for x in s.strip().strip("|").split("|")]
header = cells(lines[head])
ALIAS = {"gauss_noise": "gaussian_noise", "elastic_trans": "elastic_transform", "jpeg": "jpeg_compression"}
REF, REF_MEAN = {}, {}
for line in lines[head + 2:head + 5]:
    row = cells(line)
    method = row[0].split()[0].lower()
    vals = dict(zip(header[1:], row[1:]))
    REF_MEAN[method] = float(vals.pop("mean"))
    REF[method] = {ALIAS.get(k, k): float(v) for k, v in vals.items()}
assert set(REF) == set(METHODS) and all(set(REF[m]) == set(COR) for m in METHODS)

# ---------- derived numbers ----------
mean = lambda d: sum(d.values()) / len(d)
MEAN = {m: mean(RUNS[m]["err"]) for m in METHODS}
for m in METHODS:  # recomputed means must equal audit.json
    assert abs(MEAN[m] - AUDIT["primary_error_percent"][m]) < 1e-9
E = lambda m: RUNS[m]["err"]
src_norm = MEAN["source"] - MEAN["norm"]
norm_tent = MEAN["norm"] - MEAN["tent"]
src_tent = MEAN["source"] - MEAN["tent"]
share = src_norm / src_tent
gain = {c: E("norm")[c] - E("tent")[c] for c in COR}
assert all(v > 0 for v in gain.values())
by_gain = sorted(COR, key=lambda c: -gain[c])
nice = lambda c: c.replace("_", " ").replace("gaussian", "Gaussian")


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k + 1
    return r


norm_err = [E("norm")[c] for c in COR]
r_p = pearson(norm_err, [gain[c] for c in COR])
r_s = pearson(ranks(norm_err), ranks([gain[c] for c in COR]))
assert r_p > 0.8 and r_s > 0.8  # the discussion text relies on a strong association

dev = [(m, c, round(E(m)[c] - REF[m][c], 2)) for m in METHODS for c in COR]
over = [d for d in dev if abs(d[2]) > 0.05 + 1e-9]
max_dev = max(abs(d[2]) for d in dev)
mean_dev = {m: MEAN[m] - REF_MEAN[m] for m in METHODS}
max_mean_dev = max(abs(v) for v in mean_dev.values())
med = lambda v: sorted(v)[len(v) // 2]
t_med = {m: med(RUNS[m]["secs"][1:]) for m in METHODS}
proc = {m: RUNS[m]["rec"]["process_seconds"] for m in METHODS}

tm = re.search(r"PyTorch Version: torch=([^,]+), cuda=([^,]+), cudnn=(\d+)", RUNS["tent"]["text"])
torch_v, cudnn_i = tm.group(1), int(tm.group(3))
cudnn_v = f"{cudnn_i // 10000}.{(cudnn_i // 100) % 100}.{cudnn_i % 100}"
pk = ENV["packages"]
ckpt = ENV["checkpoint"]["sha256"]
gpu, node = ENV["gpu"], ENV["hostname"].split(".")[0]
adam = re.search(r"Adam \(.*?betas: \(([^)]*)\).*?lr: ([0-9.e-]+).*?weight_decay: ([0-9.]+)", RUNS["tent"]["text"], re.S)
assert adam and adam.group(2) == "0.001" and adam.group(3) == "0.0"
params = re.search(r"params for adaptation: (\[.*?\])\n", RUNS["tent"]["text"], re.S)
param_names = eval(params.group(1))
n_tensors = len(param_names)
assert n_tensors == 50 and all(n.endswith((".weight", ".bias")) for n in param_names)  # 25 BN layers x (gamma, beta)
# jobs that failed on gpu01 (facts verified from their own logs)
out57 = (V2 / "logs/official-v2-21057.out").read_text()
err66 = (V2 / "logs/official-v2-21066.err").read_text()
assert "Driver/library version mismatch" in out57 and "Error 804" in err66
assert not (V2 / "results/job-21057/runs").exists() and not (V2 / "results/job-21066/runs").exists()

f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
s2 = lambda x: f"{x:+.2f}".replace("-", "−")
s3 = lambda x: f"{x:+.3f}".replace("-", "−")


def table(headers, rows):
    head_ = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head_}</tr>{body}</table>"


def join_and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------- tables ----------
t_main = table(
    ["Method", "Published (%)", f"Job {JOB} (%)", "Δ (pp)"],
    [[m.title(), f"{REF_MEAN[m]:.1f}", f3(MEAN[m]), s3(mean_dev[m])] for m in METHODS],
)
t_cor = table(
    ["Corruption", "Source", "Norm", "Tent", "Norm − Tent"],
    [[nice(c), f2(E("source")[c]), f2(E("norm")[c]), f2(E("tent")[c]), f2(gain[c])] for c in COR]
    + [["<b>Mean</b>", f"<b>{f3(MEAN['source'])}</b>", f"<b>{f3(MEAN['norm'])}</b>", f"<b>{f3(MEAN['tent'])}</b>", f"<b>{f3(norm_tent)}</b>"]],
)
t_setup = table(
    ["Setting", "Value (taken from the run logs and environment records)"],
    [
        ["Code", f"Authors’ repository (DequanWang/tent), commit <code>{COMMIT[:7]}</code>. The <code>official/</code> directory is byte-identical to that commit (checked by <code>diff</code> against the GitHub archive); one process of <code>cifar10c.py</code> per method."],
        ["Model", f"RobustBench <code>Standard</code> WRN-28-10, checkpoint SHA-256 <code>{ckpt[:16]}…</code>"],
        ["Data", "CIFAR-10-C, 15 corruptions, severity 5, 10,000 images each, fixed order"],
        ["Protocol", "Batch size 200 (50 batches per corruption), online adaptation, model and optimizer reset before each corruption, seed 1, cuDNN benchmark on"],
        ["Tent optimizer", f"Adam, learning rate {adam.group(2)}, betas ({adam.group(1)}), weight decay 0, one step per batch (as printed by the official program)"],
        ["Run", f"Slurm job {JOB}: {gpu} (node {node})"],
        ["Software", f"Python {ENV['python']}, PyTorch {torch_v}, cuDNN {cudnn_v}, RobustBench {pk['robustbench']}"],
    ],
)

top3 = join_and([f"{nice(c)} ({f2(gain[c])} pp)" for c in by_gain[:3]])
low = sorted(COR, key=lambda c: gain[c])
low_min = gain[low[0]]
bottom = " and ".join(nice(c) for c in low if abs(gain[c] - low_min) < 0.0051)
by_method = {}
for m, c, d in over:
    by_method.setdefault(m, []).append(f"{nice(c)} ({s2(d)})")
over_txt = "; ".join(f"{m.title()}: {join_and(v)}" for m, v in by_method.items())

CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; color: #182c3e; max-width: 760px; margin: 24px auto; }
h1 { font-size: 21pt; color: #183f57; margin-bottom: 8pt; }
h2 { font-size: 13pt; color: #183f57; margin-top: 14pt; margin-bottom: 5pt; page-break-after: avoid; }
p { margin-top: 5pt; margin-bottom: 7pt; }
.meta, .caption { color: #526473; font-size: 9pt; }
.status { padding: 9pt; background-color: #eef4f7; border-left: 3pt solid #3d738d; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; margin-top: 6pt; margin-bottom: 6pt; }
td, th { padding: 2.5pt 5pt; border-bottom: 1px solid #c4d2db; text-align: left; }
th { background-color: #eaf0f4; }
code { font-family: monospace; font-size: 8.5pt; }
img { width: 100%; }
a { color: #235d7a; }
.references { font-size: 8.8pt; }
"""

ARTICLE = f"""
<h1>Reproducing the Official Tent Example on CIFAR-10-C</h1>
<p class="meta">CPE 663 Deep Learning · Research Reproduction · Student ID: {STUDENT_ID} · {datetime.now().date().isoformat()}</p>
<p class="status"><strong>Summary.</strong> We ran the authors’ unchanged code for Tent (commit {COMMIT[:7]}) on the Standard WRN-28-10 model and CIFAR-10-C at severity 5 (Slurm job {JOB}). Mean errors for Source, Norm and Tent were {f3(MEAN['source'])}%, {f3(MEAN['norm'])}% and {f3(MEAN['tent'])}%, within {max_mean_dev:.2f} pp of the authors’ published example ({REF_MEAN['source']}, {REF_MEAN['norm']}, {REF_MEAN['tent']}). Re-estimating normalization statistics (Norm) accounts for {share * 100:.1f}% of the total error reduction; Tent adds a further {f2(norm_tent)} pp and is better than Norm on all 15 corruptions.</p>

<h2>1. Problem</h2>
<p>A classifier trained on clean images loses accuracy under corruption. For the source model at severity 5, the error is {f2(E('source')['gaussian_noise'])}% on Gaussian-noise images but {f2(E('source')['brightness'])}% on brightness images. Test-time adaptation adjusts the model during deployment using only unlabeled test batches. Tent [1] is fully test-time: it needs neither the source data nor target labels.</p>

<h2>2. Method</h2>
<p>For each test batch X, Tent minimizes the mean entropy of the predictions: L(X) = −(1/|X|) Σ<sub>x∈X</sub> Σ<sub>c</sub> p(c|x) log p(c|x), where p(c|x) is the softmax probability of class c. Low entropy means confident predictions, which makes it a label-free signal. Tent changes only the BatchNorm layers. It normalizes with the current batch’s mean and variance and updates the per-channel scale γ and shift β, which are under 1% of the parameters [1], with one Adam step per batch. In this model the update covers {n_tensors} tensors (25 BatchNorm layers × γ, β), as listed in the official log.</p>
<p>Three methods are compared. <b>Source</b> uses the statistics stored at training time. <b>Norm</b> uses batch statistics and updates nothing. <b>Tent</b> adds the entropy update to Norm. In the official online protocol the prediction for a batch is made before that batch’s update, so each update influences only later batches; the model and optimizer are reset before every new corruption.</p>

<h2>3. Experimental setup</h2>
{t_setup}
<p>Two settings differ from the authors’: the software stack (their requirements pin PyTorch 1.8.1) and the severity list (their configuration evaluates severities 5 to 1; we run severity 5 only, which should not affect severity-5 results because the model is reset before every corruption–severity pair). Errors are read from the official program’s log, which prints two decimals, and means are computed from those values. Each of the three console logs contains the 15 results in the official order, and its SHA-256 matches the value recorded when the run finished.</p>
<p>Two earlier submissions (jobs 21057 and 21066) were placed on node gpu01, where <code>nvidia-smi</code> reported an NVML driver/library mismatch and PyTorch returned CUDA error 804. They stopped before any method ran and contribute no measurements; no RTX 4090 result is reported.</p>

<h2>4. Results</h2>
{t_main}
<p class="caption">Table 1. Mean error (%) over the 15 corruptions, severity 5. Published values are from the authors’ README [2]; Δ = measured − published.</p>
<p>All three means agree with the published values to one decimal (largest |Δ| = {max_mean_dev:.3f} pp), and the order Source &gt; Norm &gt; Tent holds. Relative to Source, Tent lowers the mean error by {f2(src_tent)} pp ({src_tent / MEAN['source'] * 100:.1f}% relative); relative to Norm, by {f2(norm_tent)} pp ({norm_tent / MEAN['norm'] * 100:.1f}% relative). Norm alone provides {f2(src_norm)} of the {f2(src_tent)} pp.</p>
{t_cor}
<p class="caption">Table 2. Per-corruption error (%), job {JOB}. The last column is the improvement of Tent over Norm in percentage points.</p>
<p>Tent is below Norm on all 15 corruptions. The largest improvements are {top3}; the smallest are {bottom} ({f2(low_min)} pp). Across the 15 corruptions, the improvement is strongly associated with the error that remains after Norm: Pearson r = {r_p:.2f}, Spearman ρ = {r_s:.2f}. The association is computed over only 15 points and does not show causation.</p>
<p><img src="assets/per_corruption.png" alt="Per-corruption error of Source, Norm and Tent"></p>
<p class="caption">Figure 1. Per-corruption error (%) of Source, Norm and Tent, job {JOB} ({gpu}).</p>
<p><b>Agreement with the published table.</b> Against the published per-corruption table, the largest difference is {max_dev:.2f} pp. Because the published values are rounded to 0.1, differences up to 0.05 pp are expected from rounding alone; {len(over)} of 45 values exceed it (differences in pp; {over_txt}). With a single run we cannot tell whether run-to-run variation or a difference between this setup and the one behind the published table explains them, and we did not isolate the source.</p>
<p><b>Cost.</b> From the log timestamps, the median time per corruption is {t_med['tent']:.0f} s for Tent against {t_med['source']:.0f} s for Source and {t_med['norm']:.0f} s for Norm. Whole-process times are {proc['tent']:.1f} s (Tent), {proc['norm']:.1f} s (Norm) and {proc['source']:.1f} s (Source); they include start-up and data loading, so they are not loop timings, and the official program records no GPU-memory figures.</p>

<h2>5. Discussion and limitations</h2>
<p>Most of the improvement comes from using the current batch’s statistics instead of the stored ones; the entropy update adds a smaller gain ({f2(norm_tent)} pp on average) that is consistent in sign across corruptions and larger where Norm still leaves a high error. This is consistent with the claim in [1] that entropy minimization over channel-wise affine parameters helps, and because Norm supplies most of the gain, Norm is the baseline against which Tent has to be judged.</p>
<p>The evidence has limits. It covers one benchmark, one severity, one model and the online fixed-order protocol of the public example. The results come from a single run with seed 1 on one GPU model, so no uncertainty across seeds or hardware is available. Low entropy measures confidence, not correctness, so these results do not show that Tent helps under other distribution shifts. The software stack differs from the authors’ pins, and the logs give only two decimals. The paper’s own Table 2 reports 40.8, 17.3 and 14.3% for a different model (R-26) with batch size 128 and offline adaptation [1]; it is not the target of this experiment and was not reproduced.</p>

<h2>6. Conclusion</h2>
<p>Running the authors’ unchanged code on the public WRN-28-10 example reproduced the published mean errors for Source, Norm and Tent to within {max_mean_dev:.2f} pp. Tent improved on Norm for all 15 corruptions, by {f2(norm_tent)} pp on average, but Norm alone delivered {share * 100:.1f}% of the total gain over Source.</p>
"""

REFS = """
<h2>References</h2>
<p>[1] D. Wang, E. Shelhamer, S. Liu, B. Olshausen and T. Darrell. “Tent: Fully Test-Time Adaptation by Entropy Minimization.” ICLR, 2021. <a href="https://arxiv.org/abs/2006.10726v3">arXiv:2006.10726v3</a>.</p>
<p>[2] Official Tent repository and CIFAR-10-C example, commit <code>e9e926a668d85244c66a6d5c006efbd2b82e83e8</code>. <a href="https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8">Pinned repository</a>.</p>
<p>[3] D. Hendrycks and T. Dietterich. “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.” ICLR, 2019. <a href="https://arxiv.org/abs/1807.01697">arXiv:1807.01697</a>.</p>
<p>[4] F. Croce et al. “RobustBench: A Standardized Adversarial Robustness Benchmark.” NeurIPS Datasets and Benchmarks, 2021. <a href="https://arxiv.org/abs/2010.09670">arXiv:2010.09670</a>.</p>
"""

PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Reproducing the Official Tent Example on CIFAR-10-C</title>
<style>{CSS}</style></head><body>
<article id="report-content">{ARTICLE}</article>
<div class="references">{REFS}</div></body></html>
"""


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self.skip += 1
        if tag in ("p", "h1", "h2", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip -= 1
        if tag in ("p", "h1", "h2", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip() + " ")


def words(fragment):
    t = Text()
    t.feed(fragment)
    return len("".join(t.parts).split())


n_words = words(ARTICLE)
assert n_words <= 2000, f"report has {n_words} words (limit 2,000)"
assert "21072" not in PAGE, "report must not mention job 21072"

(OUT / "assets").mkdir(exist_ok=True)
shutil.copyfile(V2 / f"results/job-{JOB}/figures/per_corruption.png", OUT / "assets/per_corruption.png")
(OUT / "official-tent-report.html").write_text(PAGE, encoding="utf-8")
t = Text()
t.feed(PAGE)
(OUT / "official-tent-report.txt").write_text("".join(t.parts).strip() + "\n", encoding="utf-8")
(OUT / "report-status.json").write_text(
    json.dumps(
        {
            "status": "completed",
            "results_dir": str(V2 / f"results/job-{JOB}"),
            "slurm_job_id": JOB,
            "official_commit": COMMIT,
            "word_count": n_words,
            "word_limit": 2000,
            "references_excluded": True,
        },
        indent=2,
    )
    + "\n"
)
print(f"wrote {OUT / 'official-tent-report.html'} ({n_words} words)")
print(f"corr: pearson {r_p:.3f} spearman {r_s:.3f}; over-0.05: {len(over)} -> {over_txt}; max dev {max_dev:.2f}")
