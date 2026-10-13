"""Provenance and log utilities; no model or adaptation implementation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "experiment-plan.json").read_text())
ERROR_LINE = re.compile(r"error % \[([a-z_]+)([1-5])\]: ([0-9]+(?:\.[0-9]+)?)%")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official() -> dict:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    if provenance["commit"] != PLAN["official_commit"]:
        raise RuntimeError("Official commit differs from the experiment plan")
    for item in provenance["files"]:
        path = ROOT / "official" / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Official source missing or modified: {path}")
    return provenance


def parse_errors(log: Path) -> list[dict]:
    rows = [
        {"corruption": name, "severity": int(severity), "error_percent": float(error)}
        for name, severity, error in ERROR_LINE.findall(log.read_text(errors="replace"))
    ]
    expected = [(name, PLAN["severity"]) for name in PLAN["corruptions"]]
    if [(r["corruption"], r["severity"]) for r in rows] != expected:
        raise RuntimeError(f"Expected 15 unique severity-5 results in official order: {log}")
    if any(not 0 <= r["error_percent"] <= 100 for r in rows):
        raise RuntimeError(f"Invalid error percentage: {log}")
    return rows
