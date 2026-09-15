#!/usr/bin/env python3
"""Export saved biography records or compute verified records-only paired CIs.

The export mode is read-only and uses only the standard library. Run it through
SSH on stdin to avoid editing the shared cluster checkout. Analysis imports the
unchanged short-answer reference implementation from the local repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

METHODS = ("discrete_semantic_entropy", "surface_entropy", "naive_sample_entropy")
MODELS = {
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
}
BACKENDS = (
    "exact-match", "posthoc-nli-deberta-v3-base", "posthoc-nli-deberta-v3-large",
    "posthoc-llm-judge-qwen2.5-72b-instruct",
)
JUDGE = "Qwen/Qwen2.5-72B-Instruct"
RECORD_FIELDS = (
    "prompt_id", "prompt", "model", "entailment_backend", "scores",
    "correctness_label", "correctness_label_provenance",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_records(path):
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.splitlines() if line.strip()], raw


def export_snapshot(root):
    """Project existing labeled files, checking scores against untouched inputs."""
    root = root.resolve()
    generations = sorted(root.glob("phase1/phase1-alex-bio-v3-*"))
    grades = sorted(root.glob("phase1/*-bio-grade-v3-*"))
    posthoc = sorted(p for p in (root / "phase2").iterdir() if p.is_dir())
    require((len(generations), len(grades), len(posthoc)) == (9, 9, 27),
            "expected 9 generation, 9 grading, and 27 posthoc artifacts")
    artifacts = []
    for directory in generations + grades + posthoc:
        is_grade = directory in grades
        path = directory / ("scored.jsonl" if is_grade else "scored.labeled.jsonl")
        records, raw = read_records(path)
        original_hash = None
        if not is_grade:
            original, original_raw = read_records(directory / "scored.jsonl")
            require(len(original) == len(records), f"changed record count: {path}")
            for saved, labeled in zip(original, records):
                for field in ("prompt_id", "prompt", "model", "entailment_backend"):
                    require(saved[field] == labeled[field], f"changed {field}: {path}")
                for method in METHODS:
                    require(saved["scores"][method] == labeled["scores"][method],
                            f"changed score {method}: {path}")
            original_hash = hashlib.sha256(original_raw).hexdigest()
        meta_raw = (directory / "job_meta.json").read_bytes()
        artifacts.append(dict(
            path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            original_scored_sha256=original_hash,
            metadata_sha256=hashlib.sha256(meta_raw).hexdigest(),
            metadata=json.loads(meta_raw), is_grade=is_grade,
            records=[{key: r[key] for key in RECORD_FIELDS} for r in records],
        ))
    return dict(root=str(root), cluster_commit=subprocess.check_output(
        ["git", "-C", str(root.parent), "rev-parse", "HEAD"], text=True).strip(),
        artifacts=artifacts)


def indexed(records, path):
    require(len(records) == 500, f"expected 500 records: {path}")
    by_id = {r["prompt_id"]: r for r in records}
    require(len(by_id) == 500, f"duplicate prompt_id: {path}")
    for record in records:
        require(type(record["correctness_label"]) is bool, f"nonboolean label: {path}")
        require(record["correctness_label_provenance"]["judge_model"] == JUDGE,
                f"unexpected correctness judge: {path}")
    return by_id


def validate_snapshot(snapshot, measurements, reference, np):
    """Fail before bootstrap on incomplete, mismatched, or divergent records."""
    artifacts = snapshot["artifacts"]
    require(len(artifacts) == 45, "expected 45 artifacts")
    grades, sources = {}, {}
    for artifact in artifacts:
        path, meta = Path(artifact["path"]), artifact["metadata"]
        if artifact["is_grade"]:
            source = meta["source_run_id"]
            require(source not in grades, f"duplicate grading source: {source}")
            require(meta["judge_model"] == JUDGE, f"unexpected grader: {path}")
            grades[source] = indexed(artifact["records"], path)
        elif path.parent.name.startswith("phase1-alex-bio-v3-"):
            require(path.parent.name not in sources, f"duplicate generation: {path}")
            sources[path.parent.name] = artifact
    require(len(grades) == len(sources) == 9, "expected nine generation/grading runs")
    require(set(grades) == set(sources), "grading and generation sources differ")
    cells = {}
    for artifact in artifacts:
        if artifact["is_grade"]:
            continue
        path, meta = Path(artifact["path"]), artifact["metadata"]
        source = meta.get("source_run_id", path.parent.name)
        require(source in sources, f"unknown generation source: {path}")
        generation = sources[source]
        model = MODELS[generation["metadata"]["model_name"]]
        seed = int(generation["metadata"]["seed"])
        records = indexed(artifact["records"], path)
        baseline = indexed(generation["records"], generation["path"])
        require(set(records) == set(baseline) == set(grades[source]),
                f"prompt_id sets differ: {path}")
        raw_backends = {r["entailment_backend"] for r in records.values()}
        require(len(raw_backends) == 1, f"mixed backends: {path}")
        backend = raw_backends.pop()
        backend = backend if backend == "exact-match" else "posthoc-" + backend
        require(backend in BACKENDS, f"unexpected backend: {backend}")
        for pid, record in records.items():
            grade, base = grades[source][pid], baseline[pid]
            for field in ("prompt", "model", "correctness_label", "correctness_label_provenance"):
                require(record[field] == grade[field] == base[field],
                        f"{field} differs for {pid}: {path}")
            require(record["model"] == generation["metadata"]["model_name"],
                    f"record model differs from metadata: {path}")
            for method in METHODS[1:]:
                require(record["scores"][method] == base["scores"][method],
                        f"backend-dependent {method}: {path}")
        # Canonical prompt order makes pairing independent of file ordering.
        ordered = [records[pid] for pid in sorted(records)]
        scores = {m: np.array([r["scores"][m] for r in ordered], float) for m in METHODS}
        require(all(np.isfinite(v).all() for v in scores.values()), f"nonfinite scores: {path}")
        y = np.array([not r["correctness_label"] for r in ordered], bool)
        require(0 < y.sum() < len(y), f"single-class run: {path}")
        key = (model, seed, backend)
        require(key not in cells, f"duplicate cell: {key}")
        cells[key] = (scores, y)
    expected = {(m, s, b) for m in MODELS.values() for s in (42, 43, 44) for b in BACKENDS}
    require(set(cells) == expected, "missing or unexpected generator-seed-backend cells")
    relevant = [r for r in measurements if r["arm"] == "long_form" and r["method"] in METHODS]
    require(len(relevant) == 108, "expected 108 reference measurements")
    checked = set()
    max_error = 0.0
    for row in relevant:
        key = (row["model"], int(row["seed"]), row["entailment_backend"])
        method = row["method"]
        require((key, method) not in checked, f"duplicate reference measurement: {key}, {method}")
        checked.add((key, method))
        scores, y = cells[key]
        value = reference.auroc(scores[method], y)
        require(round(value, 6) == float(row["auroc"]),
                f"measurement mismatch: {key}, {method}: {value} != {row['auroc']}")
        require(int(row["n_records"]) == len(y), f"record count mismatch: {key}")
        require(abs(float(row["accuracy"]) - float((~y).mean())) < 1e-12,
                f"accuracy mismatch: {key}")
        max_error = max(max_error, abs(value - float(row["auroc"])))
    return cells, max_error


def compute_rows(cells, reference, np, resamples=2000, seed=20260831):
    rng = np.random.default_rng(seed)
    rows = []
    for backend in BACKENDS:
        for right in METHODS[1:]:
            for model in sorted(MODELS.values()) + ["all"]:
                units = {}
                for (generator, run_seed, b), (scores, y) in sorted(cells.items()):
                    if b == backend and model in (generator, "all"):
                        units.setdefault(("bio", generator), []).append(
                            (scores[METHODS[0]], scores[right], y))
                point = reference.statistic(units, sorted(units), False, None)
                lo, hi, p = reference.bootstrap(units, resamples, "records", rng)
                rows.append(dict(
                    arm="long_form", condition="paragraph", model=model,
                    entailment_backend=backend, grader="llm_qwen2.5-72b",
                    comparison=f"{METHODS[0]}_minus_{right}",
                    left=METHODS[0], right=right, scheme="records-only",
                    n_units=len(units), n_cells=sum(map(len, units.values())),
                    n_records=sum(c[2].size for cs in units.values() for c in cs),
                    delta=round(point, 6), records_lo=round(lo, 6),
                    records_hi=round(hi, 6), records_p=round(p, 6),
                    resamples=resamples, seed=seed,
                ))
                print(f"{model} {backend} {right}: {point:+.6f} [{lo:+.6f}, {hi:+.6f}]",
                      file=sys.stderr, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, help="read-only snapshot to stdout")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.export_root:
        json.dump(export_snapshot(args.export_root), sys.stdout, separators=(",", ":"))
        return
    if not args.snapshot or not args.measurements or (not args.out and not args.validate_only):
        parser.error("analysis requires --snapshot, --measurements, and --out or --validate-only")
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paired_bootstrap_hierarchical as reference

    snapshot = json.loads(args.snapshot.read_text())
    with args.measurements.open(newline="") as handle:
        measurements = list(csv.DictReader(handle))
    cells, error = validate_snapshot(snapshot, measurements, reference, np)
    print(f"Validated 108 AUROCs exactly at six decimals; max unrounded error={error:.3g}",
          file=sys.stderr)
    if args.validate_only:
        return
    rows = compute_rows(cells, reference, np)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
