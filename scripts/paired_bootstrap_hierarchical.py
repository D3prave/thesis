#!/usr/bin/env python3
"""Hierarchical paired bootstrap for two uncertainty scores.

Answers the question a cell-level test cannot: whether a small, consistent
AUROC difference between two methods survives once record-level information is
used, and whether it survives once the nesting of seeds inside
(dataset, model) units is respected.

Three resampling schemes are reported side by side for every slice:

``units``
    Resample the 9 (dataset, model) units with replacement, holding the records
    within each cell fixed. This is the cell-level test: it uses only the
    between-unit variance and is the most conservative of the three.

``records``
    Hold the units fixed and resample records with replacement inside each cell,
    paired so both methods always score the same prompts. This uses only the
    within-cell variance and is the most permissive.

``hier``
    Resample units with replacement, then resample records inside each drawn
    cell. This is the honest interval: it carries both variance components.

Read them together. When ``records`` excludes zero and ``hier`` does not, the
apparent effect is an artifact of treating correlated records as independent
evidence about a population of tasks.

Usage:
    uv run python scripts/paired_bootstrap_hierarchical.py --list
    uv run python scripts/paired_bootstrap_hierarchical.py \
        --left discrete_semantic_entropy --right surface_entropy \
        --out results/paired_bootstrap_se_vs_surface.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CELL = re.compile(r"repl-(chat|default)-(train|eval)-(\w+)-(.+)-s(4[234])-\d+$")

#: condition -> (regime, candidate ladder roots, in preference order)
#: A cell directory name does not record its condition -- that is carried by
#: which ladder tree the cell lives under -- so the roots must be named here.
#: Relabelling trees are matched to each condition automatically, by exact
#: overlap of cell directory names, which does not depend on their naming.
CONDITIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "chat_0shot": ("chat", ("results/ladder",)),
    "default_5shot": ("default", ("results/ladder",)),
    "chat_5shot": (
        "chat",
        ("results/ladder_factorial_chat5", "results/ladder_chat5", "results/ladder_factorial"),
    ),
    "default_0shot": (
        "default",
        ("results/ladder_factorial_default0", "results/ladder_default0"),
    ),
}


def pick_ladder(candidates, regime: str) -> Path | None:
    """First candidate tree that actually holds eval cells for this regime."""
    for c in candidates:
        p = Path(c)
        if p.is_dir() and any(p.glob(f"*/repl-{regime}-eval-*/scored.jsonl")):
            return p
    return None


def discover_relabel_roots(cell_names: set[str]) -> dict[str, tuple[str, int]]:
    """Bind relabelling trees to one condition by exact cell-name overlap.

    Cell directory names carry the collection run's suffix, so the tree that
    relabelled a given condition is the one whose directory names actually
    match that condition's cells. A tree relabelling a different condition
    overlaps in zero names even though the regime prefix is identical.

    Returns grader -> (root, number of matching cells).
    """
    scored: list[tuple[int, Path]] = []
    for p in sorted(Path("results").glob("relabel*")):
        if not p.is_dir():
            continue
        names = {
            d.name
            for d in p.iterdir()
            if d.is_dir() and (d / "scored_llm.jsonl").exists()
        }
        overlap = len(names & cell_names)
        if overlap:
            scored.append((overlap, p))

    out: dict[str, tuple[str, int]] = {}
    for overlap, p in sorted(scored, key=lambda t: -t[0]):
        grader = (
            "llm_llama-3.1-70b"
            if "llama70b" in p.name or "llama-70b" in p.name
            else "llm_qwen2.5-72b"
        )
        out.setdefault(grader, (str(p), overlap))
    return out


def key(name: str):
    """Return (regime, role, dataset, model, seed) for a cell directory."""
    m = CELL.match(name)
    return m.groups() if m else None


def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Rank-based AUROC. ``y`` is True where the answer is incorrect."""
    s = np.asarray(scores, float)
    y = np.asarray(y, bool)
    pos = int(y.sum())
    neg = y.size - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(s.size, float)
    i = 0
    while i < ss.size:
        j = i
        while j + 1 < ss.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def load_grader(root: str) -> dict:
    """prompt_id -> correctness_label, per cell, from a relabelling tree."""
    out: dict = {}
    for f in glob.glob(f"{root}/*/scored_llm.jsonl"):
        k = key(Path(f).parent.name)
        if not k:
            continue
        with open(f, encoding="utf-8") as fh:
            out[k] = {
                json.loads(line)["prompt_id"]: json.loads(line)["correctness_label"]
                for line in fh
                if line.strip()
            }
    return out


def load_cell(
    path: Path, left: str, right: str, grader: str, labels: dict, excl: set, cell_key
):
    """Return (left scores, right scores, incorrect flags) for one cell."""
    ls, rs, ys = [], [], []
    label_map = labels.get(cell_key) if labels else None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["prompt_id"]
            if pid in excl:
                continue
            scores = r.get("scores", {})
            if left not in scores or right not in scores:
                continue
            if grader == "squad_token_f1":
                # Fail closed. An earlier version fell back to
                # `correctness_label` when the squad label was absent, which
                # silently graded those records with whatever label the cell
                # happened to carry -- usually an LLM judge's. That corrupted
                # token-F1 slices only, and by enough to move a slice mean by
                # 0.01 AUROC. If the squad label is missing the cell is not a
                # token-F1 cell and must not be scored as one.
                if "correctness_label_squad" not in r:
                    raise KeyError(
                        f"{path}: record {pid!r} has no "
                        f"'correctness_label_squad'; refusing to substitute "
                        f"'correctness_label' for grader 'squad_token_f1'"
                    )
                correct = bool(r["correctness_label_squad"])
            else:
                if label_map is None or pid not in label_map:
                    continue
                correct = bool(label_map[pid])
            ls.append(float(scores[left]))
            rs.append(float(scores[right]))
            ys.append(not correct)
    if not ys:
        return None
    return np.array(ls), np.array(rs), np.array(ys, bool)


def cell_delta(cell, rng=None) -> float | None:
    """Paired AUROC difference for one cell, optionally on resampled records."""
    ls, rs, ys = cell
    if rng is not None:
        idx = rng.integers(0, ys.size, size=ys.size)
        ls, rs, ys = ls[idx], rs[idx], ys[idx]
    a = auroc(ls, ys)
    b = auroc(rs, ys)
    if a is None or b is None:
        return None
    return a - b


def statistic(units, unit_order, resample_records, rng) -> float | None:
    """Mean over units of the mean-over-seeds paired AUROC difference."""
    per_unit = []
    for u in unit_order:
        deltas = [
            d
            for cell in units[u]
            if (d := cell_delta(cell, rng if resample_records else None)) is not None
        ]
        if deltas:
            per_unit.append(float(np.mean(deltas)))
    if not per_unit:
        return None
    return float(np.mean(per_unit))


def bootstrap(units, n_boot, mode, rng):
    """Percentile CI for the mean paired difference under one scheme."""
    keys = sorted(units)
    n = len(keys)
    draws = []
    for _ in range(n_boot):
        if mode == "records":
            order = keys
        else:
            order = [keys[i] for i in rng.integers(0, n, size=n)]
        s = statistic(units, order, mode in ("records", "hier"), rng)
        if s is not None:
            draws.append(s)
    if not draws:
        return None, None, None
    d = np.array(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float(lo), float(hi), min(1.0, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="discrete_semantic_entropy")
    ap.add_argument("--right", default="surface_entropy")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--dedup", default="results/dedup_triviaqa.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the discovered result trees and cell counts, then exit.",
    )
    args = ap.parse_args(argv)

    if args.list:
        print("result trees under results/:")
        for p in sorted(Path("results").iterdir()):
            if not p.is_dir():
                continue
            n_cells = len(list(p.glob("*/repl-*/scored.jsonl")))
            n_flat = len(list(p.glob("repl-*/scored.jsonl")))
            n_llm = len(list(p.glob("*/scored_llm.jsonl")))
            print(
                f"  {p.name:44s} backend-nested={n_cells:5d} "
                f"flat={n_flat:5d} scored_llm={n_llm:5d}"
            )
            if n_cells:
                for b in sorted(x.name for x in p.iterdir() if x.is_dir())[:6]:
                    print(f"      backend: {b}")
        return 0

    excl: set = set()
    dedup = Path(args.dedup)
    if dedup.exists():
        excl = set(json.loads(dedup.read_text())["excluded_eval_prompts"])
    else:
        print(f"warning: {dedup} not found; running without deduplication", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    rows = []

    print("resolved trees:")
    for condition, (regime, candidates) in CONDITIONS.items():
        ladder_path = pick_ladder(candidates, regime)
        if ladder_path is None:
            print(f"  {condition:15s} no ladder tree found in {list(candidates)}")
            continue
        cell_names = {
            f.parent.name for f in ladder_path.glob(f"*/repl-{regime}-eval-*/scored.jsonl")
        }
        found = discover_relabel_roots(cell_names)
        print(f"  {condition:15s} ladder={ladder_path}  cells={len(cell_names)}")
        for g, (root, overlap) in sorted(found.items()):
            print(f"  {'':15s}   {g:18s} <- {root} ({overlap} cells)")
        if not found:
            print(f"  {'':15s}   no relabelling tree matched; token-F1 only")
    print()

    for condition, (regime, candidates) in CONDITIONS.items():
        ladder_path = pick_ladder(candidates, regime)
        if ladder_path is None:
            continue
        cell_names = {
            f.parent.name for f in ladder_path.glob(f"*/repl-{regime}-eval-*/scored.jsonl")
        }
        graders = {"squad_token_f1": {}}
        for g, (root, _overlap) in discover_relabel_roots(cell_names).items():
            graders[g] = load_grader(root)

        for bdir in sorted(p for p in ladder_path.iterdir() if p.is_dir()):
            for grader, labels in graders.items():
                units = defaultdict(list)
                for f in sorted(bdir.glob(f"repl-{regime}-eval-*/scored.jsonl")):
                    k = key(f.parent.name)
                    if not k:
                        continue
                    _, _, dataset, model, _seed = k
                    cell = load_cell(f, args.left, args.right, grader, labels, excl, k)
                    if cell is not None:
                        units[(dataset, model)].append(cell)
                if len(units) < 2:
                    continue

                point = statistic(units, sorted(units), False, rng)
                res = {
                    m: bootstrap(units, args.resamples, m, rng)
                    for m in ("units", "records", "hier")
                }
                n_records = sum(c[2].size for cs in units.values() for c in cs)
                row = dict(
                    condition=condition,
                    entailment_backend=bdir.name,
                    grader=grader,
                    n_units=len(units),
                    n_cells=sum(len(v) for v in units.values()),
                    n_records=n_records,
                    delta=round(point, 6) if point is not None else "",
                )
                for m, (lo, hi, p) in res.items():
                    row[f"{m}_lo"] = round(lo, 6) if lo is not None else ""
                    row[f"{m}_hi"] = round(hi, 6) if hi is not None else ""
                    row[f"{m}_p"] = round(p, 6) if p is not None else ""
                rows.append(row)
                print(
                    f"{condition:15s} {bdir.name:36s} {grader:18s} "
                    f"d={row['delta']:+.4f}  "
                    f"units[{row['units_lo']:+.4f},{row['units_hi']:+.4f}] "
                    f"rec[{row['records_lo']:+.4f},{row['records_hi']:+.4f}] "
                    f"hier[{row['hier_lo']:+.4f},{row['hier_hi']:+.4f}]",
                    flush=True,
                )

    if not rows:
        print("no slices found; run with --list to check the tree layout", file=sys.stderr)
        return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} slices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
