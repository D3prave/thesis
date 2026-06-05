#!/usr/bin/env python3
"""Score a source JSONL with the P(True) supervised baseline.

Reads an exact-match source ``scored.jsonl`` (the same-source sample), builds a
P(True) prompt per record from the question, the brainstormed sampled answers,
and the most-likely answer, asks a judge model for ``P("True")``, and writes a
new ``scored.jsonl`` with ``scores["ptrue_uncertainty"] = 1 - P(True)``.

The judge is a vLLM model that reports the next-token probabilities of the
"True"/"False" continuations; ``--stub-judge`` substitutes a deterministic judge
so the wiring is testable without a GPU. Few-shot examples (paper: up to 20,
with ground-truth labels) are read from an optional labelled JSONL.

Usage::

    # Local smoke test:
    python scripts/score_ptrue_jsonl.py source.jsonl out.jsonl --stub-judge

    # Cluster (vLLM judge):
    python scripts/score_ptrue_jsonl.py source.jsonl out.jsonl \
        --judge-model meta-llama/Llama-3.1-8B-Instruct \
        --few-shot-data data/processed/triviaqa_val_500.jsonl --n-few-shot 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from semantic_entropy.ptrue import (
        JudgeFn,
        build_few_shot_prefix,
        most_likely_answer,
        score_record_ptrue,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from semantic_entropy.ptrue import (
        JudgeFn,
        build_few_shot_prefix,
        most_likely_answer,
        score_record_ptrue,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _stub_judge() -> JudgeFn:
    """Deterministic judge: confident-True when the proposed answer is the
    clear modal answer, less confident otherwise. For tests only."""
    def judge(prompt: str) -> float:
        # A trivially deterministic function of the prompt length parity so the
        # output is reproducible; real judges read True/False token logprobs.
        return 0.75 if ("Proposed answer:" in prompt and len(prompt) % 2 == 0) else 0.4
    return judge


def _few_shot_prefix(path: Path | None, n: int) -> str:
    if path is None or n <= 0:
        return ""
    examples = []
    for record in load_jsonl(path)[: n * 3]:  # oversample; some may lack labels
        if "correctness_label" not in record or not record.get("sampled_answers"):
            continue
        examples.append({
            "question": record.get("prompt", ""),
            "brainstormed_answers": record.get("sampled_answers", []),
            "proposed_answer": most_likely_answer(record),
            "is_correct": bool(record["correctness_label"]),
        })
        if len(examples) >= n:
            break
    return build_few_shot_prefix(examples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--judge-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--few-shot-data", type=Path, default=None)
    parser.add_argument("--n-few-shot", type=int, default=20)
    parser.add_argument("--stub-judge", action="store_true",
                        help="Use a deterministic judge (tests; no GPU).")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    records = load_jsonl(args.input_jsonl)
    if not records:
        print("No records to score.", file=sys.stderr)
        return 1

    few_shot = _few_shot_prefix(args.few_shot_data, args.n_few_shot)

    if args.stub_judge:
        judge = _stub_judge()
    else:  # pragma: no cover - requires GPU/vLLM
        judge = _make_vllm_true_false_judge(args.judge_model)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for record in records:
            score = score_record_ptrue(record, judge, few_shot_prefix=few_shot)
            out = dict(record)
            out_scores = dict(out.get("scores", {}))
            out_scores["ptrue_uncertainty"] = score
            out["scores"] = out_scores
            if args.run_id:
                out["run_id"] = args.run_id
            fh.write(json.dumps(out, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} records to {args.output_jsonl}")
    return 0


def _make_vllm_true_false_judge(judge_model: str) -> JudgeFn:  # pragma: no cover
    """vLLM judge: P("True") from the renormalized True/False next-token mass.

    Generates a single token with logprobs and reads the probability mass on
    the "True"/"False" continuations, renormalizing over the two options.
    """
    import math
    import os

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=judge_model,
        tensor_parallel_size=int(os.environ.get("SE_TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.environ.get("SE_GPU_MEMORY_UTILIZATION", "0.90")),
        dtype=os.environ.get("SE_VLLM_DTYPE", "auto"),
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def judge(prompt: str) -> float:
        # Authors' format: P(True) = renormalized mass on the "A" continuation
        # ("A) True") versus "B" ("B) False").
        out = llm.generate([prompt], params)[0].outputs[0]
        logprobs = out.logprobs[0] if out.logprobs else {}
        a_lp, b_lp = -math.inf, -math.inf
        for entry in logprobs.values():
            token = entry.decoded_token.strip().lower()
            if token == "a" or token.startswith("true"):
                a_lp = max(a_lp, entry.logprob)
            elif token == "b" or token.startswith("false"):
                b_lp = max(b_lp, entry.logprob)
        if a_lp == -math.inf and b_lp == -math.inf:
            return 0.5
        a, b = math.exp(a_lp), math.exp(b_lp)
        return a / (a + b) if (a + b) > 0 else 0.5

    return judge


if __name__ == "__main__":
    raise SystemExit(main())
