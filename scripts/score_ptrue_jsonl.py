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
    parser.add_argument("--judge-model", default=None,
                        help="Judge model. P(True) is a SELF-evaluation: this "
                             "must equal the model that generated the records "
                             "(sep_v3 protocol, single greedy self-eval). "
                             "Defaults to the records' own model.")
    parser.add_argument("--judge-revision", default=None,
                        help="Exact Hugging Face revision for the judge model.")
    parser.add_argument("--allow-external-judge", action="store_true",
                        help="Escape hatch for legacy/archived runs only: "
                             "permit a judge that differs from the generating "
                             "model. NEVER valid for sep_v3 in-family scoring.")
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

    # Self-eval enforcement (fail-closed, BEFORE any judge is constructed).
    # Kadavath-style P(True) reads the generating model's own probability
    # that its answer is true; scoring one model's answers with another
    # model's judge is a different (external-judge) baseline and must never
    # be presented as P(True).
    record_models = {str(record.get("model", "")).strip() for record in records}
    if len(record_models) != 1 or "" in record_models:
        print(
            "FAIL-CLOSED: input records must carry exactly one non-empty "
            f"'model'; found {sorted(record_models)!r}",
            file=sys.stderr,
        )
        return 2
    (cell_model,) = record_models
    if args.judge_model is None:
        args.judge_model = cell_model
    if args.judge_model != cell_model and not args.allow_external_judge:
        print(
            "FAIL-CLOSED: P(True) is a self-evaluation; judge model "
            f"{args.judge_model!r} does not match the records' generating "
            f"model {cell_model!r}. Pass --allow-external-judge only for "
            "legacy runs that must not be labeled ptrue in-family.",
            file=sys.stderr,
        )
        return 2

    few_shot = _few_shot_prefix(args.few_shot_data, args.n_few_shot)

    if args.stub_judge:
        judge = _stub_judge()
    else:  # pragma: no cover - requires GPU/vLLM
        judge = _make_vllm_true_false_judge(args.judge_model, args.judge_revision)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for record in records:
            score = score_record_ptrue(record, judge, few_shot_prefix=few_shot)
            out = dict(record)
            out_scores = dict(out.get("scores", {}))
            out_scores["ptrue_uncertainty"] = score
            out["scores"] = out_scores
            out["ptrue_provenance"] = {
                "protocol": "kadavath_2022_single_greedy_self_eval",
                "judge_model": args.judge_model,
                "judge_revision": args.judge_revision,
                "self_eval": args.judge_model == cell_model,
                "n_few_shot": args.n_few_shot if args.few_shot_data else 0,
                "stub_judge": bool(args.stub_judge),
            }
            if args.run_id:
                out["run_id"] = args.run_id
            fh.write(json.dumps(out, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} records to {args.output_jsonl}")
    return 0


def _make_vllm_true_false_judge(
    judge_model: str, judge_revision: str | None = None
) -> JudgeFn:  # pragma: no cover
    """vLLM judge: P("True") from the renormalized True/False next-token mass.

    Generates a single token with logprobs and reads the probability mass on
    the "True"/"False" continuations, renormalizing over the two options.
    """
    import math
    import os

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=judge_model,
        revision=judge_revision,
        tensor_parallel_size=int(os.environ.get("SE_TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.environ.get("SE_GPU_MEMORY_UTILIZATION", "0.90")),
        dtype=os.environ.get("SE_VLLM_DTYPE", "auto"),
        enforce_eager=True,
    )
    # Their get_p_true appends " A" to the prompt and returns the *raw* log
    # probability of that token:
    #
    #     input_data += ' A'
    #     target_ids_true[0, :-1] = -100
    #     model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)
    #     return -loss_true.item()
    #
    # and compute_uncertainty_measures then reports p_false_fixed = 1 - exp(p).
    # There is no renormalisation against "B". Renormalising is not a monotone
    # transform of log p(A) unless p(B) is constant, so it ranks prompts
    # differently -- which changes P(True)'s AUROC.
    #
    # prompt_logprobs is used rather than the top-k of a generated token because
    # "A" need not appear in the top-k, and silently scoring it as absent would
    # bias exactly the confident-wrong cases this baseline exists to catch.
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def judge(prompt: str) -> float:
        output = llm.generate([prompt + " A"], params)[0]
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if not prompt_logprobs:
            raise RuntimeError(
                "vLLM returned no prompt_logprobs; cannot reproduce their "
                "raw log p(' A') (fail-closed)"
            )
        final = prompt_logprobs[-1]
        if not final:
            raise RuntimeError("no logprob for the appended ' A' token")
        # prompt_logprobs[i] maps token id -> Logprob for position i; with
        # prompt_logprobs=0 the only entry is the token actually present.
        entry = next(iter(final.values()))
        logprob = float(getattr(entry, "logprob", entry))
        if not math.isfinite(logprob):
            raise RuntimeError("non-finite log p(' A')")
        # Returned as a probability so downstream uncertainty stays 1 - P(True),
        # matching their p_false_fixed.
        return math.exp(logprob)

    return judge


if __name__ == "__main__":
    raise SystemExit(main())
