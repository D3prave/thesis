#!/usr/bin/env python3
"""Recompute correctness labels with the LLM judge from their released code.

Their `--metric` accepts {squad, llm, llm_gpt-3.5, llm_gpt-4}, so an LLM judge
is a supported path in their implementation rather than something invented here.
`utils.model_based_metric` supplies the prompt, reproduced verbatim below.

Why this is needed: token-F1 > 0.5 against a short reference fails whenever a
correct answer carries commentary, and Llama-3.1 and Mistral-v0.3 append
scaffolding their 2023 stop list never anticipated -- "## Step 1: ...",
"(Note: ...)", "(Source: ...)". Measured on the sentence-length eval cells,
records whose reference appears verbatim in the answer yet are labelled
incorrect: TriviaQA 51.6%, NQ-Open 40.1%, SVAMP 7.7%, and the error runs almost
entirely in one direction.

`llm` reuses the generator, which would have each model grade its own answers.
A single independent judge is used instead and recorded in the output, the same
substitution already made for the entailment rung.

The original label is preserved as `correctness_label_squad` so the two can be
compared, which is itself a reportable result.


Usage:
    python scripts/relabel_with_llm_judge.py IN.jsonl OUT.jsonl \
        --judge-model Qwen/Qwen2.5-72B-Instruct --judge-revision <40hex>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Verbatim from utils.model_based_metric.
SINGLE = (
    "We are assessing the quality of answers to the following question: {question}\n"
    "The expected answer is: {answer}.\n"
    "The proposed answer is: {proposed}\n"
    "Within the context of the question, does the proposed answer mean the same "
    "as the expected answer? Respond only with yes or no.\nResponse:"
)
MULTI = (
    "We are assessing the quality of answers to the following question: {question}\n"
    "The following are expected answers to this question: {answers}.\n"
    "The proposed answer is: {proposed}\n"
    "Within the context of the question, does the proposed answer mean the same "
    "as any of the expected answers? Respond only with yes or no.\nResponse:"
)


def build_prompt(question: str, references: list, proposed: str) -> str:
    # Their branch is on len(correct_answers) == 1, and the multi-answer form
    # interpolates the Python list repr, which is reproduced here.
    if len(references) == 1:
        return SINGLE.format(
            question=question, answer=references[0], proposed=proposed
        )
    return MULTI.format(question=question, answers=references, proposed=proposed)


def parse(text: str) -> float | None:
    lowered = text.lower()
    if "yes" in lowered:
        return 1.0
    if "no" in lowered:
        return 0.0
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-revision", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args()

    records = []
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if args.limit and index >= args.limit:
                break
            if line.strip():
                records.append(json.loads(line))
    print(f"{len(records)} records from {args.input_jsonl}")

    prompts = []
    for record in records:
        references = [
            str(r) for r in (record.get("reference_answers") or []) if str(r).strip()
        ]
        prompts.append(
            build_prompt(
                record["prompt"], references, record.get("most_likely_answer") or ""
            )
        )

    if args.print_prompt:
        print("\n--- first prompt ---")
        print(prompts[0])
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tp = args.tensor_parallel_size
    if tp <= 0:
        try:
            import torch

            tp = torch.cuda.device_count() or 1
        except Exception:
            tp = 1
    print(f"tensor_parallel_size={tp}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model, revision=args.judge_revision
    )
    llm = LLM(
        model=args.judge_model,
        revision=args.judge_revision,
        tensor_parallel_size=tp,
        gpu_memory_utilization=float(os.environ.get("SE_GPU_MEMORY_UTILIZATION", "0.90")),
        dtype=os.environ.get("SE_VLLM_DTYPE", "auto"),
        enforce_eager=True,
    )

    def render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Their temperature is 0.01, with a single retry at 1.0 when the reply is
    # neither yes nor no, defaulting to no.
    rendered = [render(p) for p in prompts]
    print(f"judging {len(rendered)} answers ...", flush=True)
    outputs = llm.generate(rendered, SamplingParams(temperature=0.01, max_tokens=args.max_tokens))

    labels: list[float | None] = [parse(o.outputs[0].text.strip()) for o in outputs]
    retry_at = [i for i, label in enumerate(labels) if label is None]
    if retry_at:
        print(f"retrying {len(retry_at)} unparseable replies at temperature 1.0")
        retries = llm.generate(
            [rendered[i] for i in retry_at],
            SamplingParams(temperature=1.0, max_tokens=args.max_tokens),
        )
        for i, out in zip(retry_at, retries):
            labels[i] = parse(out.outputs[0].text.strip())
    defaulted = sum(1 for label in labels if label is None)
    labels = [0.0 if label is None else label for label in labels]

    agree = changed_to_true = changed_to_false = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for record, label in zip(records, labels):
            old = bool(record["correctness_label"])
            new = bool(label)
            record["correctness_label_squad"] = old
            record["correctness_label"] = new
            record["correctness_judge"] = {
                "metric": "llm",
                "source": "farquhar_model_based_metric",
                "judge_model": args.judge_model,
                "judge_revision": args.judge_revision,
                "temperature": 0.01,
                "retry_temperature": 1.0,
            }
            agree += old == new
            changed_to_true += (not old) and new
            changed_to_false += old and (not new)
            target.write(json.dumps(record, sort_keys=True) + "\n")

    total = len(records)
    print(
        f"\nagreement with squad labels: {agree}/{total} ({100*agree/total:.1f}%)\n"
        f"  squad said wrong, judge says right: {changed_to_true} "
        f"({100*changed_to_true/total:.1f}%)\n"
        f"  squad said right, judge says wrong: {changed_to_false} "
        f"({100*changed_to_false/total:.1f}%)\n"
        f"  replies still unparseable after retry (defaulted to no): {defaulted}"
    )
    new_rate = sum(labels) / total
    print(f"accuracy under the judge: {new_rate:.3f}")
    if new_rate in (0.0, 1.0):
        raise SystemExit("judge returned a single label for every record")
    print(f"wrote {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
