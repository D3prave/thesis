#!/usr/bin/env python3
"""Recluster a scored JSONL using a generative LLM as the entailment judge.

Farquhar et al. use GPT-3.5 as the entailment judge for the sentence-length
setting of the main text and a DeBERTa cross-encoder for the short-phrase
setting of Supplementary Note 7. Paid API access is out of scope for this
study, so this rung substitutes an open model of comparable size. That is a
substitution, not a reproduction, and belongs in the deviation register.

Everything except the judge is held fixed against the cross-encoder rungs:
the same stored generations, the same order-dependent anchor scan including its
reassignment behaviour, the same strict bidirectional merge rule, the same
scoring. The judge is therefore the only variable, which is the whole point of
the ladder.

Their released ``entailment.py`` prompts the model per ordered pair, naming the
question and both answers, and asks for one of three verdicts. That prompt is
reproduced verbatim below.

Batching note: the ten samples per record are heavily duplicated, so pairs are
formed over *distinct* answers and deduplicated globally by
(question, premise, hypothesis) before being sent to the model. Identical
answers are labelled entailment without asking, exactly as the cross-encoder
path does.


Usage:
    python scripts/posthoc_recluster_llm_judge.py IN.jsonl OUT.jsonl \
        --judge-model Qwen/Qwen2.5-72B-Instruct --judge-revision <40hex> \
        --entailment-backend qwen2.5-72b-instruct --run-id <id>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from semantic_entropy.cluster_check import check_cluster_consistency  # noqa: E402
from semantic_entropy.clustering import NLI_ENTAILMENT  # noqa: E402
from semantic_entropy.schema import validate_record  # noqa: E402


def _load_sibling_module():
    """Import the cross-encoder reclustering script for its shared helpers.

    Importing rather than reimplementing matters: the anchor scan, the merge
    rule and the score recomputation must be bit-identical to the other rungs
    or the comparison measures the implementation instead of the judge.
    """
    path = REPO_ROOT / "scripts" / "posthoc_recluster_jsonl.py"
    spec = importlib.util.spec_from_file_location("_posthoc_recluster", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PR = _load_sibling_module()

# Verbatim from their released entailment.py (EntailmentGPT.check_implication).
ENTAILMENT_PROMPT = (
    'We are evaluating answers to the question "{question}"\n'
    "Here are two possible answers:\n"
    "Possible Answer 1: {text1}\n"
    "Possible Answer 2: {text2}\n"
    "Does Possible Answer 1 semantically entail Possible Answer 2? "
    "Respond with entailment, contradiction, or neutral."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_jsonl", type=Path)
    p.add_argument("output_jsonl", type=Path)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-revision", default=None)
    p.add_argument("--entailment-backend", default=None,
                   help="label written into each record; defaults to the model id")
    p.add_argument("--run-id", default=None)
    p.add_argument("--cluster-source", choices=("raw", "normalized"), default="raw")
    p.add_argument("--limit", type=int, default=0, help="0 means all records")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--tensor-parallel-size", type=int, default=0,
                   help="0 means autodetect from visible GPUs")
    p.add_argument("--print-prompt", action="store_true",
                   help="print the first prompt and exit without loading a model")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    records = PR._load_records(args.input_jsonl)
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} records from {args.input_jsonl}")

    # ---- pair collection over distinct answers -------------------------------
    # Key includes the question: the judge is explicitly question-conditioned by
    # the prompt, so the same answer pair under two different questions is a
    # different query and must not share a cached verdict.
    pair_maps: list[dict[tuple[int, int], tuple[tuple, tuple]]] = []
    labels: dict[tuple, str] = {}
    pending: list[tuple] = []
    seen: set[tuple] = set()

    for record in records:
        answers = PR._cluster_answers(record, args.cluster_source)
        question = record["prompt"]
        pair_map: dict[tuple[int, int], tuple[tuple, tuple]] = {}
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                forward = (question, answers[i], answers[j])
                backward = (question, answers[j], answers[i])
                pair_map[(i, j)] = (forward, backward)
                if answers[i] == answers[j]:
                    labels[forward] = NLI_ENTAILMENT
                    labels[backward] = NLI_ENTAILMENT
                    continue
                for key in (forward, backward):
                    if key not in labels and key not in seen:
                        pending.append(key)
                        seen.add(key)
        pair_maps.append(pair_map)

    total_pairs = sum(len(m) for m in pair_maps) * 2
    print(
        f"{total_pairs} ordered pairs, {len(pending)} need the judge "
        f"({100 * (1 - len(pending) / max(total_pairs, 1)):.1f}% resolved by "
        "identity or repetition)"
    )

    if args.print_prompt:
        question, text1, text2 = pending[0]
        print("\n--- first prompt ---")
        print(ENTAILMENT_PROMPT.format(question=question, text1=text1, text2=text2))
        return 0

    # ---- run the judge -------------------------------------------------------
    import os

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tp = args.tensor_parallel_size
    if tp <= 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        tp = len(visible.split(",")) if visible else 1
        try:
            import torch

            tp = torch.cuda.device_count() or tp
        except Exception:
            pass
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
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": ENTAILMENT_PROMPT.format(
                        question=question, text1=text1, text2=text2
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question, text1, text2 in pending
    ]

    print(f"judging {len(prompts)} pairs ...", flush=True)
    outputs = llm.generate(prompts, params)

    verdict_counts: dict[str, int] = {}
    for key, output in zip(pending, outputs):
        raw = output.outputs[0].text.strip()
        label = PR._canonical_label(raw)
        labels[key] = label
        verdict_counts[label] = verdict_counts.get(label, 0) + 1
    print("verdicts:", dict(sorted(verdict_counts.items())))
    if len(verdict_counts) == 1:
        raise SystemExit(
            "the judge returned a single verdict for every pair; the prompt or "
            "the parse is wrong (fail-closed)"
        )

    # ---- cluster and rescore -------------------------------------------------
    backend = args.entailment_backend or args.judge_model
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for record, pair_map in zip(records, pair_maps):
            answers = PR._cluster_answers(record, args.cluster_source)
            clusters, _raw_representatives = PR._clusters_by_anchor_scan(
                answers, pair_map, labels
            )
            # Clustering runs over the raw generations, but the stored
            # representatives must come from normalized_answers or
            # check_cluster_consistency rejects the record. The cross-encoder
            # path overrides them the same way.
            representatives = PR._representatives_from_normalized(record, clusters)
            updated = PR._with_updated_clusters(record, clusters, representatives)
            updated["entailment_backend"] = backend
            updated["nli_model_id"] = args.judge_model
            if args.judge_revision:
                updated["nli_model_revision"] = args.judge_revision
                updated["nli_tokenizer_revision"] = args.judge_revision
            updated["nli_tokenizer_id"] = args.judge_model
            if args.run_id:
                updated["run_id"] = args.run_id
            validate_record(updated)
            check_cluster_consistency(updated)
            target.write(json.dumps(updated, sort_keys=True) + "\n")
            written += 1
            if written == 1 or written % 100 == 0:
                print(f"  wrote {written}", flush=True)

    print(f"wrote {written} reclustered records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
