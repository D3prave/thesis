"""Re-run semantic clustering on a Phase-2 JSONL using an LLM judge.

This script is the LLM-as-judge counterpart of
:mod:`scripts.posthoc_recluster_jsonl`. It reuses saved
``sampled_answers`` and ``normalized_answers``, asks an
instruction-tuned LLM judge (Qwen2.5-72B-Instruct by default) for
bidirectional entailment on every ordered pair within each record,
runs union-find clustering on the merged pairs, and writes a new
scored JSONL with updated ``semantic_clusters``,
``cluster_representatives``, entropy scores, and
``entailment_backend``. Generation is not rerun.

Each record is judged independently --- the question is carried with
the pair so the judge has the right context. Identical normalized
strings are short-circuited as mutually entailing without consulting
the judge. Across records we deduplicate identical
``(question, premise, hypothesis)`` triples so the same triple is
never sent to the judge twice.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.clustering import NLI_ENTAILMENT
from semantic_entropy.llm_judge import BatchedLlmJudge, JudgeTriple
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import score_record


OrderedPair = tuple[str, str]
PairMap = dict[tuple[int, int], tuple[OrderedPair, OrderedPair]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "HuggingFace model ID for the LLM judge. Defaults to "
            "SE_JUDGE_MODEL or 'Qwen/Qwen2.5-72B-Instruct'."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "vLLM tensor-parallel degree. Defaults to "
            "SE_TENSOR_PARALLEL_SIZE or 4."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization. Defaults to SE_GPU_MEMORY_UTILIZATION or 0.90.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max_model_len cap. Defaults to SE_MAX_MODEL_LEN or unset.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="vLLM dtype. Defaults to SE_VLLM_DTYPE or 'auto'.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        help="Pass enforce_eager=True to vLLM (required on Alex).",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=1024,
        help="Triples reported per progress line (vLLM batches internally).",
    )
    parser.add_argument(
        "--entailment-backend",
        default=None,
        help=(
            "Label written into each output record as 'entailment_backend'. "
            "Defaults to a slug derived from the judge model id."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run_id to write into each output record.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_distinct_paths(args.input_jsonl, args.output_jsonl)
    if args.judge_batch_size <= 0:
        raise ValueError("--judge-batch-size must be positive")

    records = _load_records(args.input_jsonl)
    judge_model = _resolve_judge_model(args)
    backend = args.entailment_backend or _default_backend(judge_model)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect all (question, premise, hypothesis) triples to judge.
    # Identical-string ordered pairs are pre-resolved as entailment.
    # ------------------------------------------------------------------
    pair_maps, pending_triples, labels = _collect_triple_tasks(records)
    print(
        f"prepared {len(pending_triples)} unique LLM-judge triples "
        f"after identical-string short-circuiting "
        f"and cross-record deduplication",
        flush=True,
    )

    if pending_triples:
        judge = BatchedLlmJudge(
            model_name=judge_model,
            tensor_parallel_size=_resolve_int(
                args.tensor_parallel_size, "SE_TENSOR_PARALLEL_SIZE", 4
            ),
            gpu_memory_utilization=_resolve_float(
                args.gpu_memory_utilization,
                "SE_GPU_MEMORY_UTILIZATION",
                0.90,
            ),
            dtype=args.dtype or os.environ.get("SE_VLLM_DTYPE", "auto"),
            max_model_len=_resolve_optional_int(
                args.max_model_len, "SE_MAX_MODEL_LEN"
            ),
            enforce_eager=_resolve_enforce_eager(args.enforce_eager),
        )
        predicted = judge.judge_in_batches(
            pending_triples, batch_size=args.judge_batch_size
        )
        labels.update(predicted)

    # ------------------------------------------------------------------
    # Apply union-find per record using the prefilled label table.
    # ------------------------------------------------------------------
    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for index, (record, pair_map) in enumerate(
            zip(records, pair_maps), start=1
        ):
            clusters, representatives = _clusters_from_triple_labels(
                record["prompt"],
                record["normalized_answers"],
                pair_map,
                labels,
            )
            updated = _with_updated_clusters(record, clusters, representatives)
            updated["entailment_backend"] = backend
            if args.run_id is not None:
                updated["run_id"] = args.run_id

            validate_record(updated)
            check_cluster_consistency(updated)
            target.write(json.dumps(updated, sort_keys=True) + "\n")
            if index == 1 or index % 25 == 0:
                print(
                    f"  wrote {index}/{len(records)} reclustered records",
                    flush=True,
                )

    print(
        f"wrote {len(records)} reclustered records to {args.output_jsonl}",
        flush=True,
    )
    return 0


# ----------------------------------------------------------------------
# Triple collection and union-find
# ----------------------------------------------------------------------


def _collect_triple_tasks(
    records: list[dict[str, Any]],
) -> tuple[list[PairMap], list[JudgeTriple], dict[JudgeTriple, str]]:
    """Collect unique (question, premise, hypothesis) triples across records.

    Returns:
        ``(pair_maps, pending_triples, prefilled_labels)`` where
        ``pair_maps[k]`` maps ``(i, j)`` for record ``k`` to its forward
        and backward ordered pairs (strings only), ``pending_triples``
        is the deduplicated list of triples that still need a judge
        call, and ``prefilled_labels`` already contains
        :data:`NLI_ENTAILMENT` entries for identical-string pairs.
    """
    labels: dict[JudgeTriple, str] = {}
    pending_seen: set[JudgeTriple] = set()
    pending_triples: list[JudgeTriple] = []
    pair_maps: list[PairMap] = []

    for record in records:
        question = record["prompt"]
        answers = record["normalized_answers"]
        pair_map: PairMap = {}
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                forward = (answers[i], answers[j])
                backward = (answers[j], answers[i])
                pair_map[(i, j)] = (forward, backward)
                if answers[i] == answers[j]:
                    labels[(question,) + forward] = NLI_ENTAILMENT
                    labels[(question,) + backward] = NLI_ENTAILMENT
                    continue
                for ordered in (forward, backward):
                    triple = (question, ordered[0], ordered[1])
                    if triple not in labels and triple not in pending_seen:
                        pending_triples.append(triple)
                        pending_seen.add(triple)
        pair_maps.append(pair_map)

    return pair_maps, pending_triples, labels


def _clusters_from_triple_labels(
    question: str,
    answers: list[str],
    pair_map: PairMap,
    labels: dict[JudgeTriple, str],
) -> tuple[list[int], list[str]]:
    """Union-find on bidirectional entailment for one record's pair map."""
    if not answers:
        raise ValueError("normalized_answers must not be empty")

    parent = list(range(len(answers)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j), (forward, backward) in pair_map.items():
        fwd_triple = (question, forward[0], forward[1])
        bwd_triple = (question, backward[0], backward[1])
        if (
            labels[fwd_triple] == NLI_ENTAILMENT
            and labels[bwd_triple] == NLI_ENTAILMENT
        ):
            union(i, j)

    root_to_id: dict[int, int] = {}
    cluster_ids: list[int] = []
    for i in range(len(answers)):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_ids.append(root_to_id[root])

    representatives = [""] * len(root_to_id)
    seen_cids: set[int] = set()
    for i, cid in enumerate(cluster_ids):
        if cid not in seen_cids:
            representatives[cid] = answers[i]
            seen_cids.add(cid)

    return cluster_ids, representatives


# ----------------------------------------------------------------------
# I/O and record assembly
# ----------------------------------------------------------------------


def _with_updated_clusters(
    record: dict[str, Any],
    clusters: list[int],
    representatives: list[str],
) -> dict[str, Any]:
    updated = dict(record)
    preserved_scores = {
        key: value
        for key, value in record["scores"].items()
        if key
        not in {
            "naive_sample_entropy",
            "surface_entropy",
            "discrete_semantic_entropy",
        }
    }
    updated["semantic_clusters"] = clusters
    updated["cluster_representatives"] = representatives
    updated["scores"] = score_record(
        record["sampled_answers"],
        record["normalized_answers"],
        clusters,
    )
    updated["scores"].update(preserved_scores)
    return updated


def _load_records(input_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = _load_record(line, line_number)
            validate_record(record)
            records.append(record)
    if not records:
        raise ValueError("input_jsonl must contain at least one record")
    print(f"loaded {len(records)} records from {input_jsonl}", flush=True)
    return records


def _load_record(line: str, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise SchemaError(f"line {line_number}: empty JSONL record")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"line {line_number}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(record, dict):
        raise SchemaError(
            f"line {line_number}: record must be a JSON object"
        )
    return record


def _ensure_distinct_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "input_jsonl and output_jsonl must be different files"
        )


# ----------------------------------------------------------------------
# Config resolution helpers
# ----------------------------------------------------------------------


def _resolve_judge_model(args: argparse.Namespace) -> str:
    return (
        args.judge_model
        or os.environ.get("SE_JUDGE_MODEL")
        or "Qwen/Qwen2.5-72B-Instruct"
    )


def _resolve_int(value: int | None, env_key: str, default: int) -> int:
    if value is not None:
        return value
    raw = os.environ.get(env_key)
    return int(raw) if raw is not None else default


def _resolve_optional_int(value: int | None, env_key: str) -> int | None:
    if value is not None:
        return value
    raw = os.environ.get(env_key)
    return int(raw) if raw else None


def _resolve_float(
    value: float | None, env_key: str, default: float
) -> float:
    if value is not None:
        return value
    raw = os.environ.get(env_key)
    return float(raw) if raw is not None else default


def _resolve_enforce_eager(flag: bool | None) -> bool:
    if flag is not None:
        return flag
    raw = os.environ.get("SE_VLLM_ENFORCE_EAGER")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_backend(judge_model: str) -> str:
    slug = judge_model.rsplit("/", maxsplit=1)[-1].lower()
    return f"llm-judge-{slug}"


if __name__ == "__main__":
    raise SystemExit(main())
