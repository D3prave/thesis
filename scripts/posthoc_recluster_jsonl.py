"""Re-run semantic clustering on an existing scored JSONL file.

The script reuses saved ``sampled_answers`` and ``normalized_answers``. It
does not call the generation model. It updates semantic clusters,
representatives, core entropy scores, and the ``entailment_backend`` label.
Optional extension scores such as ``kle`` are preserved when already present.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.clustering import (
    NLI_ENTAILMENT,
    exact_match_entailment_fn,
    nli_cluster,
)
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import score_record

OrderedPair = tuple[str, str]
PairMap = dict[tuple[int, int], tuple[OrderedPair, OrderedPair]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--nli-model",
        default=None,
        help=(
            "Hugging Face NLI model for batched clustering. Defaults to "
            "SE_NLI_MODEL or cross-encoder/nli-deberta-v3-base."
        ),
    )
    parser.add_argument(
        "--nli-device",
        default=None,
        help="Device for the NLI pipeline. Defaults to SE_NLI_DEVICE or cpu.",
    )
    parser.add_argument(
        "--nli-batch-size",
        type=int,
        default=32,
        help="Batch size for batched NLI inference.",
    )
    parser.add_argument(
        "--entailment-backend",
        default=None,
        help="Label to write into each output record.",
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        help="Use exact-match entailment for local smoke tests.",
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
    if args.nli_batch_size <= 0:
        raise ValueError("--nli-batch-size must be positive")

    records = _load_records(args.input_jsonl)
    backend = args.entailment_backend or _default_backend(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.exact_match:
        clustered = _cluster_exact(records)
    else:
        clustered = _cluster_batched(records, args)

    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for count, record in enumerate(clustered, start=1):
            record["entailment_backend"] = backend
            if args.run_id is not None:
                record["run_id"] = args.run_id

            validate_record(record)
            check_cluster_consistency(record)
            target.write(json.dumps(record, sort_keys=True) + "\n")
            if count == 1 or count % 25 == 0:
                print(f"  wrote {count} reclustered records", flush=True)

    print(f"wrote {len(clustered)} reclustered records to {args.output_jsonl}")
    return 0


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


def _cluster_exact(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    print("using exact-match reclustering", flush=True)
    return [
        _with_updated_clusters(
            record,
            *nli_cluster(record["normalized_answers"], exact_match_entailment_fn),
        )
        for record in records
    ]


def _cluster_batched(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    pair_maps, pending_pairs, labels = _collect_pair_tasks(records)
    print(
        f"prepared {len(pending_pairs)} unique ordered NLI pairs "
        f"after exact duplicate caching",
        flush=True,
    )

    if pending_pairs:
        entailer = _BatchedNli(
            model_name=_nli_model_name(args),
            device=_nli_device(args),
            batch_size=args.nli_batch_size,
        )
        predicted = entailer.predict(pending_pairs)
        labels.update(predicted)

    clustered: list[dict[str, Any]] = []
    for index, (record, pair_map) in enumerate(zip(records, pair_maps), start=1):
        clusters, representatives = _clusters_from_pair_labels(
            record["normalized_answers"],
            pair_map,
            labels,
        )
        clustered.append(_with_updated_clusters(record, clusters, representatives))
        if index == 1 or index % 25 == 0:
            print(f"  clustered {index}/{len(records)} records", flush=True)
    return clustered


def _collect_pair_tasks(
    records: list[dict[str, Any]],
) -> tuple[list[PairMap], list[OrderedPair], dict[OrderedPair, str]]:
    labels: dict[OrderedPair, str] = {}
    pending_seen: set[OrderedPair] = set()
    pending_pairs: list[OrderedPair] = []
    pair_maps: list[PairMap] = []

    for record in records:
        answers = record["normalized_answers"]
        pair_map: PairMap = {}
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                forward = (answers[i], answers[j])
                backward = (answers[j], answers[i])
                pair_map[(i, j)] = (forward, backward)
                if answers[i] == answers[j]:
                    labels[forward] = NLI_ENTAILMENT
                    labels[backward] = NLI_ENTAILMENT
                    continue
                for pair in (forward, backward):
                    if pair not in labels and pair not in pending_seen:
                        pending_pairs.append(pair)
                        pending_seen.add(pair)
        pair_maps.append(pair_map)

    return pair_maps, pending_pairs, labels


class _BatchedNli:
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "Batched NLI requires transformers and torch. "
                "Install with: pip install -e '.[hf]'"
            ) from exc

        self.batch_size = batch_size
        self.pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            device=_pipeline_device(device),
            batch_size=batch_size,
        )
        print(
            f"loaded NLI model {model_name} on {device}; batch_size={batch_size}",
            flush=True,
        )

    def predict(self, pairs: list[OrderedPair]) -> dict[OrderedPair, str]:
        labels: dict[OrderedPair, str] = {}
        total = len(pairs)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = pairs[start:end]
            inputs = [
                {"text": premise, "text_pair": hypothesis}
                for premise, hypothesis in batch
            ]
            outputs = self.pipe(
                inputs,
                batch_size=self.batch_size,
                truncation=True,
            )
            for pair, output in zip(batch, outputs):
                labels[pair] = _canonical_label(output["label"])
            if end == total or start == 0 or end % (self.batch_size * 20) == 0:
                print(f"  NLI predicted {end}/{total} ordered pairs", flush=True)
        return labels


def _clusters_from_pair_labels(
    answers: list[str],
    pair_map: PairMap,
    labels: dict[OrderedPair, str],
) -> tuple[list[int], list[str]]:
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
        if labels[forward] == NLI_ENTAILMENT and labels[backward] == NLI_ENTAILMENT:
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


def _canonical_label(raw_label: str) -> str:
    label = raw_label.lower()
    if "entail" in label:
        return NLI_ENTAILMENT
    if "contradict" in label:
        return "contradiction"
    return "neutral"


def _pipeline_device(device: str) -> str | int:
    return -1 if device == "cpu" else device


def _nli_model_name(args: argparse.Namespace) -> str:
    return args.nli_model or os.environ.get(
        "SE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
    )


def _nli_device(args: argparse.Namespace) -> str:
    return args.nli_device or os.environ.get("SE_NLI_DEVICE", "cpu")


def _default_backend(args: argparse.Namespace) -> str:
    if args.exact_match:
        return "exact-match"
    return _nli_model_name(args).rsplit("/", maxsplit=1)[-1]


def _load_record(line: str, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise SchemaError(f"line {line_number}: empty JSONL record")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise SchemaError(f"line {line_number}: record must be a JSON object")
    return record


def _ensure_distinct_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_jsonl and output_jsonl must be different files")


if __name__ == "__main__":
    raise SystemExit(main())
