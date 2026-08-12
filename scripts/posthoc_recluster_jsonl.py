"""Re-run semantic clustering on an existing scored JSONL file.

The script reuses saved ``sampled_answers`` and ``normalized_answers``. It
does not call the generation model. It updates semantic clusters,
representatives, core entropy scores, and the ``entailment_backend`` label.
Optional extension scores such as ``kle`` are preserved when already present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.clustering import (
    NLI_CONTRADICTION,
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
    parser.add_argument(
        "--clustering",
        choices=["strict", "nondefeating", "anchor"],
        default="strict",
        help="Bidirectional-entailment rule and closure. 'strict' requires "
        "entailment in both directions and takes a transitive "
        "connected-components closure (default); 'nondefeating' matches "
        "Farquhar et al.'s strict_entailment=False (no contradiction either "
        "way, >=1 entailment); 'anchor' requires entailment in both "
        "directions but assigns clusters with the order-dependent anchor scan "
        "of the Farquhar et al. main text, comparing each answer only against "
        "the first member of each open cluster.",
    )
    parser.add_argument(
        "--question-format",
        choices=["labelled", "paper"],
        default="labelled",
        help="How the question is attached to each answer for "
        "question-conditioned entailment. 'labelled' (default) uses "
        "'Question: ...\\nAnswer: ...' and reproduces this study's earlier "
        "runs; 'paper' joins them with a single space and no labels, matching "
        "the released compute_uncertainty_measures.py.",
    )
    parser.add_argument(
        "--cluster-source",
        choices=["normalized", "raw"],
        default="normalized",
        help="Which strings are fed to the entailment model. 'normalized' "
        "(default) uses normalized_answers; 'raw' uses sampled_answers, "
        "matching Farquhar et al., who cluster the generations as produced. "
        "The surface-entropy baseline is scored over normalized_answers "
        "either way, so this flag does not disturb it.",
    )
    parser.add_argument(
        "--question-conditioned",
        action="store_true",
        help=("Prefix both answers in every NLI pair with the record question. "
              "This is required for the v2 primary analysis."),
    )
    parser.add_argument(
        "--source-split-role",
        choices=["train", "eval"],
        default=None,
        help=(
            "Explicit role of the source artifact. Required for v2 "
            "question-conditioned runs so train and held-out evaluation "
            "artifacts remain distinguishable downstream."
        ),
    )
    parser.add_argument(
        "--nli-revision",
        default=None,
        help="Exact Hugging Face NLI model revision; required for v2 NLI runs.",
    )
    parser.add_argument(
        "--nli-tokenizer-revision",
        default=None,
        help=(
            "Exact Hugging Face NLI tokenizer revision; required for v2 NLI "
            "runs and loaded explicitly instead of being inferred by pipeline."
        ),
    )
    parser.add_argument(
        "--code-commit",
        default=None,
        help="Clean repository commit used for the run; required for v2 runs.",
    )
    parser.add_argument(
        "--job-meta-output",
        type=Path,
        default=None,
        help="Optional normalized v2 job-metadata sidecar written after success.",
    )
    parser.add_argument(
        "--protocol-manifest",
        default=None,
        help="Path of the locked SEP v2 protocol manifest, when applicable.",
    )
    parser.add_argument(
        "--protocol-manifest-sha256",
        default=None,
        help="Verified SHA-256 of the locked SEP v2 protocol manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_distinct_paths(args.input_jsonl, args.output_jsonl)
    if args.nli_batch_size <= 0:
        raise ValueError("--nli-batch-size must be positive")
    if args.question_conditioned:
        _validate_v2_args(args)

    records = _load_records(args.input_jsonl)
    locked_source_run_id: str | None = None
    if args.question_conditioned:
        source_run_ids = {str(record.get("run_id", "")).strip() for record in records}
        if "" in source_run_ids or len(source_run_ids) != 1:
            raise ValueError(
                "question-conditioned input must contain one non-empty source run_id"
            )
        locked_source_run_id = next(iter(source_run_ids))
    backend = args.entailment_backend or _default_backend(args)
    source_sha256 = _sha256_file(args.input_jsonl)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.exact_match:
        clustered = _cluster_exact(records, args.cluster_source)
    else:
        clustered = _cluster_batched(records, args)

    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for count, record in enumerate(clustered, start=1):
            source_run_id = str(record.get("run_id", ""))
            record["entailment_backend"] = backend
            record["question_conditioned_entailment"] = args.question_conditioned
            # "question_answer_v1" is the historical label for the labelled
            # format and is recorded in already-sealed artifacts, so it is kept
            # rather than renamed.
            record["entailment_input_format"] = (
                ("question_answer_v1" if args.question_format == "labelled"
                 else "question_answer_paper")
                if args.question_conditioned
                else "answer_only"
            )
            record["source_run_id"] = source_run_id
            record["source_artifact_sha256"] = source_sha256
            record["analysis_code_commit"] = args.code_commit or "unrecorded"
            record["code_commit"] = args.code_commit or "unrecorded"
            record["question_conditioned"] = args.question_conditioned
            if args.question_conditioned:
                record["split_role"] = args.source_split_role
                if args.protocol_manifest_sha256 is not None:
                    record["protocol_manifest_sha256"] = (
                        args.protocol_manifest_sha256
                    )
            record["clustering_rule"] = (
                "nondefeating" if args.clustering == "nondefeating" else "bidirectional"
            )
            record["clustering_closure"] = (
                "anchor_scan" if args.clustering == "anchor" else "connected_components"
            )
            record["cluster_input"] = (
                "sampled_answers"
                if args.cluster_source == "raw"
                else "normalized_answers"
            )
            if not args.exact_match:
                nli_model = _nli_model_name(args)
                nli_revision = _nli_revision(args) or "unrecorded"
                nli_tokenizer_revision = (
                    _nli_tokenizer_revision(args) or "unrecorded"
                )
                record["entailment_model_id"] = nli_model
                record["entailment_model_revision"] = nli_revision
                record["nli_model_id"] = nli_model
                record["nli_model_revision"] = nli_revision
                record["nli_tokenizer_id"] = nli_model
                record["nli_tokenizer_revision"] = nli_tokenizer_revision
            if args.run_id is not None:
                record["run_id"] = args.run_id

            validate_record(record)
            check_cluster_consistency(record)
            target.write(json.dumps(record, sort_keys=True) + "\n")
            if count == 1 or count % 25 == 0:
                print(f"  wrote {count} reclustered records", flush=True)

    if args.job_meta_output is not None:
        _write_v2_job_metadata(
            args.job_meta_output,
            records=records,
            run_id=args.run_id,
            source_run_id=locked_source_run_id,
            source_sha256=source_sha256,
            args=args,
        )

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


def _representatives_from_normalized(
    record: dict[str, Any], cluster_ids: list[int]
) -> list[str]:
    """Cluster representatives taken from ``normalized_answers``.

    Clusters may be induced over raw generations, but the schema invariant
    checked by :func:`check_cluster_consistency` requires each representative to
    appear among the normalized answers. Taking the first member of each cluster
    by index keeps that invariant regardless of the clustering input.
    """
    normalized = record["normalized_answers"]
    representatives = [""] * (max(cluster_ids) + 1)
    seen: set[int] = set()
    for index, cid in enumerate(cluster_ids):
        if cid not in seen:
            representatives[cid] = normalized[index]
            seen.add(cid)
    return representatives


def _cluster_answers(
    record: dict[str, Any], cluster_source: str = "normalized"
) -> list[str]:
    """Strings fed to the entailment model for this record.

    ``raw`` selects the generations as produced, matching Farquhar et al.
    The surface-entropy baseline is scored separately over
    ``normalized_answers`` and is unaffected by this choice.
    """
    field = "sampled_answers" if cluster_source == "raw" else "normalized_answers"
    answers = record.get(field)
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"record is missing a non-empty {field}")
    return answers


def _cluster_exact(
    records: list[dict[str, Any]], cluster_source: str = "normalized"
) -> list[dict[str, Any]]:
    print("using exact-match reclustering", flush=True)
    return [
        _with_updated_clusters(
            record,
            *nli_cluster(
                _cluster_answers(record, cluster_source), exact_match_entailment_fn
            ),
        )
        for record in records
    ]


def _cluster_batched(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    cluster_source = getattr(args, "cluster_source", "normalized")
    print(f"clustering over {cluster_source} answers", flush=True)
    pair_maps, pending_pairs, labels = _collect_pair_tasks(
        records,
        question_conditioned=args.question_conditioned,
        cluster_source=cluster_source,
        question_format=getattr(args, "question_format", "labelled"),
    )
    print(
        f"prepared {len(pending_pairs)} unique ordered NLI pairs "
        f"after exact duplicate caching",
        flush=True,
    )

    if pending_pairs:
        entailer = _BatchedNli(
            model_name=_nli_model_name(args),
            model_revision=_nli_revision(args),
            tokenizer_revision=_nli_tokenizer_revision(args),
            device=_nli_device(args),
            batch_size=args.nli_batch_size,
        )
        predicted = entailer.predict(pending_pairs)
        labels.update(predicted)

    mode = getattr(args, "clustering", "strict")
    clustered: list[dict[str, Any]] = []
    for index, (record, pair_map) in enumerate(zip(records, pair_maps), start=1):
        answers = _cluster_answers(record, cluster_source)
        if mode == "anchor":
            clusters, representatives = _clusters_by_anchor_scan(
                answers, pair_map, labels
            )
        else:
            clusters, representatives = _clusters_from_pair_labels(
                answers,
                pair_map,
                labels,
                strict=(mode == "strict"),
            )
        representatives = _representatives_from_normalized(record, clusters)
        clustered.append(_with_updated_clusters(record, clusters, representatives))
        if index == 1 or index % 25 == 0:
            print(f"  clustered {index}/{len(records)} records", flush=True)
    return clustered


def _collect_pair_tasks(
    records: list[dict[str, Any]],
    *,
    question_conditioned: bool = False,
    cluster_source: str = "normalized",
    question_format: str = "labelled",
) -> tuple[list[PairMap], list[OrderedPair], dict[OrderedPair, str]]:
    labels: dict[OrderedPair, str] = {}
    pending_seen: set[OrderedPair] = set()
    pending_pairs: list[OrderedPair] = []
    pair_maps: list[PairMap] = []

    for record in records:
        answers = _cluster_answers(record, cluster_source)
        entailment_inputs = (
            [
                _question_answer_input(record["prompt"], answer, question_format)
                for answer in answers
            ]
            if question_conditioned
            else answers
        )
        pair_map: PairMap = {}
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                forward = (entailment_inputs[i], entailment_inputs[j])
                backward = (entailment_inputs[j], entailment_inputs[i])
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
    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
        device: str,
        batch_size: int,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "Batched NLI requires transformers and torch. "
                "Install with: pip install -e '.[hf]'"
            ) from exc

        self.batch_size = batch_size
        tokenizer_kwargs: dict[str, Any] = {}
        if tokenizer_revision is not None:
            tokenizer_kwargs["revision"] = tokenizer_revision
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        pipeline_kwargs: dict[str, Any] = {
            "model": model_name,
            "tokenizer": tokenizer,
            "device": _pipeline_device(device),
            "batch_size": batch_size,
        }
        if model_revision is not None:
            pipeline_kwargs["revision"] = model_revision
        self.pipe = hf_pipeline("text-classification", **pipeline_kwargs)
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


def _merges(
    pair_labels: tuple[str, str],
    mode: str,
) -> bool:
    """Whether two answers are equivalent under the configured relation."""
    forward, backward = pair_labels
    if mode == "nondefeating":
        both = {forward, backward}
        return NLI_CONTRADICTION not in both and NLI_ENTAILMENT in both
    return forward == NLI_ENTAILMENT and backward == NLI_ENTAILMENT


def _clusters_by_anchor_scan(
    answers: list[str],
    pair_map: PairMap,
    labels: dict[OrderedPair, str],
) -> tuple[list[int], list[str]]:
    """Order-dependent anchor scan of the Farquhar et al. main text.

    Each unassigned answer opens a cluster and every later unassigned answer is
    compared against that anchor alone. This is the published algorithm: "it is
    sufficient to check if the sequence bidirectionally entails any of the
    existing sequences in that cluster (we arbitrarily pick the first one),
    given the transitivity of semantic equivalence."

    Supplementary Note 3 concedes transitivity fails in practice, so this
    returns at least as many clusters as the connected-components closure.
    """
    if not answers:
        raise ValueError("cluster source must not be empty")

    cluster_ids = [-1] * len(answers)
    next_id = 0
    for i in range(len(answers)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_id
        for j in range(i + 1, len(answers)):
            # No guard on cluster_ids[j]: the released get_semantic_ids
            # overwrites an existing assignment, so a later anchor can take a
            # member from an earlier cluster. Reproduced deliberately.
            key = (i, j) if (i, j) in pair_map else (j, i)
            forward, backward = pair_map[key]
            if _merges((labels[forward], labels[backward]), "strict"):
                cluster_ids[j] = next_id
        next_id += 1

    representatives = [answers[cluster_ids.index(cid)] for cid in range(next_id)]
    return cluster_ids, representatives


def _clusters_from_pair_labels(
    answers: list[str],
    pair_map: PairMap,
    labels: dict[OrderedPair, str],
    strict: bool = True,
) -> tuple[list[int], list[str]]:
    """Union-find clustering from directional NLI labels.

    ``strict=True`` (default) merges two answers only when *both* directions are
    entailment (Farquhar et al. ``strict_entailment=True``; our
    :func:`semantic_entropy.clustering.nli_cluster`). ``strict=False`` matches
    their default ``strict_entailment=False``: merge when neither direction is a
    contradiction and at least one is entailment (the "non-defeating" rule).

    Both settings take a transitive connected-components closure. For the
    published anchor scan see :func:`_clusters_by_anchor_scan`.
    """
    if not answers:
        raise ValueError("cluster source must not be empty")

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
        pair_labels = {labels[forward], labels[backward]}
        if strict:
            merge = labels[forward] == NLI_ENTAILMENT and labels[backward] == NLI_ENTAILMENT
        else:
            merge = (
                NLI_CONTRADICTION not in pair_labels and NLI_ENTAILMENT in pair_labels
            )
        if merge:
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
    # Recompute the clustering-dependent entropies under the new clusters
    # (discrete and full semantic entropy). naive_* entropies are clustering-
    # independent but score_record recomputes them identically when logprobs
    # are present, so they are excluded from preservation too.
    preserved_scores = {
        key: value
        for key, value in record["scores"].items()
        if key
        not in {
            "naive_sample_entropy",
            "surface_entropy",
            "discrete_semantic_entropy",
            "naive_entropy",
            "semantic_entropy_full",
        }
    }
    updated["semantic_clusters"] = clusters
    updated["cluster_representatives"] = representatives
    updated["scores"] = score_record(
        record["sampled_answers"],
        record["normalized_answers"],
        clusters,
        sequence_logprobs=record.get("sequence_logprobs"),
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


def _question_answer_input(
    question: str, answer: str, question_format: str = "labelled"
) -> str:
    """Build the string handed to the entailment model.

    Two formats. ``paper`` reproduces the released
    ``compute_uncertainty_measures.py``, which conditions a DeBERTa backend by
    joining question and answer with a single space and no labels::

        if args.condition_on_question and args.entailment_model == 'deberta':
            responses = [f'{question} {r}' for r in responses]

    ``labelled`` is this study's earlier format, which adds ``Question:`` and
    ``Answer:`` markers on separate lines. It is retained as the default so
    that previously recorded runs reproduce exactly; the faithful runs pass
    ``paper``. The choice is not cosmetic: the entailment model scores the
    literal string, so it moves cluster boundaries on every record.
    """
    question_text = str(question).strip()
    if not question_text:
        raise ValueError("question-conditioned entailment requires a non-empty prompt")
    if question_format == "paper":
        return f"{question_text} {answer}"
    if question_format == "labelled":
        return f"Question: {question_text}\nAnswer: {answer}"
    raise ValueError(f"unknown question format {question_format!r}")


def _nli_model_name(args: argparse.Namespace) -> str:
    return args.nli_model or os.environ.get(
        "SE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
    )


def _nli_revision(args: argparse.Namespace) -> str | None:
    return args.nli_revision or os.environ.get("SE_NLI_REVISION")


def _nli_tokenizer_revision(args: argparse.Namespace) -> str | None:
    return args.nli_tokenizer_revision or os.environ.get(
        "SE_NLI_TOKENIZER_REVISION"
    )


def _nli_device(args: argparse.Namespace) -> str:
    return args.nli_device or os.environ.get("SE_NLI_DEVICE", "cpu")


def _default_backend(args: argparse.Namespace) -> str:
    if args.exact_match:
        backend = "exact-match"
    else:
        backend = _nli_model_name(args).rsplit("/", maxsplit=1)[-1]
    return f"{backend}-qcond" if args.question_conditioned else backend


def _validate_v2_args(args: argparse.Namespace) -> None:
    if args.run_id is None or not args.run_id.strip():
        raise ValueError("--run-id is required with --question-conditioned")
    if args.source_split_role not in {"train", "eval"}:
        raise ValueError(
            "--source-split-role=train|eval is required with "
            "--question-conditioned"
        )
    if not _is_full_revision(args.code_commit):
        raise ValueError(
            "--code-commit must be a full 40-character hexadecimal commit"
        )
    if not args.exact_match and not _is_full_revision(_nli_revision(args)):
        raise ValueError(
            "--nli-revision must be a full 40-character hexadecimal revision"
        )
    if not args.exact_match and not _is_full_revision(
        _nli_tokenizer_revision(args)
    ):
        raise ValueError(
            "--nli-tokenizer-revision must be a full 40-character "
            "hexadecimal revision"
        )
    if (args.protocol_manifest is None) != (
        args.protocol_manifest_sha256 is None
    ):
        raise ValueError(
            "--protocol-manifest and --protocol-manifest-sha256 must be provided together"
        )
    if args.protocol_manifest_sha256 is not None and not _is_sha256(
        args.protocol_manifest_sha256
    ):
        raise ValueError(
            "--protocol-manifest-sha256 must contain 64 lowercase hexadecimal characters"
        )


def _is_full_revision(value: str | None) -> bool:
    return (
        value is not None
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_value(records: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [record.get(field) for record in records]
    canonical = {json.dumps(value, sort_keys=True) for value in values}
    if len(canonical) != 1 or values[0] in (None, ""):
        raise ValueError(f"v2 input must contain one non-empty common {field}")
    return values[0]


def _write_v2_job_metadata(
    path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    run_id: str | None,
    source_run_id: str | None,
    source_sha256: str,
    args: argparse.Namespace,
) -> None:
    if not args.question_conditioned or args.exact_match:
        raise ValueError(
            "--job-meta-output is reserved for question-conditioned NLI v2 runs"
        )
    decoding = _common_value(records, "decoding")
    if not isinstance(decoding, dict):
        raise ValueError("v2 input decoding must be an object")
    seed = decoding.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("v2 input decoding.seed must be an integer")
    nli_model = _nli_model_name(args)
    nli_revision = _nli_revision(args)
    nli_tokenizer_revision = _nli_tokenizer_revision(args)
    payload = {
        "run_id": run_id,
        "source_jsonl": str(args.input_jsonl),
        "source_run_id": source_run_id,
        "source_artifact_sha256": source_sha256,
        "dataset": _common_value(records, "dataset"),
        "model": _common_value(records, "model"),
        "seed": seed,
        "split": _common_value(records, "split"),
        "split_role": args.source_split_role,
        "code_commit": args.code_commit,
        "nli_model_id": nli_model,
        "nli_model_revision": nli_revision,
        "nli_tokenizer_id": nli_model,
        "nli_tokenizer_revision": nli_tokenizer_revision,
        "question_conditioned": True,
        "entailment_input_format": "question_answer_v1",
        "clustering_rule": (
            "bidirectional" if args.clustering == "strict" else "nondefeating"
        ),
        "decoding": decoding,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.protocol_manifest_sha256 is not None:
        payload["protocol_manifest"] = args.protocol_manifest
        payload["protocol_manifest_sha256"] = args.protocol_manifest_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
