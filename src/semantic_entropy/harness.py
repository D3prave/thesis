"""Sampling harness for Phase 1 pipeline plumbing.

This module bridges the dataset ingestion layer (:mod:`semantic_entropy.datasets`)
and the model sampling / scoring steps. It defines :class:`RunConfig`, which
bundles every piece of run metadata needed to produce a fully-specified JSONL
record, and a chain of pipeline functions:

* :func:`make_presampling_record` — converts a
  :class:`~semantic_entropy.datasets.PromptItem` + :class:`RunConfig` into a
  *pre-sampling* dict with all schema keys present and pending fields left
  empty or ``None``.
* :func:`sample_record` — calls a caller-supplied model function to fill
  ``sampled_answers``, then derives ``normalized_answers``,
  ``semantic_clusters``, and ``cluster_representatives`` using the
  current clustering backend.
* :func:`evaluate_correctness` — sets ``correctness_label`` (bool) by
  checking whether any normalized sampled answer matches any normalized
  reference answer.
* :func:`run_pipeline` — chains all of the above with entropy scoring and
  schema validation, writing fully-scored JSONL records to disk.

Pre-sampling records produced by :func:`make_presampling_record` are
intentionally *not* valid against
:func:`semantic_entropy.schema.validate_record` — they are an intermediate
format used only within the pipeline. The validation gate applies only to
records that have passed every downstream step, as enforced inside
:func:`run_pipeline`.

The module also provides :func:`make_presampling_jsonl` for writing
pre-sampling records without running inference, and a :func:`main` CLI
entry point registered as ``semantic-entropy-make-presampling-jsonl`` in
``pyproject.toml``.

**Example pipeline usage**::

    from semantic_entropy.harness import RunConfig, run_pipeline
    from semantic_entropy.datasets import load_triviaqa_records

    def my_model(prompt: str, num_samples: int) -> list[str]:
        # Replace with real model call
        return [f"Paris (sample {i})" for i in range(num_samples)]

    config = RunConfig(run_id="pilot-001", model="my-model", model_tier="small")
    items = load_triviaqa_records()
    n = run_pipeline(items, config, my_model, Path("out/scored.jsonl"))
    print(f"wrote {n} records")

**Example CLI usage**::

    # Both stubs, default smoke config
    semantic-entropy-make-presampling-jsonl out/presampling.jsonl

    # TriviaQA only, custom run ID
    semantic-entropy-make-presampling-jsonl --dataset triviaqa \\
        --run-id pilot-001 out/tqa_presampling.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_entropy.cluster_check import check_cluster_consistency
from semantic_entropy.clustering import (
    NliFn,
    exact_match_entailment_fn,
    nli_cluster,
)
from semantic_entropy.datasets import (
    SVAMP_STUB_PATH,
    TRIVIAQA_STUB_PATH,
    DatasetError,
    PromptItem,
    load_svamp_records,
    load_triviaqa_records,
)
from semantic_entropy.eval_normalize import normalize_answers_for_dataset
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import score_record

# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Immutable bundle of run-level metadata for a Phase 1/2 experiment.

    All fields except *run_id* have defaults for a small smoke run.

    Attributes:
        run_id: Unique identifier for this run, e.g. ``"smoke-001"`` or
            ``"phase1-tinygpu-7b-seed0"``.
        phase: One of ``"phase1"`` or ``"phase2"``.
        cluster: Compute cluster name: ``"local"``, ``"tinygpu"``, or
            ``"alex"``.
        model: Model name or path string, e.g. ``"synthetic-model"`` or
            ``"meta-llama/Llama-3.1-8B-Instruct"``.
        model_tier: One of ``"small"``, ``"7b_8b"``, or ``"70b_plus"``.
        entailment_backend: NLI backend used for semantic clustering, e.g.
            ``"synthetic-preclustered"`` or ``"cross-encoder/nli-deberta-v3-base"``.
        num_samples: Number of model samples per prompt (``M`` in the thesis).
        temperature: Decoding temperature.
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum number of new tokens per sample.
        seed: Global RNG seed for reproducibility.
        task: Task type: ``"qa"`` (default, TriviaQA/SVAMP) or ``"bio"``
            (long-form biography generation). When ``"bio"``, the default
            ``max_new_tokens`` rises to 256 and free-form normalization is
            applied instead of dataset-specific normalization.
    """

    run_id: str
    phase: str = "phase1"
    cluster: str = "local"
    model: str = "synthetic-model"
    model_tier: str = "small"
    entailment_backend: str = "synthetic-preclustered"
    num_samples: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_new_tokens: int = 64
    seed: int = 0
    task: str = "qa"  # "qa" or "bio"
    # When True (QA only), set correctness_label from the single most-likely
    # answer using the paper's rule (SQuAD-F1 > 0.5 for free-text, exact for
    # SVAMP), matching Farquhar et al. (2024). Default False preserves the
    # repo's "any sample matches a reference" labeling.
    paper_accuracy: bool = False


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def make_presampling_record(item: PromptItem, config: RunConfig) -> dict[str, Any]:
    """Return a pre-sampling record dict for *item* under *config*.

    The returned dict has every top-level field defined in the experimental
    JSONL schema, with the following pending fields left empty or ``None``
    to be filled by downstream pipeline steps:

    * ``sampled_answers`` — populated by the model sampling step.
    * ``normalized_answers`` — populated by the normalization step.
    * ``semantic_clusters`` — populated by the NLI clustering step.
    * ``cluster_representatives`` — populated by the NLI clustering step.
    * ``scores`` — populated by the scoring step.
    * ``correctness_label`` — populated after answer evaluation.

    Args:
        item: Source prompt extracted from a dataset.
        config: Run-level metadata and decoding hyperparameters.

    Returns:
        A ``dict`` with all schema-level keys present. The ``scores``
        sub-dict is an empty mapping and ``correctness_label`` is ``None``.
    """
    return {
        "run_id": config.run_id,
        "phase": config.phase,
        "cluster": config.cluster,
        "prompt_id": item.prompt_id,
        "dataset": item.dataset,
        "split": item.split,
        "prompt": item.prompt,
        "reference_answers": list(item.reference_answers),
        "model": config.model,
        "model_tier": config.model_tier,
        "decoding": {
            "num_samples": config.num_samples,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
        },
        "sampled_answers": [],
        "normalized_answers": [],
        "semantic_clusters": [],
        "cluster_representatives": [],
        "scores": {},
        "entailment_backend": config.entailment_backend,
        "correctness_label": None,
    }


def make_presampling_jsonl(
    items: Iterable[PromptItem],
    config: RunConfig,
    output_path: Path,
) -> int:
    """Write pre-sampling records to *output_path* in JSONL format.

    Args:
        items: Sequence of :class:`~semantic_entropy.datasets.PromptItem`
            objects to convert.
        config: Run-level metadata applied to every record.
        output_path: Destination file path. Created or overwritten.

    Returns:
        The number of records written.

    Raises:
        OSError: If *output_path* cannot be opened for writing.
    """
    records_written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            record = make_presampling_record(item, config)
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            records_written += 1
    return records_written


# ---------------------------------------------------------------------------
# Sampling, correctness evaluation, and end-to-end pipeline
# ---------------------------------------------------------------------------

#: Type alias for a model sampling function.
#:
#: A ``ModelFn`` receives a *prompt* string and the requested number of
#: independent samples *num_samples*, and returns a list of exactly
#: *num_samples* answer strings. Each string is one raw model generation
#: (untokenized, untruncated beyond the caller's ``max_new_tokens`` budget).
#:
#: For smoke testing, a trivial implementation suffices::
#:
#:     def stub_model(prompt: str, num_samples: int) -> list[str]:
#:         return ["Paris"] * num_samples
ModelFn = Callable[[str, int], list[str]]


# Markers that signal the answer is OVER (a hallucinated new chat turn / system
# block / fabricated next question). We cut everything from here on. These are
# NOT answer wrappers — wrappers like [ANSWER]/[OUT] are stripped, not cut, so a
# prefix wrapper does not empty the answer.
_ANSWER_STOP_MARKERS = (
    "</s>", "<s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    "[QUESTION]", "[/QUESTION]",
)
# Paired answer wrappers: keep the inner text.
_ANSWER_WRAPPERS = (
    ("<A>", "</A>"), ("[CHAT]", "[/CHAT]"), ("[ANSWER]", "[/ANSWER]"),
    ("[OUT]", "[/OUT]"), ("[RESP]", "[/RESP]"), ("[ANS]", "[/ANS]"),
)


def _extract_answer(raw: str, single_line: bool = True) -> str:
    """Strip chat-template echo / hallucinated next-turn text from one sample.

    Instruct models without stop sequences run past their answer into a fake
    new chat turn (``"<A>Davis</A>\\n\\n<s>[INST] ..."``); keeping that text
    contaminates clustering, entropy, and the string-match correctness label. This unwraps
    answer tags, cuts at the first template marker, and — for short-answer
    datasets (``single_line=True``) — keeps only the first non-empty line.
    For paragraph tasks (biography) pass ``single_line=False`` so multi-line
    answers survive; only the template echo is removed.
    """
    s = raw.replace("<0x0A>", "\n").replace("<0x0D>", "")
    for op, cl in _ANSWER_WRAPPERS:
        if op in s and cl in s:
            inner = s.split(op, 1)[1].split(cl, 1)[0]
            if inner.strip():
                s = inner
                break
    cut = len(s)
    for marker in _ANSWER_STOP_MARKERS:
        idx = s.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    s = s[:cut]
    if single_line:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        s = lines[0] if lines else s.strip()
    # Strip any leftover answer-wrapper tags the model emits unpaired or as a
    # prefix: angle tags ("<A>") and short bracket tags ("[OUT]", "[ANS]",
    # "[RESP]", "[ANSWER]", "[A]", ...). Generic by design so we catch arbitrary
    # wrapper variants, not an enumerated list. Normal short answers do not
    # contain "<word>" or "[WORD]" tokens, so this is safe.
    s = re.sub(r"</?[A-Za-z]{1,10}>", "", s)
    s = re.sub(r"\[/?[A-Za-z][A-Za-z0-9]{0,11}\]", "", s)
    return s.strip()


def sample_record(
    record: dict[str, Any],
    model_fn: ModelFn,
    entailment_fn: NliFn | None = None,
) -> dict[str, Any]:
    """Fill the sampling fields of a pre-sampling record.

    Calls *model_fn* with the record's prompt and requested sample count,
    normalizes the returned answers using the dataset-faithful normalizer
    (:func:`semantic_entropy.eval_normalize.normalize_answers_for_dataset`),
    and assigns cluster IDs via bidirectional-entailment clustering
    (:func:`semantic_entropy.clustering.nli_cluster`).
    The input record is not modified; a new dict is returned.

    The fields populated are:

    * ``sampled_answers`` — raw strings returned by *model_fn*.
    * ``normalized_answers`` — canonicalized forms via the dataset normalizer.
    * ``semantic_clusters`` — integer cluster IDs, one per sample.
    * ``cluster_representatives`` — one representative string per cluster.

    The fields ``scores``, ``correctness_label``, and all run-metadata
    fields are passed through unchanged.

    Args:
        record: A pre-sampling record dict as produced by
            :func:`make_presampling_record`.  Must contain ``"prompt"`` and
            ``"decoding"`` with a positive ``"num_samples"`` key.
        model_fn: Callable ``(prompt, num_samples) -> list[str]``.  Must
            return a list whose length equals *num_samples*; a
            :class:`ValueError` is raised otherwise.
        entailment_fn: NLI function used for semantic clustering; see
            :data:`semantic_entropy.clustering.NliFn`. Defaults to
            :func:`semantic_entropy.clustering.exact_match_entailment_fn`
            (string equality), which reproduces exact-match clustering and
            requires no ML model. Pass a real DeBERTa-v3 cross-encoder
            wrapper here for Phase 1/2 runs.

    Returns:
        A shallow copy of *record* with the four sampling fields populated.

    Raises:
        ValueError: If *model_fn* returns the wrong number of answers.
        KeyError: If *record* is missing ``"prompt"`` or
            ``"decoding.num_samples"``.
    """
    prompt: str = record["prompt"]
    num_samples: int = record["decoding"]["num_samples"]

    sampled, sequence_logprobs = _split_model_output(model_fn(prompt, num_samples))
    if len(sampled) != num_samples:
        raise ValueError(
            f"model_fn returned {len(sampled)} answers; expected {num_samples}"
        )
    if sequence_logprobs is not None and len(sequence_logprobs) != num_samples:
        raise ValueError(
            f"model_fn returned {len(sequence_logprobs)} logprobs; expected {num_samples}"
        )

    dataset: str = record.get("dataset", "")
    # Defensive extraction: strip any chat-template echo the generator emitted
    # past its answer. With stop sequences in the vLLM sampler this is usually a
    # no-op; it guarantees clustering, entropy, and the label see only the
    # answer. Biography is paragraph-length, so keep multi-line there.
    sampled = [_extract_answer(a, single_line=(dataset != "bio")) for a in sampled]
    normalized = normalize_answers_for_dataset(sampled, dataset)
    ent_fn = entailment_fn if entailment_fn is not None else exact_match_entailment_fn
    cluster_ids, representatives = nli_cluster(normalized, ent_fn)

    out = {
        **record,
        "sampled_answers": list(sampled),
        "normalized_answers": normalized,
        "semantic_clusters": cluster_ids,
        "cluster_representatives": representatives,
    }
    if sequence_logprobs is not None:
        out["sequence_logprobs"] = [float(lp) for lp in sequence_logprobs]
    return out


def _split_model_output(
    result: Any,
) -> tuple[list[str], list[float] | None]:
    """Accept either ``list[str]`` or ``(answers, sequence_logprobs)``.

    A model_fn may optionally return per-sequence length-normalized
    log-probabilities alongside its answers as a 2-tuple. Plain answer lists
    remain fully supported (logprobs treated as absent), so existing adapters
    are unaffected.
    """
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], (list, tuple))
        and isinstance(result[1], (list, tuple))
    ):
        answers, logprobs = result
        return list(answers), list(logprobs)
    return list(result), None


def evaluate_correctness(record: dict[str, Any]) -> dict[str, Any]:
    """Set ``correctness_label`` based on normalized answer/reference overlap.

    A record is considered *correct* if at least one normalized sampled
    answer is an exact string match for at least one normalized reference
    answer. Both sets of answers are normalized with
    :func:`semantic_entropy.normalization.normalize_answers` before
    comparison, so surface differences in casing and whitespace are ignored.

    This is the surface-form exact-match correctness criterion used by the
    Phase 1 evaluation and is appropriate for both TriviaQA (which accepts
    multiple aliases) and SVAMP (which has a single numeric reference). It
    will under-score cases where a semantically correct paraphrase does not
    normalize to the reference form; that limitation is addressed in
    Stage 3 by the NLI-based clustering and a richer normalizer.

    Args:
        record: A record dict that already has ``normalized_answers`` and
            ``reference_answers`` populated.

    Returns:
        A shallow copy of *record* with ``correctness_label`` set to a
        ``bool``.
    """
    dataset: str = record.get("dataset", "")
    normalized_refs = set(normalize_answers_for_dataset(record["reference_answers"], dataset))
    correct = any(ans in normalized_refs for ans in record["normalized_answers"])
    return {**record, "correctness_label": correct}


def evaluate_correctness_paper(
    record: dict[str, Any], model_fn: "ModelFn | None" = None
) -> dict[str, Any]:
    """Set ``correctness_label`` from one recorded point-estimate answer.

    Accuracy is assessed on one high-likelihood answer and counted correct when
    the SQuAD token-F1 against any reference exceeds 0.5 (exact match for
    SVAMP). The selected answer is, in priority order, the canonical SEP greedy
    response, a dedicated low-temperature generation exposed by the vLLM
    adapter, or the modal sampled answer. These alternatives are not claimed to
    be interchangeable or an exact reproduction of one external protocol.

    The chosen answer is stored under ``most_likely_answer`` for provenance.
    """
    from semantic_entropy.accuracy import best_answer_correct, modal_answer

    dataset: str = record.get("dataset", "")
    sep_greedy_answer = record.get("sep_greedy_answer")
    best_fn = getattr(model_fn, "best_answer", None)
    if isinstance(sep_greedy_answer, str):
        # Genuine SEP already generated the paper's high-likelihood response;
        # reuse that exact response for accuracy instead of falling back to a
        # modal high-temperature sample through the CLI's stub placeholder.
        best = sep_greedy_answer
    elif best_fn is not None:
        best = str(best_fn(record["prompt"]))
    else:
        best = modal_answer(record["sampled_answers"])
    correct = best_answer_correct(best, record["reference_answers"], dataset)
    return {**record, "most_likely_answer": best, "correctness_label": correct}


#: Type alias for a sentence-embedding function used by Kernel Language
#: Entropy.  Receives a sequence of strings and returns an ``(n, d)``
#: embedding matrix as either a ``numpy.ndarray`` or a ``list[list[float]]``.
#: See :func:`semantic_entropy.models.make_embedding_fn` for a concrete
#: factory.
EmbeddingFn = Callable[[Sequence[str]], Any]

#: Type alias for a generation-model function that *also* returns hidden
#: states.  See :data:`semantic_entropy.models.ModelFnWithStates` for the
#: full description.  Re-exported here so callers of :func:`run_pipeline`
#: have a single place to import the pipeline contract.
ModelFnWithStates = Callable[[str, int], Any]


def run_pipeline(
    items: Iterable[PromptItem],
    config: RunConfig,
    model_fn: ModelFn,
    output_path: Path,
    *,
    entailment_fn: NliFn | None = None,
    embedding_fn: EmbeddingFn | None = None,
    kle_kernel: str = "rbf",
    model_fn_with_states: ModelFnWithStates | None = None,
    sep_probe: Any | None = None,
) -> int:
    """Run the full Phase 1 pipeline and write schema-valid scored JSONL.

    For each :class:`~semantic_entropy.datasets.PromptItem` in *items* the
    pipeline executes in order:

    1. :func:`make_presampling_record` — populate run metadata and prompt fields.
    2. :func:`sample_record` — call *model_fn*, normalize, cluster via *entailment_fn*.
    3. :func:`evaluate_correctness` — set ``correctness_label``.
    4. :func:`semantic_entropy.scoring.score_record` — compute entropy scores.
    5. Optional: when *embedding_fn* is provided, compute
       :func:`semantic_entropy.kle.compute_kle` over the sampled answers
       and merge it into the ``scores`` mapping under the key ``"kle"``.
    6. :func:`semantic_entropy.schema.validate_record` and
       :func:`semantic_entropy.cluster_check.check_cluster_consistency` —
       validate the fully-scored record.
    7. Write one JSON line (keys sorted) to *output_path*.

    Args:
        items: Source prompts to process.
        config: Run-level metadata applied to every record.
        model_fn: Model sampling function; see :data:`ModelFn`.
        output_path: Destination JSONL file. Created or overwritten.
        entailment_fn: NLI function for semantic clustering (keyword-only).
            Defaults to :func:`~semantic_entropy.clustering.exact_match_entailment_fn`.
            Pass a real NLI wrapper here for Phase 1/2 production runs.
        embedding_fn: Optional sentence-embedding function used to add a
            Kernel Language Entropy score to every record. When ``None``
            (default) no KLE score is computed and the legacy scores dict
            is written unchanged.  See :data:`EmbeddingFn` and
            :func:`semantic_entropy.models.make_embedding_fn`.
        kle_kernel: Kernel name forwarded to
            :func:`semantic_entropy.kle.compute_kle`. Ignored when
            *embedding_fn* is ``None``.
        model_fn_with_states: Alternative sampler that returns both entropy
            samples and a separate greedy SEP feature, or the legacy tuple of
            answers and per-sample hidden states. When provided, this replaces
            *model_fn* for the sampling step.
        sep_probe: Optional trained :class:`semantic_entropy.probes.SEPProbe`.
            Only a semantic-entropy-target artifact is accepted. It is
            applied to the separate greedy ``sep_hidden_state`` and written
            into ``scores["probe_uncertainty"]``.

    Returns:
        The number of records successfully written.

    Raises:
        ValueError: If *model_fn* returns the wrong number of answers for
            any prompt.
        SchemaError: If a fully-scored record fails schema or cluster
            consistency validation, indicating a pipeline bug.
        OSError: If *output_path* cannot be opened for writing.
    """
    # Defer the KLE import so the pipeline can run without numpy when
    # KLE scoring is disabled (which is the default Phase 1 path).
    compute_kle = None
    if embedding_fn is not None:
        from semantic_entropy.kle import compute_kle as _compute_kle
        compute_kle = _compute_kle

    score_probe_for_record = None
    probe_provenance: dict[str, Any] | None = None
    presampling_records: list[dict[str, Any]] | None = None
    if sep_probe is not None:
        from semantic_entropy.probes import (
            canonical_sep_provenance,
            validate_canonical_sep_scoring_prompt_content,
        )
        from semantic_entropy.probes import (
            score_probe_for_record as _score_probe_for_record,
        )
        if sep_probe.effective_probe_kind != "semantic_entropy_probe":
            raise ValueError(
                "--sep-probe requires a semantic-entropy-target artifact; "
                "a correctness-target artifact is an accuracy probe"
            )
        if model_fn_with_states is None:
            raise ValueError(
                "SEP scoring requires model_fn_with_states to collect the "
                "audited greedy feature"
            )
        score_probe_for_record = _score_probe_for_record
        probe_provenance = canonical_sep_provenance(sep_probe)
        # The probe artifact names one exact held-out split. Audit the complete
        # prompt content before any inference or output is written.
        presampling_records = [
            make_presampling_record(item, config) for item in list(items)
        ]
        validate_canonical_sep_scoring_prompt_content(
            presampling_records, sep_probe
        )

    records_written = 0
    record_source: Iterable[dict[str, Any]]
    if presampling_records is not None:
        record_source = presampling_records
    else:
        record_source = (make_presampling_record(item, config) for item in items)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in record_source:

            if model_fn_with_states is not None:
                # Keep stochastic entropy samples separate from the single
                # greedy-response probe feature.
                hidden_record = _sample_with_states(
                    record, model_fn_with_states, entailment_fn,
                )
                record = hidden_record
            else:
                record = sample_record(record, model_fn, entailment_fn)

            # For bio task, correctness_label is set to a placeholder
            # (False) here and filled posthoc by score_longform_correctness.py
            # after a separate LLM-judge sbatch.  For QA tasks, use the
            # standard string-overlap heuristic.
            if config.task == "bio":
                record = {**record, "correctness_label": False}
            elif getattr(config, "paper_accuracy", False):
                record = evaluate_correctness_paper(record, model_fn)
            else:
                record = evaluate_correctness(record)
            record["scores"] = score_record(
                record["sampled_answers"],
                record["normalized_answers"],
                record["semantic_clusters"],
                sequence_logprobs=record.get("sequence_logprobs"),
            )
            if compute_kle is not None:
                record["scores"]["kle"] = compute_kle(
                    record["sampled_answers"],
                    embedding_fn,
                    kernel=kle_kernel,
                )
            if score_probe_for_record is not None:
                probe_uncertainty = score_probe_for_record(record, sep_probe)
                if probe_uncertainty is not None:
                    record["scores"]["probe_uncertainty"] = probe_uncertainty
                    record["probe_provenance"] = dict(probe_provenance or {})
            try:
                validate_record(record)
                check_cluster_consistency(record)
            except SchemaError as exc:
                raise SchemaError(
                    f"pipeline validation failed for prompt_id={record.get('prompt_id')!r}: {exc}"
                ) from exc
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            records_written += 1
    return records_written


def _sample_with_states(
    record: dict[str, Any],
    model_fn_with_states: ModelFnWithStates,
    entailment_fn: NliFn | None,
) -> dict[str, Any]:
    """Helper: call a ModelFnWithStates and populate the sampling fields.

    Mirrors :func:`sample_record` exactly except that the model call also
    yields per-sample hidden states, which are appended to the record
    under ``"hidden_states"``.
    """
    prompt: str = record["prompt"]
    num_samples: int = record["decoding"]["num_samples"]

    result = model_fn_with_states(prompt, num_samples)
    from semantic_entropy.models import (
        CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS,
        SEPGeneration,
    )

    if isinstance(result, SEPGeneration):
        sampled = result.sampled_answers
        hidden_states = None
    else:
        sampled, hidden_states = result
    if len(sampled) != num_samples:
        raise ValueError(
            f"model_fn_with_states returned {len(sampled)} answers; "
            f"expected {num_samples}"
        )
    if hidden_states is not None and len(hidden_states) != num_samples:
        raise ValueError(
            f"model_fn_with_states returned {len(hidden_states)} hidden "
            f"state rows; expected {num_samples}"
        )

    dataset: str = record.get("dataset", "")
    sampled = [_extract_answer(a, single_line=(dataset != "bio")) for a in sampled]
    normalized = normalize_answers_for_dataset(sampled, dataset)
    ent_fn = (
        entailment_fn if entailment_fn is not None else exact_match_entailment_fn
    )
    cluster_ids, representatives = nli_cluster(normalized, ent_fn)

    out = {
        **record,
        "sampled_answers": list(sampled),
        "normalized_answers": normalized,
        "semantic_clusters": cluster_ids,
        "cluster_representatives": representatives,
    }
    if (
        isinstance(result, SEPGeneration)
        and getattr(result, "sequence_logprobs", None) is not None
    ):
        out["sequence_logprobs"] = [
            float(logprob) for logprob in result.sequence_logprobs
        ]
    if isinstance(result, SEPGeneration):
        metadata = dict(result.metadata)
        decoding = record["decoding"]
        expected = {
            "sampling_temperature": float(decoding["temperature"]),
            "sampling_top_p": float(decoding["top_p"]),
            "sampling_top_k": int(decoding.get("top_k", -1)),
            "sampling_num_responses": int(num_samples),
            "sampling_max_new_tokens": int(decoding["max_new_tokens"]),
            "base_seed": int(decoding["seed"]),
        }
        for field, expected_value in expected.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"SEP adapter metadata {field}={metadata.get(field)!r} "
                    f"does not match RunConfig value {expected_value!r}"
                )
        if metadata.get("feature_capture") == "disabled":
            # v3 non-canonical temperature collection: sampling consistency
            # was verified above, but no SEP feature exists and none of the
            # canonical-feature pins apply. The greedy answer is preserved
            # as the most-likely answer (paper accuracy rule / P(True));
            # no sep_* fields are written, so the record stays valid at any
            # sampling temperature.
            if result.hidden_state is not None:
                raise ValueError(
                    "feature_capture=disabled result must not carry a "
                    "hidden state"
                )
            out.setdefault(
                "most_likely_answer",
                _extract_answer(
                    result.greedy_answer, single_line=(dataset != "bio")
                ),
            )
            return out
        canonical = {
            "sampling_temperature": 1.0,
            "sampling_top_p": 0.9,
            "sampling_top_k": 50,
            "sampling_num_responses": 10,
            "feature_response_max_new_tokens": (
                CANONICAL_SEP_FEATURE_MAX_NEW_TOKENS
            ),
        }
        for field, expected_value in canonical.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"Kossen-style SEP requires {field}={expected_value!r}; "
                    f"got {metadata.get(field)!r}"
                )
        out.update(
            {
                "sep_greedy_answer": _extract_answer(
                    result.greedy_answer, single_line=(dataset != "bio")
                ),
                "sep_hidden_state": (
                    None
                    if result.hidden_state is None
                    else list(result.hidden_state)
                ),
                "sep_feature_metadata": metadata,
            }
        )
    else:
        out["hidden_states"] = [list(row) for row in hidden_states]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DATASET_CHOICES = ("triviaqa", "svamp", "all")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: write pre-sampling JSONL from dataset stubs.

    Reads the committed stub files for the requested dataset(s) and writes
    one pre-sampling record per prompt to *output_jsonl*. All run-metadata
    fields can be overridden via command-line flags; sensible Phase 1 smoke
    defaults apply when flags are omitted.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pre-sampling JSONL file from TriviaQA and/or SVAMP "
            "stub data. The output records have all schema fields populated "
            "except sampled_answers, normalized_answers, semantic_clusters, "
            "cluster_representatives, scores, and correctness_label, which "
            "are left empty or null for the downstream sampling harness to fill."
        )
    )
    parser.add_argument("output_jsonl", type=Path, metavar="output.jsonl")
    parser.add_argument(
        "--dataset",
        choices=_DATASET_CHOICES,
        default="all",
        help="Which stub(s) to load (default: all).",
    )
    parser.add_argument(
        "--run-id",
        default="smoke-presampling-001",
        help="Run identifier written into every record (default: smoke-presampling-001).",
    )
    parser.add_argument("--phase", default="phase1", choices=("phase1", "phase2"))
    parser.add_argument(
        "--cluster",
        default="local",
        choices=("local", "tinygpu", "alex"),
    )
    parser.add_argument("--model", default="synthetic-model")
    parser.add_argument(
        "--model-tier",
        default="small",
        choices=("small", "7b_8b", "70b_plus"),
        dest="model_tier",
    )
    parser.add_argument(
        "--entailment-backend",
        default="synthetic-preclustered",
        dest="entailment_backend",
    )
    parser.add_argument("--num-samples", type=int, default=4, dest="num_samples")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, dest="max_new_tokens"
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    config = RunConfig(
        run_id=args.run_id,
        phase=args.phase,
        cluster=args.cluster,
        model=args.model,
        model_tier=args.model_tier,
        entailment_backend=args.entailment_backend,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    try:
        items = list(_load_items(args.dataset))
    except DatasetError as error:
        parser.exit(status=1, message=f"error loading dataset: {error}\n")

    try:
        n = make_presampling_jsonl(items, config, args.output_jsonl)
    except OSError as error:
        parser.exit(status=1, message=f"error writing output: {error}\n")

    print(f"wrote {n} pre-sampling records to {args.output_jsonl}")
    return 0


def _load_items(dataset: str) -> Iterable[PromptItem]:
    if dataset in ("triviaqa", "all"):
        yield from load_triviaqa_records(TRIVIAQA_STUB_PATH)
    if dataset in ("svamp", "all"):
        yield from load_svamp_records(SVAMP_STUB_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
