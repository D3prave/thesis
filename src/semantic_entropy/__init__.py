"""Public interface for the semantic-entropy thesis pilot.

The package implements the Phase 1 discrete semantic-entropy baseline of the
thesis: synthetic answer normalization, dataset-faithful evaluation
normalization for TriviaQA and SVAMP, structural validation of the shared
JSONL record schema, two uncertainty estimators -- surface-form entropy and
discrete semantic entropy -- a naive raw sampled-answer entropy, a
dataset-ingestion layer for TriviaQA and SVAMP stubs, bidirectional NLI
clustering via union-find (``nli_cluster``), exact-match clustering and
entailment stub (Stage 3 baseline), model adapter interface with synthetic
stub and deferred HuggingFace/vLLM/NLI factories, a run-pipeline CLI for HPC
job dispatch, and a sampling harness that chains ingested prompts through
sampling, clustering, correctness evaluation, and scoring.
Probability-weighted semantic entropy and the Phase 2 extensions (SEP, KLE,
Semantic Energy, Semantic Volume, adaptive Bayesian SE) are out of scope and
are not re-exported here.
"""

from semantic_entropy.cluster_check import (
    check_cluster_consistency,
    check_jsonl_cluster_consistency,
)
from semantic_entropy.clustering import (
    NLI_CONTRADICTION,
    NLI_ENTAILMENT,
    NLI_NEUTRAL,
    NliFn,
    exact_match_cluster,
    exact_match_entailment_fn,
    nli_cluster,
)
from semantic_entropy.datasets import (
    DatasetError,
    PromptItem,
    SVAMP_STUB_PATH,
    TRIVIAQA_STUB_PATH,
    load_svamp_records,
    load_triviaqa_records,
)
from semantic_entropy.eval_normalize import (
    KNOWN_DATASETS,
    SVAMP_DATASET,
    TRIVIAQA_DATASET,
    normalize_answer_for_dataset,
    normalize_answers_for_dataset,
    normalize_svamp_answer,
    normalize_triviaqa_answer,
)
from semantic_entropy.harness import (
    EmbeddingFn,
    ModelFn,
    RunConfig,
    evaluate_correctness,
    make_presampling_jsonl,
    make_presampling_record,
    run_pipeline,
    sample_record,
)
from semantic_entropy.kle import compute_kle
from semantic_entropy.models import (
    ModelFnWithStates,
    SyntheticModel,
    make_embedding_fn,
    make_hf_model,
    make_hf_model_with_states,
    make_nli_fn,
    make_vllm_model,
)
from semantic_entropy.probes import (
    SEPProbe,
    score_probe,
    score_probe_for_record,
    train_probe,
)
from semantic_entropy.pipeline_cli import load_callable
from semantic_entropy.normalization import normalize_answer, normalize_answers
try:
    from semantic_entropy.plotting_extensions import (
        generate_all_plots,
        plot_accuracy_vs_auroc,
        plot_method_comparison_by_dataset,
        plot_phase_comparison,
    )
except ImportError:
    # matplotlib is an optional dependency; plotting functions unavailable
    # when it is not installed.  Install with: uv pip install matplotlib
    generate_all_plots = None  # type: ignore[assignment]
    plot_accuracy_vs_auroc = None  # type: ignore[assignment]
    plot_method_comparison_by_dataset = None  # type: ignore[assignment]
    plot_phase_comparison = None  # type: ignore[assignment]
from semantic_entropy.result_aggregator import (
    MetricRow,
    aggregate_results,
    scan_results_directory,
    write_metric_summary_csv,
    write_metric_summary_json,
)
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import (
    discrete_semantic_entropy,
    naive_sample_entropy,
    score_record,
    surface_entropy,
)

__all__ = [
    "DatasetError",
    "EmbeddingFn",
    "KNOWN_DATASETS",
    "MetricRow",
    "ModelFn",
    "ModelFnWithStates",
    "PromptItem",
    "RunConfig",
    "SchemaError",
    "SEPProbe",
    "SyntheticModel",
    "SVAMP_DATASET",
    "SVAMP_STUB_PATH",
    "TRIVIAQA_DATASET",
    "TRIVIAQA_STUB_PATH",
    "NLI_CONTRADICTION",
    "NLI_ENTAILMENT",
    "NLI_NEUTRAL",
    "NliFn",
    "aggregate_results",
    "check_cluster_consistency",
    "check_jsonl_cluster_consistency",
    "compute_kle",
    "discrete_semantic_entropy",
    "evaluate_correctness",
    "exact_match_cluster",
    "exact_match_entailment_fn",
    "generate_all_plots",
    "load_callable",
    "load_svamp_records",
    "load_triviaqa_records",
    "make_embedding_fn",
    "make_hf_model",
    "make_hf_model_with_states",
    "make_nli_fn",
    "make_presampling_jsonl",
    "make_presampling_record",
    "make_vllm_model",
    "naive_sample_entropy",
    "nli_cluster",
    "normalize_answer",
    "normalize_answer_for_dataset",
    "normalize_answers",
    "normalize_answers_for_dataset",
    "normalize_svamp_answer",
    "normalize_triviaqa_answer",
    "plot_accuracy_vs_auroc",
    "plot_method_comparison_by_dataset",
    "plot_phase_comparison",
    "run_pipeline",
    "sample_record",
    "scan_results_directory",
    "score_probe",
    "score_probe_for_record",
    "score_record",
    "surface_entropy",
    "train_probe",
    "validate_record",
    "write_metric_summary_csv",
    "write_metric_summary_json",
]
