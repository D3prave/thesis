"""Public interface for the semantic-entropy thesis pilot.

The package implements the Phase 1 discrete semantic-entropy baseline of the
thesis: synthetic answer normalization, dataset-faithful evaluation
normalization for TriviaQA and SVAMP, structural validation of the shared
JSONL record schema, two uncertainty estimators -- surface-form entropy and
discrete semantic entropy -- a naive raw sampled-answer entropy, a
dataset-ingestion layer for TriviaQA and SVAMP stubs, exact-match clustering
(Stage 3 NLI stub), and a sampling harness that chains ingested prompts
through sampling, clustering, correctness evaluation, and scoring.
Probability-weighted semantic entropy and the Phase 2 extensions (SEP, KLE,
Semantic Energy, Semantic Volume, adaptive Bayesian SE) are out of scope and
are not re-exported here.
"""

from semantic_entropy.cluster_check import (
    check_cluster_consistency,
    check_jsonl_cluster_consistency,
)
from semantic_entropy.clustering import exact_match_cluster
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
    ModelFn,
    RunConfig,
    evaluate_correctness,
    make_presampling_jsonl,
    make_presampling_record,
    run_pipeline,
    sample_record,
)
from semantic_entropy.normalization import normalize_answer, normalize_answers
from semantic_entropy.schema import SchemaError, validate_record
from semantic_entropy.scoring import (
    discrete_semantic_entropy,
    naive_sample_entropy,
    score_record,
    surface_entropy,
)

__all__ = [
    "DatasetError",
    "KNOWN_DATASETS",
    "ModelFn",
    "PromptItem",
    "RunConfig",
    "SchemaError",
    "SVAMP_DATASET",
    "SVAMP_STUB_PATH",
    "TRIVIAQA_DATASET",
    "TRIVIAQA_STUB_PATH",
    "check_cluster_consistency",
    "check_jsonl_cluster_consistency",
    "discrete_semantic_entropy",
    "evaluate_correctness",
    "exact_match_cluster",
    "load_svamp_records",
    "load_triviaqa_records",
    "make_presampling_jsonl",
    "make_presampling_record",
    "naive_sample_entropy",
    "normalize_answer",
    "normalize_answer_for_dataset",
    "normalize_answers",
    "normalize_answers_for_dataset",
    "normalize_svamp_answer",
    "normalize_triviaqa_answer",
    "run_pipeline",
    "sample_record",
    "score_record",
    "surface_entropy",
    "validate_record",
]
