"""Paper-faithful accuracy assessment (Farquhar et al. 2024).

The paper assesses model accuracy on a *single* answer — a low-temperature
("most likely") generation — rather than on the diversity of the
high-temperature samples used for entropy. For short-phrase QA it counts the
answer correct when the SQuAD token-level F1 against any reference exceeds 0.5;
for the arithmetic SVAMP task an exact (normalized) match is the natural rule.

This module provides:

* :func:`squad_f1` — token-level F1 between a prediction and a reference,
  after dataset-faithful normalization.
* :func:`best_answer_correct` — the paper's correctness rule for the
  most-likely answer (F1 > 0.5 for free-text QA; exact match for SVAMP).

The harness uses these (opt-in via ``RunConfig.paper_accuracy``) to set
``correctness_label`` from the most-likely answer, instead of the repo's
default "any of the M samples matches a reference" rule. Both are kept; the
default preserves historical behaviour, the opt-in reproduces the paper.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from semantic_entropy.eval_normalize import (
    SVAMP_DATASET,
    normalize_answer_for_dataset,
)

#: Datasets graded by exact (normalized) match rather than token F1. SVAMP is
#: arithmetic — a single number — so token-overlap F1 is inappropriate.
EXACT_MATCH_DATASETS = {SVAMP_DATASET}

#: SQuAD-style threshold above which a free-text answer counts as correct.
F1_THRESHOLD = 0.5


def _tokens(text: str) -> list[str]:
    return text.split()


def squad_f1(prediction: str, reference: str, dataset: str) -> float:
    """Token-level F1 between *prediction* and *reference* (SQuAD metric).

    Both strings are normalized with the dataset-faithful normalizer first.
    Returns 0.0 when either side is empty after normalization (except the
    degenerate case where both are empty, which scores 1.0, matching the SQuAD
    convention for empty-vs-empty).
    """
    pred_tokens = _tokens(normalize_answer_for_dataset(prediction, dataset))
    ref_tokens = _tokens(normalize_answer_for_dataset(reference, dataset))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_answer_correct(
    best_answer: str,
    reference_answers: Sequence[str],
    dataset: str,
    *,
    f1_threshold: float = F1_THRESHOLD,
) -> bool:
    """Return whether the most-likely answer is correct (paper's rule).

    * SVAMP (and any dataset in :data:`EXACT_MATCH_DATASETS`): exact match after
      normalization against any reference.
    * Otherwise (TriviaQA, NQ-Open, …): SQuAD token F1 against the best-matching
      reference exceeds ``f1_threshold`` (0.5 in the paper).
    """
    if not reference_answers:
        return False
    if dataset in EXACT_MATCH_DATASETS:
        norm_best = normalize_answer_for_dataset(best_answer, dataset)
        return any(
            norm_best == normalize_answer_for_dataset(ref, dataset)
            for ref in reference_answers
        )
    best_f1 = max(squad_f1(best_answer, ref, dataset) for ref in reference_answers)
    # Their utils.get_metric compares `results['f1'] >= 50.0` on a 0-100 scale,
    # so the boundary is inclusive. Using a strict `>` disagrees only on an
    # exact tie, but a tie is not rare for short answers where a single token
    # decides the score.
    return best_f1 >= f1_threshold


def modal_answer(sampled_answers: Sequence[str]) -> str:
    """Return the most frequent sampled answer (a point estimate of the mode).

    Used as the most-likely answer when no dedicated low-temperature generation
    is available. Ties are broken by first occurrence.
    """
    if not sampled_answers:
        raise ValueError("sampled_answers must not be empty")
    counts = Counter(sampled_answers)
    return max(sampled_answers, key=lambda a: (counts[a], -list(sampled_answers).index(a)))
