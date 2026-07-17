"""Provenance contracts for long-form factuality labels.

The published FactualBio annotations and locally generated LLM-judge labels
answer related but non-interchangeable questions. These helpers keep that
distinction machine-readable so an automated reference-consistency judgment
cannot silently masquerade as a published human factuality annotation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

AUTOMATED_LABEL_SOURCE = "automated_llm_reference_judge"
AUTOMATED_LABEL_SCOPE = "exploratory_proxy"
PUBLISHED_HUMAN_LABEL_SOURCE = "published_factualbio_human"
PUBLISHED_HUMAN_LABEL_SCOPE = "primary_human_annotation"
PUBLISHED_FACTUALBIO_REPOSITORY = "https://github.com/jlko/long_hallucinations"
PUBLISHED_FACTUALBIO_RELEASE = "v.1.0.0"
PUBLISHED_FACTUALBIO_COMMIT = "957b806033d0b769db34beb7ce34316bd498f373"
PUBLISHED_FACTUALBIO_FILE = "data.py"
PUBLISHED_FACTUALBIO_SHA256 = (
    "b2ae5bc7fb86bc60c2c8ec189a510edb1d553ae573b9157a8832056ecc81f543"
)


def claim_identity_sha256(claim_id: Any, proposition: Any) -> str:
    """Return a stable identity hash binding a label to one exact claim."""

    claim_id_text = str(claim_id).strip()
    proposition_text = str(proposition).strip()
    if not claim_id_text:
        raise ValueError("claim_id must not be empty")
    if not proposition_text:
        raise ValueError("proposition must not be empty")
    payload = json.dumps(
        [claim_id_text, proposition_text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def automated_label_provenance(
    *,
    judge_model: str,
    target: str,
    representative_policy: str | None = None,
) -> dict[str, str]:
    """Return provenance for a locally generated reference-judge label."""

    judge_model = judge_model.strip()
    target = target.strip()
    if not judge_model:
        raise ValueError("judge_model must not be empty")
    if not target:
        raise ValueError("target must not be empty")
    provenance = {
        "source": AUTOMATED_LABEL_SOURCE,
        "scope": AUTOMATED_LABEL_SCOPE,
        "judge_model": judge_model,
        "target": target,
    }
    if representative_policy:
        provenance["representative_policy"] = representative_policy
    return provenance


def published_human_label_provenance() -> dict[str, str]:
    """Return provenance for labels shipped with the published FactualBio data."""

    return {
        "source": PUBLISHED_HUMAN_LABEL_SOURCE,
        "scope": PUBLISHED_HUMAN_LABEL_SCOPE,
        "target": "published_claim_factuality",
        "source_repository": PUBLISHED_FACTUALBIO_REPOSITORY,
        "source_release": PUBLISHED_FACTUALBIO_RELEASE,
        "source_commit": PUBLISHED_FACTUALBIO_COMMIT,
        "source_file": PUBLISHED_FACTUALBIO_FILE,
        "source_sha256": PUBLISHED_FACTUALBIO_SHA256,
        "annotation_unit": "claim",
        "false_label_mapping": "Major False or Minor False",
    }


def validate_label_provenance(value: Any) -> dict[str, Any]:
    """Validate and copy one supported label-provenance object."""

    if not isinstance(value, Mapping):
        raise ValueError("correctness_label_provenance must be an object")
    provenance = dict(value)
    source = provenance.get("source")
    scope = provenance.get("scope")
    expected_scopes = {
        AUTOMATED_LABEL_SOURCE: AUTOMATED_LABEL_SCOPE,
        PUBLISHED_HUMAN_LABEL_SOURCE: PUBLISHED_HUMAN_LABEL_SCOPE,
    }
    if source not in expected_scopes:
        raise ValueError(f"unsupported correctness label source: {source!r}")
    expected_scope = expected_scopes[source]
    if scope != expected_scope:
        raise ValueError(
            f"label source {source!r} requires scope {expected_scope!r}, got {scope!r}"
        )
    if not isinstance(provenance.get("target"), str) or not provenance["target"].strip():
        raise ValueError("correctness label provenance must name a non-empty target")
    if source == AUTOMATED_LABEL_SOURCE:
        judge_model = provenance.get("judge_model")
        if not isinstance(judge_model, str) or not judge_model.strip():
            raise ValueError("automated label provenance must name judge_model")
    else:
        required = {
            "target": "published_claim_factuality",
            "source_repository": PUBLISHED_FACTUALBIO_REPOSITORY,
            "source_release": PUBLISHED_FACTUALBIO_RELEASE,
            "source_commit": PUBLISHED_FACTUALBIO_COMMIT,
            "source_file": PUBLISHED_FACTUALBIO_FILE,
            "source_sha256": PUBLISHED_FACTUALBIO_SHA256,
            "annotation_unit": "claim",
            "false_label_mapping": "Major False or Minor False",
        }
        for field, expected in required.items():
            if provenance.get(field) != expected:
                raise ValueError(
                    f"published FactualBio provenance requires {field}={expected!r}"
                )
    return provenance
