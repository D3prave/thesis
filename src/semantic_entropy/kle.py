"""Kernel Language Entropy (KLE) score for semantic uncertainty.

KLE is the soft-clustering counterpart of discrete semantic entropy
(:mod:`semantic_entropy.scoring`).  Rather than committing each sampled
generation to a hard cluster via NLI, KLE embeds every answer in a sentence
embedding space, builds a positive-semi-definite kernel matrix ``K`` over
the answer set, and returns the *von Neumann* entropy of the normalised
density matrix ``K / trace(K)``.

Reference: Nikitin, Petrov, Janowski, Janik (2024), "Kernel Language
Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic
Similarities".

For a kernel matrix ``K`` with eigenvalues ``lambda_i`` of ``K / trace(K)``,

.. math::

    H_\\mathrm{KLE} = -\\sum_i \\lambda_i \\log \\lambda_i.

Two kernels are supported:

* ``"rbf"`` — Gaussian / Radial Basis Function kernel
  ``K[i, j] = exp(-||e_i - e_j||^2 / (2 sigma^2))``.  When ``sigma`` is
  not specified, the *median heuristic* sets ``sigma^2`` to the median
  of the pairwise squared distances; this is the standard choice in the
  kernel-methods literature.
* ``"cosine"`` — affine-shifted cosine kernel
  ``K[i, j] = 0.5 (1 + cos(e_i, e_j))``.  The 0.5 (1 + .) form maps cosine
  similarities from ``[-1, 1]`` to ``[0, 1]``, which keeps the kernel PSD
  with strictly non-negative entries and yields a numerically stable
  density matrix.

The module deliberately defers the ``numpy`` import to the point of first
use so that importing :mod:`semantic_entropy` continues to require only the
standard library — matching the rest of the package.  When ``numpy`` is
unavailable, calling :func:`compute_kle` raises :class:`ImportError` with
an install hint pointing at the ``kle`` optional dependency group.

**Typical usage** in the run pipeline (see :func:`semantic_entropy.harness.run_pipeline`)::

    from semantic_entropy.models import make_embedding_fn
    from semantic_entropy.kle import compute_kle

    embed = make_embedding_fn("sentence-transformers/all-MiniLM-L6-v2")
    score = compute_kle(["paris", "Paris, France", "lyon"], embed)

KLE is content-free with respect to dataset normalisation: it operates on
the raw sampled answers (or any string list).  Pass the *sampled* answers
rather than the normalised forms to match the spirit of the original paper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    EmbeddingArray = np.ndarray
else:
    EmbeddingArray = object  # noqa: F841 (alias used in docstrings only)


#: A function that maps a list of strings to a list-of-lists embedding
#: matrix.  Each row is the embedding of the corresponding input string.
#: The matrix may also be returned as a ``numpy.ndarray``; :func:`compute_kle`
#: coerces both forms.
EmbeddingFn = Callable[[Sequence[str]], "list[list[float]] | EmbeddingArray"]


KernelName = Literal["rbf", "cosine"]


def compute_kle(
    answers: Sequence[str],
    embedding_fn: EmbeddingFn,
    *,
    kernel: KernelName = "rbf",
    sigma: float | None = None,
) -> float:
    """Return the Kernel Language Entropy of *answers* under *embedding_fn*.

    Args:
        answers: One or more sampled answer strings.  Must be non-empty.
        embedding_fn: A function returning an ``(n, d)`` embedding matrix
            (either ``numpy.ndarray`` or ``list[list[float]]``) for an
            input list of ``n`` strings.
        kernel: Kernel function used to build the similarity matrix.
            ``"rbf"`` (default) uses a Gaussian kernel with the *median
            heuristic* bandwidth when *sigma* is ``None``; ``"cosine"``
            uses the affine-shifted cosine kernel.
        sigma: Bandwidth for the RBF kernel.  ``None`` (default) selects
            ``sigma`` so that ``sigma^2`` equals the median of the
            pairwise squared distances of the embeddings; if every pair
            of embeddings is identical the bandwidth falls back to ``1.0``
            and the kernel collapses to a constant matrix (KLE = 0).
            Ignored when ``kernel="cosine"``.

    Returns:
        The von Neumann entropy of the normalised kernel matrix, in nats.
        The value is non-negative and is bounded above by ``log(n)`` where
        ``n = len(answers)``.

    Raises:
        ValueError: If *answers* is empty or *kernel* is not a supported
            kernel name.
        ImportError: If ``numpy`` is not installed.  Install via
            ``pip install -e '.[kle]'``.
    """
    if not answers:
        raise ValueError("answers must not be empty")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - covered by deps install
        raise ImportError(
            "KLE requires numpy. Install the optional kle dependencies with: "
            "pip install -e '.[kle]'"
        ) from exc

    embeddings = np.asarray(embedding_fn(list(answers)), dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(answers):
        raise ValueError(
            "embedding_fn must return a 2D array with one row per input "
            f"answer; got shape {embeddings.shape} for {len(answers)} answers"
        )

    if kernel == "rbf":
        kmat = _rbf_kernel(embeddings, sigma)
    elif kernel == "cosine":
        kmat = _cosine_kernel(embeddings)
    else:
        raise ValueError(
            f"unknown kernel: {kernel!r} (expected 'rbf' or 'cosine')"
        )

    return _von_neumann_entropy(kmat)


# ---------------------------------------------------------------------------
# Kernel constructors (numpy required at call time)
# ---------------------------------------------------------------------------


def _rbf_kernel(embeddings: "EmbeddingArray", sigma: float | None) -> "EmbeddingArray":
    """Return ``K[i, j] = exp(-||e_i - e_j||^2 / (2 sigma^2))``.

    When *sigma* is ``None`` the bandwidth is set to the median pairwise
    Euclidean distance over the upper-triangular entries with strictly
    positive distance; if no such pair exists ``sigma`` falls back to
    ``1.0`` so the kernel is well-defined.
    """
    import numpy as np

    # Pairwise squared Euclidean distances via the (a-b)^2 = a^2 + b^2 - 2ab
    # identity, then clipped to >= 0 to absorb rounding error.
    sq_norms = np.sum(embeddings ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * embeddings @ embeddings.T
    sq_dists = np.clip(sq_dists, 0.0, None)

    if sigma is None:
        # Median heuristic over the strict upper triangle.
        triu_idx = np.triu_indices_from(sq_dists, k=1)
        upper = sq_dists[triu_idx]
        positive = upper[upper > 0]
        if positive.size == 0:
            sigma_sq = 1.0  # all embeddings identical → bandwidth irrelevant
        else:
            sigma_sq = float(np.median(positive))
    else:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}")
        sigma_sq = float(sigma) ** 2

    return np.exp(-sq_dists / (2.0 * sigma_sq))


def _cosine_kernel(embeddings: "EmbeddingArray") -> "EmbeddingArray":
    """Return ``K[i, j] = 0.5 (1 + cos(e_i, e_j))``.

    The affine shift maps the cosine similarities from ``[-1, 1]`` into
    ``[0, 1]`` and guarantees a non-negative PSD kernel.  Zero-norm
    embedding rows are replaced with a small epsilon to avoid a
    division-by-zero; in that degenerate case the corresponding cosine
    values are zero, which is the natural extension of cosine similarity
    to the origin.
    """
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=1)
    # Avoid division-by-zero for empty embeddings.
    safe_norms = np.where(norms > 0, norms, 1.0)
    normed = embeddings / safe_norms[:, None]
    # Rows that were zero must contribute zero similarity.
    normed = np.where(norms[:, None] > 0, normed, 0.0)
    cos_sim = normed @ normed.T
    # Numerical safety: cosine similarities can stray slightly outside
    # [-1, 1] for very high-dimensional embeddings.
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 0.5 * (1.0 + cos_sim)


# ---------------------------------------------------------------------------
# Von Neumann entropy of a normalised kernel matrix
# ---------------------------------------------------------------------------


def _von_neumann_entropy(kmat: "EmbeddingArray") -> float:
    """Return the von Neumann entropy of ``kmat / trace(kmat)`` in nats.

    The kernel is symmetrised and clipped to non-negative eigenvalues
    before the entropy is computed; the symmetrisation removes the tiny
    asymmetries introduced by floating-point matmul, and the clipping
    absorbs the negative eigenvalue noise that can appear for kernels
    that are PSD only up to rounding.
    """
    import numpy as np

    trace = float(np.trace(kmat))
    if trace <= 0:
        # Pathological case: degenerate kernel.  Returning 0 matches the
        # convention that entropy of a delta distribution is zero.
        return 0.0

    rho = (kmat + kmat.T) / (2.0 * trace)
    eigvals = np.linalg.eigvalsh(rho)
    # Clip negative noise and tiny positive values that would log to -inf.
    eigvals = np.clip(eigvals, 1e-12, None)
    # Re-normalise so that the eigenvalues sum to exactly 1 after clipping.
    eigvals = eigvals / float(np.sum(eigvals))
    # Exclude effectively-zero eigenvalues from the sum to avoid the
    # 0 * log(0) limit producing NaN under some BLAS implementations.
    nonzero = eigvals[eigvals > 1e-12]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))
