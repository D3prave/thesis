"""Content-addressed cache for entailment / judge decisions.

The LLM-judge clustering pass is the most expensive step in the grid: it asks a
72B model about every ordered pair of generations. Re-running it (e.g. across
seeds, or after an unrelated change) repeats identical questions. Inspired by
the authors' implementation (jlko/semantic_uncertainty), which md5-hashes each
entailment prompt and pickles the result, this cache keys decisions by a hash of
``(model, premise, hypothesis, mode)`` and persists them as JSON, so a rerun
over the same generations reuses prior labels instead of re-querying the model.

Usage::

    cache = EntailmentCache.load(path)            # path may not exist yet
    label = cache.get(model, premise, hypothesis, mode)
    if label is None:
        label = expensive_model_call(...)
        cache.set(model, premise, hypothesis, label, mode)
    cache.save()                                  # atomic write

The cache is content-addressed, so it is safe to share one file across runs and
seeds — identical (model, premise, hypothesis, mode) tuples map to the same key.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class EntailmentCache:
    """A persistent, content-addressed map from entailment queries to labels."""

    def __init__(self, path: Path | str | None = None, data: dict[str, str] | None = None):
        self.path = Path(path) if path is not None else None
        self._data: dict[str, str] = dict(data or {})
        self._dirty = False
        self.hits = 0
        self.misses = 0

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None) -> "EntailmentCache":
        """Load a cache from ``path``; an absent or empty path is an empty cache."""
        if path is None:
            return cls(None)
        p = Path(path)
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                return cls(p, payload.get("entries", {}))
            except (json.JSONDecodeError, OSError):
                # Corrupt cache: start fresh rather than failing the job.
                return cls(p)
        return cls(p)

    # -- keys ----------------------------------------------------------------

    @staticmethod
    def make_key(model: str, premise: str, hypothesis: str, mode: str = "entailment") -> str:
        digest = hashlib.md5()
        for part in (model, mode, premise, hypothesis):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    # -- access --------------------------------------------------------------

    def get(self, model: str, premise: str, hypothesis: str, mode: str = "entailment") -> str | None:
        label = self._data.get(self.make_key(model, premise, hypothesis, mode))
        if label is None:
            self.misses += 1
        else:
            self.hits += 1
        return label

    def set(self, model: str, premise: str, hypothesis: str, label: str, mode: str = "entailment") -> None:
        self._data[self.make_key(model, premise, hypothesis, mode)] = label
        self._dirty = True

    def __len__(self) -> int:
        return len(self._data)

    # -- persistence ---------------------------------------------------------

    def save(self, force: bool = False) -> None:
        """Atomically write the cache to ``self.path`` (no-op without a path)."""
        if self.path is None or (not self._dirty and not force):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"version": 1, "entries": self._data}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.path)
            self._dirty = False
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
