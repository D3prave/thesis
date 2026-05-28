"""Prepare biography dataset for the long-form hallucination study.

Reads a topics JSON file (list of {entity, wikipedia_title, frequency_bin}
objects), fetches the Wikipedia intro paragraph for each entity via the
official MediaWiki REST summary endpoint, filters by intro length, performs
a deterministic 500/500 train/eval split, and writes JSONL files.

File layout (matches the existing data/processed pattern)::

    data/raw/factscore/topics.json         # committed, small
    data/raw/wikipedia_summaries/          # gitignored, fetched at prep time
    data/processed/bio_train.jsonl         # 500 records, committed
    data/processed/bio_eval.jsonl          # 500 records, committed
    data/processed/bio_smoke.jsonl         # 5 records, committed, for tests

Record shape (compatible with existing PromptItem / harness schema)::

    {
      "prompt_id": "bio-train-0001",
      "dataset": "bio",
      "split": "train",
      "prompt": "Tell me a bio of Marie Curie.",
      "reference_answers": ["<wikipedia intro paragraph>"],
      "metadata": {
        "entity": "Marie Curie",
        "wikipedia_title": "Marie_Curie",
        "frequency_bin": "very_frequent",
        "reference_source": "wikipedia_rest_summary",
        "reference_revision_id": 1234567890
      }
    }

Usage::

    uv run python scripts/prepare_bio.py \\
        --topics external/factscore/topics.json \\
        --out-dir data/processed \\
        --train-size 500 --eval-size 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Wikipedia REST API
# ---------------------------------------------------------------------------

WIKIPEDIA_SUMMARY_URL = (
    "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
)
USER_AGENT = "semantic-entropy-bio-study/1.0 (jakub.wisniewski.cherry@gmail.com)"

MIN_INTRO_CHARS: int = 200
MAX_INTRO_CHARS: int = 2000

_RATE_LIMIT_SLEEP: float = 0.2  # ≤ 5 req/s


def fetch_wikipedia_summary(
    title: str,
    cache_dir: Path,
    *,
    fetch_fn=None,
) -> dict | None:
    """Fetch the Wikipedia REST summary for *title*, using a disk cache.

    Args:
        title: Wikipedia page title (spaces replaced by underscores accepted).
        cache_dir: Directory where fetched JSON responses are cached.
        fetch_fn: Optional callable (url: str) -> bytes for testing.

    Returns:
        Parsed JSON dict from the REST endpoint, or ``None`` if the page
        does not exist (404) or the network fetch fails.
    """
    slug = title.replace(" ", "_")
    cache_path = cache_dir / f"{slug}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # re-fetch on corrupt cache

    url = WIKIPEDIA_SUMMARY_URL.format(title=urllib.request.quote(slug, safe=""))

    try:
        if fetch_fn is not None:
            raw = fetch_fn(url)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
        payload = json.loads(raw)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        print(f"  HTTP {exc.code} for {title}: {exc}", flush=True)
        return None
    except Exception as exc:
        print(f"  Error fetching {title}: {exc}", flush=True)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    time.sleep(_RATE_LIMIT_SLEEP)
    return payload


def extract_intro(payload: dict) -> tuple[str | None, int | None]:
    """Extract the intro paragraph and revision ID from a summary payload.

    Returns:
        ``(intro_text, revision_id)`` or ``(None, None)`` if unusable.
    """
    intro = payload.get("extract", "").strip()
    if not intro:
        return None, None
    rev_id = payload.get("revision")
    return intro, rev_id


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def make_bio_record(
    *,
    prompt_id: str,
    split: str,
    entity: str,
    wikipedia_title: str,
    frequency_bin: str,
    intro: str,
    revision_id: int | None,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "dataset": "bio",
        "split": split,
        "prompt": f"Tell me a bio of {entity}.",
        "reference_answers": [intro],
        "metadata": {
            "entity": entity,
            "wikipedia_title": wikipedia_title.replace(" ", "_"),
            "frequency_bin": frequency_bin,
            "reference_source": "wikipedia_rest_summary",
            "reference_revision_id": revision_id,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare(
    topics_path: Path,
    out_dir: Path,
    *,
    train_size: int = 500,
    eval_size: int = 500,
    seed: int = 42,
    cache_dir: Path | None = None,
    smoke_size: int = 5,
    fetch_fn=None,
) -> tuple[int, int]:
    """Run the full data-preparation pipeline.

    Returns:
        ``(n_train, n_eval)`` — number of records written to each split.
    """
    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    if not topics:
        raise ValueError(f"No topics found in {topics_path}")

    # Deduplicate by wikipedia_title (a few topics entries share the same title).
    seen_titles: set[str] = set()
    unique_topics = []
    for t in topics:
        title = t.get("wikipedia_title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_topics.append(t)

    _wiki_cache_dir = cache_dir or (topics_path.parent.parent / "raw" / "wikipedia_summaries")

    # Fetch and filter.
    usable: list[dict] = []
    total = len(unique_topics)
    print(f"Fetching Wikipedia summaries for {total} topics …", flush=True)
    for idx, topic in enumerate(unique_topics, start=1):
        entity = topic.get("entity", "")
        wiki_title = topic.get("wikipedia_title", entity)
        freq_bin = topic.get("frequency_bin", "unknown")

        payload = fetch_wikipedia_summary(wiki_title, _wiki_cache_dir, fetch_fn=fetch_fn)
        if payload is None:
            print(f"  [{idx}/{total}] SKIP (fetch failed): {entity}", flush=True)
            continue

        intro, rev_id = extract_intro(payload)
        if intro is None:
            print(f"  [{idx}/{total}] SKIP (no extract): {entity}", flush=True)
            continue

        if not (MIN_INTRO_CHARS <= len(intro) <= MAX_INTRO_CHARS):
            print(
                f"  [{idx}/{total}] SKIP (len {len(intro)}): {entity}",
                flush=True,
            )
            continue

        usable.append(
            {
                "entity": entity,
                "wikipedia_title": wiki_title,
                "frequency_bin": freq_bin,
                "intro": intro,
                "revision_id": rev_id,
            }
        )
        if idx % 50 == 0:
            print(f"  [{idx}/{total}] usable so far: {len(usable)}", flush=True)

    print(
        f"Fetched and filtered: {len(usable)} usable of {total} topics.",
        flush=True,
    )

    need = train_size + eval_size
    if len(usable) < need:
        raise ValueError(
            f"Need {need} usable topics but only got {len(usable)}. "
            "Add more topics to external/factscore/topics.json or "
            "lower --train-size / --eval-size."
        )

    # Deterministic shuffle and split.
    rng = random.Random(seed)
    rng.shuffle(usable)
    train_items = usable[:train_size]
    eval_items = usable[train_size : train_size + eval_size]
    smoke_items = eval_items[:smoke_size]  # first 5 eval items for smoke test

    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(items: list[dict], split: str, path: Path) -> int:
        with path.open("w", encoding="utf-8") as fh:
            for i, item in enumerate(items, start=1):
                prompt_id = f"bio-{split}-{i:04d}"
                record = make_bio_record(
                    prompt_id=prompt_id,
                    split=split,
                    entity=item["entity"],
                    wikipedia_title=item["wikipedia_title"],
                    frequency_bin=item["frequency_bin"],
                    intro=item["intro"],
                    revision_id=item.get("revision_id"),
                )
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return len(items)

    n_train = _write(train_items, "train", out_dir / "bio_train.jsonl")
    n_eval = _write(eval_items, "eval", out_dir / "bio_eval.jsonl")
    _write(smoke_items, "eval", out_dir / "bio_smoke.jsonl")

    print(
        f"Wrote bio_train.jsonl ({n_train}), bio_eval.jsonl ({n_eval}), "
        f"bio_smoke.jsonl ({len(smoke_items)}) → {out_dir}",
        flush=True,
    )
    return n_train, n_eval


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--topics",
        type=Path,
        default=Path("external/factscore/topics.json"),
        help="Path to the topics JSON file.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        dest="out_dir",
        help="Directory for output JSONL files.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        dest="cache_dir",
        help="Directory for cached Wikipedia summaries (default: data/raw/wikipedia_summaries).",
    )
    p.add_argument("--train-size", type=int, default=500, dest="train_size")
    p.add_argument("--eval-size", type=int, default=500, dest="eval_size")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prepare(
        topics_path=args.topics,
        out_dir=args.out_dir,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
