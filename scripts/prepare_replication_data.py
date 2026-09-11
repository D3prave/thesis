#!/usr/bin/env python3
"""Reproduce the datasets and the exact question selection of Farquhar et al.

Their `data/data_utils.py::load_ds` builds the splits and
`generate_answers.py` then draws from them. The draws share one RNG stream, so
the *order* of the draws decides which questions are evaluated:

    random.seed(args.random_seed)                       # 10
    answerable_indices, unanswerable_indices = split_dataset(train_dataset)
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)          # 5
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))
    p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)    # 20
    remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
    for dataset_split in ['train', 'validation']:       # train first!
        ...
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))

Reproducing this matters. The previous preparation took the first 500 rows of
each split, and rows 0-499 turned out to be measurably harder than rows 500-999
(TriviaQA: all nine cells, 0.082-0.114 lower accuracy), so a prefix slice is not
a valid sample of these files.

`list(set(...))` is used deliberately: their code builds `remaining_answerable`
that way, and since the elements are ints -- for which `hash(i) == i` -- the
resulting order is deterministic. Reproducing the same construction reproduces
the same order, and therefore the same sample.

A dataset revision can be supplied and is recorded. The retained collection
manifests have no Hub revision ID; they preserve source fingerprints and the
datasets version. Reuse the prepared inputs for exact question selection.

This script was initially untracked during collection; it is now retained
with the source for inspection.

Usage:
    python scripts/prepare_replication_data.py --dataset trivia_qa
    python scripts/prepare_replication_data.py --all --out-root data/replication
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# Their defaults, from utils.get_parser.
RANDOM_SEED = 10
NUM_FEW_SHOT = 5
P_TRUE_NUM_FEWSHOT = 20
NUM_SAMPLES = 400

DATASETS = ("trivia_qa", "nq", "svamp")


def md5hash(text: str) -> str:
    """Their id for NQ-Open: md5 of the question as a decimal string."""
    return str(int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16))


# Their code uses the bare id `nq_open`, which current `datasets` releases
# reject: repository ids must be namespaced. This is the canonical location of
# the same dataset. The other two ids they use are already namespaced.
HF_IDS = {
    "nq": "google-research-datasets/nq_open",
    "svamp": "ChilleD/SVAMP",
    "trivia_qa": "TimoImhof/TriviaQA-in-SQuAD-format",
}

#: Filled in by load_ds so the manifest can record exactly what was read.
LAST_FINGERPRINTS: dict[str, str] = {}


def load_ds(dataset_name: str, seed: int, revision: str | None):
    """Port of their data_utils.load_ds for the three datasets in scope."""
    import datasets

    kwargs = {"revision": revision} if revision else {}
    LAST_FINGERPRINTS.clear()

    def note(split_name, split):
        fingerprint = getattr(split, "_fingerprint", None)
        if fingerprint:
            LAST_FINGERPRINTS[split_name] = str(fingerprint)

    if dataset_name == "svamp":
        dataset = datasets.load_dataset(HF_IDS["svamp"], **kwargs)
        note("train", dataset["train"])
        note("test", dataset["test"])
        train_dataset, validation_dataset = dataset["train"], dataset["test"]

        def reformat(x):
            return {
                "question": x["Question"],
                "context": x["Body"],
                "type": x["Type"],
                "equation": x["Equation"],
                "id": x["ID"],
                "answers": {"text": [str(x["Answer"])]},
            }

        return [reformat(d) for d in train_dataset], [
            reformat(d) for d in validation_dataset
        ]

    if dataset_name == "nq":
        dataset = datasets.load_dataset(HF_IDS["nq"], **kwargs)
        train_dataset, validation_dataset = dataset["train"], dataset["validation"]
        note("train", train_dataset)
        note("validation", validation_dataset)

        def reformat(x):
            return {
                # Their loader appends the question mark.
                "question": x["question"] + "?",
                "answers": {"text": x["answer"]},
                "context": "",
                "id": md5hash(str(x["question"])),
            }

        return [reformat(d) for d in train_dataset], [
            reformat(d) for d in validation_dataset
        ]

    if dataset_name == "trivia_qa":
        dataset = datasets.load_dataset(HF_IDS["trivia_qa"], **kwargs)["unmodified"]
        note("unmodified", dataset)
        # The split is part of the protocol, not a convenience: validation is
        # the 20% test side of a seeded split of a third-party conversion.
        #
        # Caveat worth stating in the thesis: reproducing *their* split needs
        # both the same dataset build and the same `datasets` version, since the
        # shuffle that backs train_test_split is an implementation detail. We
        # record the fingerprint and library version so at least this run is
        # reproducible.
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        note("split_train", dataset["train"])
        note("split_test", dataset["test"])
        return dataset["train"], dataset["test"]

    raise ValueError(f"unsupported dataset {dataset_name!r}")


def split_dataset(dataset):
    """Their utils.split_dataset: answerable means at least one reference."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable = [i for i, ex in enumerate(dataset) if clen(ex) == 0]
    assert set(answerable) | set(unanswerable) == set(range(len(dataset)))
    return answerable, unanswerable


def as_record(example, index: int, dataset_name: str) -> dict:
    """Write the on-disk shape `semantic_entropy.datasets` expects.

    Their in-memory examples are `{question, context, answers: {text: [...]}}`;
    the loaders here want the original dataset field names, so the two have to
    be bridged rather than assumed compatible.
    """
    answers = [str(a) for a in example["answers"]["text"]]
    identifier = str(example.get("id", index))

    if dataset_name == "svamp":
        # load_svamp_records: ID, Body, Question, numeric Answer. The prompt is
        # Body + " " + Question, which is why their loader forces
        # use_context=True for this dataset alone.
        raw = answers[0] if answers else "0"
        try:
            numeric: float | int = int(raw)
        except ValueError:
            numeric = float(raw)
        return {
            "ID": identifier,
            "Body": example.get("context") or "",
            "Question": example["question"],
            "Answer": numeric,
            "source_index": index,
        }

    # load_triviaqa_records / load_nqopen_records: QuestionId, Question and
    # Answer as {Value, Aliases}. Their reference list is flat, so the first
    # entry becomes Value and the rest Aliases -- which is also the convention
    # their construct_fewshot_prompt_from_indices uses when it takes text[0].
    return {
        "QuestionId": identifier,
        "Question": example["question"],
        "Answer": {
            "Value": answers[0] if answers else "",
            "Aliases": answers[1:],
        },
        "source_index": index,
    }


def prepare(dataset_name: str, out_root: Path, revision: str | None,
            num_samples: int) -> dict:
    train_dataset, validation_dataset = load_ds(dataset_name, RANDOM_SEED, revision)
    print(
        f"{dataset_name}: train {len(train_dataset)}, "
        f"validation {len(validation_dataset)}"
    )

    # ---- reproduce their RNG sequence exactly --------------------------------
    random.seed(RANDOM_SEED)
    answerable_indices, unanswerable_indices = split_dataset(train_dataset)

    prompt_indices = random.sample(answerable_indices, NUM_FEW_SHOT)
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))
    p_true_indices = random.sample(answerable_indices, P_TRUE_NUM_FEWSHOT)
    remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

    selected: dict[str, list[int]] = {}
    # Their loop runs train before validation, and both draws come from the same
    # stream, so swapping the order would change the validation sample.
    for split_role in ("train", "validation"):
        if split_role == "train":
            dataset = train_dataset
            possible_indices = list(
                set(remaining_answerable) | set(unanswerable_indices)
            )
        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))
        indices = random.sample(
            list(possible_indices), min(num_samples, len(dataset))
        )
        selected[split_role] = indices

    # ---- write ---------------------------------------------------------------
    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_role, indices in selected.items():
        source = train_dataset if split_role == "train" else validation_dataset
        path = out_dir / f"{split_role}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in indices:
                handle.write(
                    json.dumps(
                        as_record(source[index], index, dataset_name),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  wrote {len(indices)} -> {path}")

    demos = [
        {
            "question": train_dataset[i]["question"],
            # Their construct_fewshot_prompt_from_indices takes text[0].
            "answer": train_dataset[i]["answers"]["text"][0],
            "context": train_dataset[i].get("context") or "",
            "source_index": i,
        }
        for i in prompt_indices
    ]
    (out_dir / "fewshot.json").write_text(
        json.dumps(demos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ptrue = [as_record(train_dataset[i], i, dataset_name) for i in p_true_indices]
    (out_dir / "ptrue_examples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ptrue) + "\n",
        encoding="utf-8",
    )

    overlap = set(selected["train"]) & (set(prompt_indices) | set(p_true_indices))
    try:
        import datasets as _datasets

        datasets_version = _datasets.__version__
    except Exception:
        datasets_version = "unknown"

    manifest = {
        "dataset": dataset_name,
        "hf_id": HF_IDS[dataset_name],
        "revision": revision,
        # train_test_split reproducibility depends on both of these.
        "datasets_version": datasets_version,
        "source_fingerprints": dict(LAST_FINGERPRINTS),
        "random_seed": RANDOM_SEED,
        "num_few_shot": NUM_FEW_SHOT,
        "p_true_num_fewshot": P_TRUE_NUM_FEWSHOT,
        "num_samples": num_samples,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "prompt_indices": prompt_indices,
        "p_true_indices": p_true_indices,
        "train_selected": selected["train"],
        "validation_selected": selected["validation"],
        # The train draw must avoid the few-shot and P(True) questions, or the
        # probe would be trained on prompts embedded in its own prefix.
        "train_overlap_with_prompts": sorted(overlap),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if overlap:
        print(f"  WARNING: {len(overlap)} train questions also appear in a prefix")
    print(f"  few-shot indices : {prompt_indices}")
    print(f"  p_true indices   : {p_true_indices[:5]} ...")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-root", default="data/replication")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "--revision",
        default=None,
        help="pin the dataset build; recorded in the manifest either way",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("give --dataset or --all")

    targets = DATASETS if args.all else (args.dataset,)
    out_root = Path(args.out_root)
    for name in targets:
        prepare(name, out_root, args.revision, args.num_samples)
        print()
    print(f"output: {out_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
