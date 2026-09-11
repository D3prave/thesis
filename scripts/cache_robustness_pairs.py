#!/usr/bin/env python3
"""Bounded, offline NLI pair cache for the preregistered robustness subset."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from posthoc_recluster_jsonl import _BatchedNli

MODEL = 'cross-encoder/nli-deberta-v3-large'
REVISION = 'bab4bc7178836f731dcfd18c06ca9def0a137712'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-seconds', type=int, default=5400)
    args = ap.parse_args()
    assert not args.output.exists(), 'Preserve existing cache; use a new output path'
    data = json.loads(args.inputs.read_text())
    assert len(data) == 540 and all(len(r['clustering_input']['raw']) == 10 for r in data)
    pairs = set()
    for row in data:
        r = row['clustering_input']
        for variant in ['raw', 'normalized']:
            for first in r[variant]:
                for second in r[variant]:
                    if first != second:
                        pairs.add((r['question'] + ' ' + first, r['question'] + ' ' + second))
    pairs = sorted(pairs)
    assert len(pairs) <= 97200
    start = time.monotonic()
    entailer = _BatchedNli(model_name=MODEL, model_revision=REVISION,
                         tokenizer_revision=REVISION, device='cuda:0',
                         batch_size=args.batch_size)
    loaded = time.monotonic()
    pilot_n = min(512, len(pairs))
    labels = entailer.predict(pairs[:pilot_n])
    pilot_seconds = time.monotonic() - loaded
    projected_seconds = pilot_seconds * len(pairs) / max(pilot_n, 1)
    pilot = dict(pairs=pilot_n, seconds=pilot_seconds,
                 projected_inference_seconds=projected_seconds, total_pairs=len(pairs))
    args.output.with_suffix('.pilot.json').write_text(json.dumps(pilot, indent=2))
    print('PILOT ' + json.dumps(pilot), flush=True)
    if projected_seconds > args.max_seconds:
        raise SystemExit('Pilot exceeds the bounded inference budget; stop for scope review')
    for first in range(pilot_n, len(pairs), 2048):
        if time.monotonic() - loaded > args.max_seconds:
            raise SystemExit('Bounded inference time reached')
        labels.update(entailer.predict(pairs[first:first + 2048]))
        print(f'CACHE {len(labels)}/{len(pairs)}', flush=True)
    codes = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    rows = []
    for row in data:
        r = row['clustering_input']
        matrices = {}
        for variant in ['raw', 'normalized']:
            matrices[variant] = [[1 if a == b else codes[labels[(r['question'] + ' ' + a,
                                                              r['question'] + ' ' + b)]]
                                 for b in r[variant]] for a in r[variant]]
        rows.append(dict(qid=row['qid'], condition=row['condition'], dataset=row['dataset'],
                         model=row['model'], seed=row['seed'], matrices=matrices))
    import torch
    report = dict(model=MODEL, revision=REVISION, input_sha256=hashlib.sha256(
        args.inputs.read_bytes()).hexdigest(), code_commit=os.environ['VALIDATION_COMMIT'],
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), pilot=pilot,
        total_seconds=time.monotonic() - start, model_load_seconds=loaded - start,
        gpu=torch.cuda.get_device_name(), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy']},
        label_codes=codes, rows=rows)
    with gzip.open(args.output, 'wt') as f:
        json.dump(report, f)
    print('COMPLETE ' + json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
