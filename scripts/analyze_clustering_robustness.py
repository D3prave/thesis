#!/usr/bin/env python3
"""Recluster bounded cached pairs under normalization, order, and sample count."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_entropy.robustness import anchor_clusters, entropy, weighted_auroc

IDENTITY = ['condition', 'dataset', 'model', 'seed', 'qid']
GRADERS = ['squad_token_f1', 'llm_qwen2.5-72b', 'llm_llama-3.1-70b']


def partition(labels):
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


def ordered_partition(matrix, indices):
    """Return membership in original answer order, independent of cluster IDs."""
    assigned = anchor_clusters(matrix, indices)
    restored = assigned[np.argsort(indices)]
    return partition(restored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path)
    ap.add_argument('cache', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    input_bytes = args.inputs.read_bytes()
    if args.inputs.suffix == '.gz':
        input_bytes = gzip.decompress(input_bytes)
    inputs = json.loads(input_bytes)
    cache = json.loads(gzip.decompress(args.cache.read_bytes()))
    if cache['input_sha256'] != hashlib.sha256(input_bytes).hexdigest():
        raise ValueError('Pair cache input hash mismatch')
    key = lambda r: tuple(r[k] for k in IDENTITY)  # noqa: E731
    lookup = {key(r): r for r in cache['rows']}
    if len(lookup) != len(inputs) or {key(r) for r in inputs} != set(lookup):
        raise ValueError('Pair cache identities mismatch')
    if cache['label_codes'] != {'contradiction': 0, 'entailment': 1, 'neutral': 2}:
        raise ValueError('Unexpected label encoding')
    rng = np.random.default_rng(20260911)
    diagnostics, values = [], []
    for r in inputs:
        ident = {k: r[k] for k in IDENTITY}
        matrices = {v: np.asarray(m) for v, m in lookup[key(r)]['matrices'].items()}
        if any(m.shape != (10, 10) or not np.isin(m, [0, 1, 2]).all()
               for m in matrices.values()):
            raise ValueError('Invalid pair matrix')
        matrices = {v: m == 1 for v, m in matrices.items()}
        original = r['clustering_input']['clusters']
        normalized = r['clustering_input']['normalized']
        bases = {v: anchor_clusters(m) for v, m in matrices.items()}
        row = ident | dict(historical_entropy=r['scores']['discrete_semantic_entropy'])
        for v in matrices:
            row[v + '_entropy'] = entropy(bases[v])
            row[v + '_historical_partition_match'] = np.array_equal(
                partition(bases[v]), partition(original))
        row['normalization_partition_changed'] = not np.array_equal(
            partition(bases['raw']), partition(bases['normalized']))
        changed = {v: [] for v in matrices}
        order_entropy = {v: [] for v in matrices}
        for rep in range(50):
            order = rng.permutation(10)
            for v, matrix in matrices.items():
                score = entropy(anchor_clusters(matrix, order))
                order_entropy[v].append(score)
                changed[v].append(not np.array_equal(
                    ordered_partition(matrix, order), partition(bases[v])))
                values.append(ident | dict(experiment='order', samples=10, repeat=rep,
                    method=v, score=score, **r['labels']))
        for v in matrices:
            scores = np.asarray(order_entropy[v])
            row[v + '_order_partition_change_fraction'] = np.mean(changed[v])
            row[v + '_order_entropy_change_fraction'] = np.mean(
                ~np.isclose(scores, entropy(bases[v]), atol=1e-12, rtol=0))
            row[v + '_order_entropy_range'] = np.ptp(np.r_[scores, entropy(bases[v])])
        diagnostics.append(row)
        for m in [2, 4, 6, 8, 10]:
            for rep in range(1 if m == 10 else 50):
                # Keep original relative order to isolate sample-count sensitivity.
                subset = np.sort(rng.choice(10, m, replace=False)) if m < 10 else np.arange(10)
                scores = {v: entropy(anchor_clusters(matrix, subset))
                          for v, matrix in matrices.items()}
                scores['surface'] = entropy([normalized[i] for i in subset])
                for method, score in scores.items():
                    values.append(ident | dict(experiment='samples', samples=m, repeat=rep,
                        method=method, score=score, **r['labels']))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(args.out / 'clustering_records.csv', index=False)
    # Aggregate within each fixed 30-question cell, then equally over eligible cells.
    # Labels do not change across methods or repetitions, so eligibility is identical.
    frame = pd.DataFrame(values)
    group_cols = ['condition', 'experiment', 'samples', 'repeat', 'method']
    cell_rows = []
    for identity, group in frame.groupby(group_cols + ['dataset', 'model'], sort=True):
        for grader in GRADERS:
            y = ~group[grader].to_numpy(bool)
            cell_rows.append(dict(zip(group_cols + ['dataset', 'model'], identity)) | dict(
                grader=grader, records=len(y), incorrect=int(y.sum()),
                auroc=float(weighted_auroc(group.score, y, np.ones(len(y)))[0])))
    cells = pd.DataFrame(cell_rows)
    aggregate = cells.groupby(group_cols + ['grader']).agg(
        mean_auroc=('auroc', 'mean'), valid_cells=('auroc', 'count')).reset_index()
    aggregate.to_csv(args.out / 'clustering_replicates.csv', index=False)
    eligible = cells[(cells.experiment == 'samples') & (cells.samples == 10)
                     & (cells.method == 'raw')][
                         ['condition', 'dataset', 'model', 'grader', 'records', 'incorrect']]
    eligible['eligible'] = (eligible.incorrect > 0) & (eligible.incorrect < eligible.records)
    eligible.to_csv(args.out / 'clustering_eligibility.csv', index=False)
    summary = aggregate.groupby(['condition', 'experiment', 'samples', 'method', 'grader']).agg(
        mean_auroc=('mean_auroc', 'mean'), minimum=('mean_auroc', 'min'),
        maximum=('mean_auroc', 'max'), repeats=('mean_auroc', 'count'),
        valid_cells=('valid_cells', 'min')).reset_index()
    summary.to_csv(args.out / 'clustering_summary.csv', index=False)
    (args.out / 'clustering_manifest.json').write_text(json.dumps({
        k: v for k, v in cache.items() if k != 'rows'} | dict(
            pair_cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
            seed=20260911, records=len(inputs), permutations=50, subsets_per_m=50,
            subset_order='original relative order',
            entropy_tie_rule='sum contributions in sorted count order',
            ranges='observed min/max; not confidence intervals'
        ), indent=2) + '\n')
    print('COMPLETE: 540 records; cached pair inference only', flush=True)


if __name__ == '__main__':
    main()
