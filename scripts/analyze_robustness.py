#!/usr/bin/env python3
"""Validate saved AUROCs and run the recorded main-grid sensitivity analyses."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_entropy.robustness import bootstrap, finite_mean, weighted_auroc

DSE = 'discrete_semantic_entropy'
SURFACE = 'surface_entropy'
GRADERS = ['squad_token_f1', 'llm_qwen2.5-72b', 'llm_llama-3.1-70b']
SCHEMES = ['independent_records', 'independent_hierarchical',
           'linked_questions', 'crossed_questions']


def auc(scores, incorrect):
    if not len(scores):
        return np.nan
    return float(weighted_auroc(scores, incorrect, np.ones(len(scores)))[0])


def load_inputs(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def as_cell(d, grader):
    records = sorted(d['records'], key=lambda r: r['qid'])
    return dict(dataset=d['dataset'], model=d['model'], seed=d['seed'],
                qids=[r['qid'] for r in records],
                left=[r['scores'][DSE] for r in records],
                right=[r['scores'][SURFACE] for r in records],
                incorrect=[not r['labels'][grader] for r in records])


def verify_point_estimates(data, measurements):
    index = measurements[measurements.temperature == 1.0].set_index(
        ['condition', 'entailment_backend', 'dataset', 'model', 'seed', 'grader', 'method'])
    rows = []
    for d in data:
        if d['type'] not in {'cell', 'probe'}:
            continue
        methods = ([DSE, SURFACE, 'naive_entropy', 'naive_sample_entropy',
                    'semantic_entropy_full'] if d['type'] == 'cell' else
                   ['probe_uncertainty', 'accuracy_probe_uncertainty'])
        for grader in GRADERS:
            y = [not r['labels'][grader] for r in d['records']]
            for method in methods:
                key = (d['condition'], d['backend'], d['dataset'], d['model'],
                       d['seed'], grader, method)
                expected = index.loc[key]
                value = auc([r['scores'][method] for r in d['records']], y)
                error = abs(value - float(expected.auroc))
                if error > 5.01e-7 or len(y) != int(expected.n_records):
                    raise ValueError(f'Saved AUROC mismatch: {key}: {value}, {expected.auroc}')
                rows.append(dict(zip(index.index.names, key)) | dict(
                    reproduced_auroc=value, recorded_auroc=float(expected.auroc),
                    absolute_error=error, records=len(y)))
    return pd.DataFrame(rows)


def probe_alignment(provenance, fit, records):
    """Require the saved training threshold and its recorded evaluation target."""
    if provenance['label_source'] != 'semantic_entropy_threshold':
        raise ValueError('Wrong probe target kind')
    if provenance['threshold_fit_scope'] != 'training_records_only':
        raise ValueError('Threshold was not established on training records')
    if provenance['label_threshold'] != fit['threshold']:
        raise ValueError('Threshold differs from saved fit')
    if provenance['entropy_target_provenance'] != fit['entropy_target_provenance']:
        raise ValueError('Entropy source provenance differs from saved fit')
    source = provenance['entropy_target_provenance']['eval_source_artifact_sha256']
    if any(r['source_artifact_sha256'] != source for r in records):
        raise ValueError('Evaluation entropy source hash differs from target provenance')
    if provenance['source_score_field'] != DSE:
        raise ValueError('Unexpected entropy target field')
    return np.array([r['scores'][DSE] >= provenance['label_threshold'] for r in records])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--probe-fits', type=Path, required=True,
                    help='JSON file listing the saved probe fits under "probe_fits"')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.inputs)
    measurements = pd.read_csv('data/measurements.csv')
    verified = verify_point_estimates(data, measurements)
    verified.to_csv(args.out / 'point_estimate_verification.csv', index=False)
    print(f'VERIFIED {len(verified)} saved AUROCs', flush=True)
    cells = [d for d in data if d['type'] == 'cell']
    slices = sorted({(d['condition'], d['backend']) for d in cells})
    boot_rows, leave_rows, diagnostics, probes = [], [], [], []
    for condition, backend in slices:
        source = [d for d in cells if (d['condition'], d['backend']) == (condition, backend)]
        for grader in GRADERS:
            group = [as_cell(d, grader) for d in source]
            identity = dict(condition=condition, backend=backend, grader=grader)
            unit = {}
            for c in group:
                delta = auc(c['left'], c['incorrect']) - auc(c['right'], c['incorrect'])
                unit.setdefault((c['dataset'], c['model']), []).append(delta)
            point = np.mean([np.mean(v) for v in unit.values()])
            for axis, name in [(0, 'dataset'), (1, 'generator')]:
                for omitted in sorted({u[axis] for u in unit}):
                    mean = np.mean([np.mean(v) for u, v in unit.items() if u[axis] != omitted])
                    leave_rows.append(identity | dict(omitted_factor=name, omitted=omitted,
                        full_delta=point, retained_delta=mean, change=mean - point))
            for scheme in SCHEMES:
                draws, counts = bootstrap(group, scheme, draws=args.resamples)
                good = draws[np.isfinite(draws)]
                lo, hi = np.quantile(good, [.025, .975])
                boot_rows.append(identity | dict(scheme=scheme, point_delta=point,
                    lower=lo, upper=hi, resamples=args.resamples, seed=20260911,
                    finite_draws=len(good), **counts))
            pd.DataFrame(boot_rows).to_csv(args.out / 'bootstrap_sensitivity.csv', index=False)
            print(f'BOOTSTRAP {condition} {backend} {grader}', flush=True)
    pd.DataFrame(leave_rows).to_csv(args.out / 'leave_one_out.csv', index=False)
    for d in cells:
        records = d['records']
        lengths = np.array([r['graded_chars'] for r in records])
        sample_lengths = np.array([r['sampled_chars'] for r in records])
        cap = np.array([np.nan if r['sample_cap_count'] is None else r['sample_cap_count']
                        for r in records], float)
        entropy = np.array([r['scores'][DSE] for r in records])
        surface = np.array([r['scores'][SURFACE] for r in records])
        median = np.median(lengths)
        for grader in GRADERS:
            y = np.array([not r['labels'][grader] for r in records])
            row = {k: d[k] for k in ['condition', 'backend', 'dataset', 'model', 'seed']}
            row.update(grader=grader, records=len(y), graded_chars_mean=lengths.mean(),
                sampled_chars_mean=sample_lengths.mean(), graded_chars_median=median,
                graded_length_auroc=auc(lengths, y), sampled_length_auroc=auc(sample_lengths, y),
                dse_auroc=auc(entropy, y), surface_auroc=auc(surface, y),
                cap_metadata_records=int(np.isfinite(cap).sum()),
                sample_cap_fraction=float(finite_mean(cap / 10, axis=0)),
                any_cap_response_fraction=float(finite_mean(
                    np.where(np.isfinite(cap), (cap > 0).astype(float), np.nan), axis=0)))
            for name, mask in [('shorter', lengths <= median), ('longer', lengths > median),
                               ('no_cap', cap == 0), ('any_cap', cap > 0)]:
                row[name + '_records'] = int(mask.sum())
                row[name + '_incorrect'] = int(y[mask].sum())
                row[name + '_dse_auroc'] = auc(entropy[mask], y[mask])
                row[name + '_surface_auroc'] = auc(surface[mask], y[mask])
            diagnostics.append(row)
    pd.DataFrame(diagnostics).to_csv(args.out / 'length_truncation.csv', index=False)
    manifest = json.loads(args.probe_fits.read_text())
    fits = {f['sha256']: f for f in manifest['probe_fits']}
    for d in data:
        if d['type'] != 'probe':
            continue
        provenance = d['probe_provenance']
        fit = fits[provenance['probe_artifact_sha256']]
        target = probe_alignment(provenance, fit, d['records'])
        ident = provenance['cell_identity']
        assert ident['dataset'] == d['dataset'] and ident['decoding']['seed'] == d['seed']
        assert ident['model_id'].replace('/', '_') == d['model']
        scores = np.array([r['scores']['probe_uncertainty'] for r in d['records']])
        predicted = scores >= .5
        balanced = (np.mean(predicted[target]) + np.mean(~predicted[~target])) / 2
        row = {k: d[k] for k in ['condition', 'dataset', 'model', 'seed']}
        row.update(records=len(target), entropy_threshold=provenance['label_threshold'],
                   high_entropy_fraction=target.mean(), target_auroc=auc(scores, target),
                   target_balanced_accuracy=balanced,
                   fit_sha256=provenance['probe_artifact_sha256'])
        for grader in GRADERS:
            row[grader + '_incorrectness_auroc'] = auc(scores, [
                not r['labels'][grader] for r in d['records']])
        probes.append(row)
    pd.DataFrame(probes).to_csv(args.out / 'probe_target_validation.csv', index=False)
    (args.out / 'run_manifest.json').write_text(json.dumps(dict(
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        resamples=args.resamples, seed=20260911, verified_aurocs=len(verified),
        bootstrap_comparisons=len(boot_rows), probe_cells=len(probes)), indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
