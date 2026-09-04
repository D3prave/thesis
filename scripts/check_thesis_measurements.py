#!/usr/bin/env python3
"""Check the recorded thesis grid before aggregating it into tables.

This validates the exported measurements, not the cluster's raw artifacts.
An absent whole configuration is an error as well as an absent model/seed row.
"""
from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/independent_analysis/data/measurements.csv'
DATASETS = ('triviaqa', 'nqopen', 'svamp')
MODELS = ('mistralai_Mistral-7B-Instruct-v0.3',
          'meta-llama_Llama-3.1-8B-Instruct', 'meta-llama_Llama-3.1-70B-Instruct')
SEEDS = ('42', '43', '44')
GRADERS = ('squad_token_f1', 'llm_qwen2.5-72b', 'llm_llama-3.1-70b')
BACKENDS = ('microsoft_deberta-v2-xlarge-mnli',
            'cross-encoder_nli-deberta-v3-large', 'Qwen_Qwen2.5-72B-Instruct')
ENTROPY = ('discrete_semantic_entropy', 'semantic_entropy_full',
           'surface_entropy', 'naive_entropy', 'naive_sample_entropy')
MAIN_CONDITIONS = ('chat_0shot', 'default_5shot')
ADDED_CONDITIONS = ('chat_5shot', 'default_0shot')
CONFIG = ('arm', 'condition', 'temperature', 'entailment_backend', 'grader', 'method')


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def check_measurements(rows):
    expected = set()
    for c, b, g, m in itertools.product(MAIN_CONDITIONS, BACKENDS, GRADERS, ENTROPY):
        expected.add(('short_answer', c, '1.0', b, g, m))
    for c, g, m in itertools.product(ADDED_CONDITIONS, GRADERS, ENTROPY):
        expected.add(('short_answer', c, '1.0', BACKENDS[0], g, m))
    for t, c, b, g, m in itertools.product(('0.1','0.3','0.5','0.7'),
            MAIN_CONDITIONS, (BACKENDS[0], BACKENDS[2]), GRADERS, ENTROPY):
        expected.add(('short_answer', c, t, b, g, m))
    for c, g in itertools.product(MAIN_CONDITIONS, GRADERS):
        expected.add(('short_answer', c, '1.0', BACKENDS[0], g, 'ptrue_uncertainty'))
        for m in ('probe_uncertainty', 'accuracy_probe_uncertainty'):
            expected.add(('short_answer', c, '1.0', BACKENDS[1], g, m))
    for b, m in itertools.product(('exact-match', 'posthoc-nli-deberta-v3-base',
            'posthoc-nli-deberta-v3-large', 'posthoc-llm-judge-qwen2.5-72b-instruct'),
            ('discrete_semantic_entropy','surface_entropy','naive_sample_entropy','correctness_score')):
        expected.add(('long_form','paragraph','1.0',b,'llm_qwen2.5-72b',m))
    groups = defaultdict(set)
    for row in rows:
        config = tuple(row[k] for k in CONFIG)
        cell = tuple(row[k] for k in ('dataset','model','seed'))
        if cell in groups[config]:
            raise ValueError(f'duplicate measurement: {config}, {cell}')
        groups[config].add(cell)
        for metric in ('accuracy','auroc','aurac_paper','aurac_mean_retained'):
            if row['arm'] == 'long_form' and metric == 'aurac_mean_retained':
                continue  # Not exported by the separate long-form pipeline.
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'invalid {metric}: {config}, {cell}')
        if row['arm'] == 'short_answer':
            length = float(row['mean_answer_chars'])
            if not math.isfinite(length) or length < 0:
                raise ValueError(f'invalid answer length: {config}, {cell}')
        expected_n = {'triviaqa':378,'nqopen':400,'svamp':300,'bio':500}[row['dataset']]
        if int(row['n_records']) != expected_n:
            raise ValueError(f'unexpected record count: {config}, {cell}')
    if set(groups) != expected:
        raise ValueError(f'configuration mismatch: {len(expected-set(groups))} missing, '
                         f'{len(set(groups)-expected)} unexpected')
    for config, actual in groups.items():
        ds = DATASETS if config[0] == 'short_answer' else ('bio',)
        models = MODELS if config[0] == 'short_answer' else ('Llama-3.1-70B','Llama-3.1-8B','Mistral-7B')
        if actual != set(itertools.product(ds, models, SEEDS)):
            raise ValueError(f'incomplete dataset/model/seed grid: {config}')
    return len(groups)


def check_supporting_exports(rows):
    by_key = {tuple(r[k] for k in CONFIG) + tuple(r[k] for k in ('dataset','model','seed')):
              float(r['auroc']) for r in rows}
    boot = read_rows(ROOT / 'results/paired_bootstrap_se_vs_surface.csv')
    observed = set()
    for r in boot:
        k = (r['condition'], r['entailment_backend'], r['grader'])
        if k in observed:
            raise ValueError(f'duplicate bootstrap comparison: {k}')
        observed.add(k)
        if (int(r['n_units']), int(r['n_cells']), int(r['n_records'])) != (9, 27, 9702):
            raise ValueError(f'unexpected bootstrap coverage: {k}')
        for scheme in ('units', 'records', 'hier'):
            lo, hi, p = (float(r[f'{scheme}_{field}']) for field in ('lo', 'hi', 'p'))
            if not (-1 <= lo <= hi <= 1 and 0 <= p <= 1):
                raise ValueError(f'invalid bootstrap interval: {k}, {scheme}')
        differences = []
        for cell in itertools.product(DATASETS, MODELS, SEEDS):
            prefix = ('short_answer', k[0], '1.0', k[1], k[2])
            differences.append(by_key[prefix+('discrete_semantic_entropy',)+cell] -
                               by_key[prefix+('surface_entropy',)+cell])
        if abs(sum(differences)/len(differences)-float(r['delta'])) > 0.000002:
            raise ValueError(f'bootstrap point estimate differs from measurements: {k}')
    expected = {(r['condition'],r['entailment_backend'],r['grader']) for r in rows
                if r['arm']=='short_answer' and r['temperature']=='1.0'
                and r['method']=='discrete_semantic_entropy'}
    if observed != expected:
        raise ValueError('bootstrap comparisons do not cover the recorded grid')
    ood = read_rows(ROOT / 'results/e7_cross_dataset.csv')
    expected_ood = set()
    for (c,g),m,cell in itertools.product(
            (('chat_0shot',GRADERS[2]),('default_5shot',GRADERS[0])),
            ('accuracy_probe_uncertainty','probe_uncertainty'),
            itertools.product(DATASETS,MODELS,SEEDS)):
        expected_ood.add((c,g,m)+cell)
    keys = [(r['condition'],r['grader'],r['method'],r['dataset'],r['model'],r['seed']) for r in ood]
    if len(keys)!=len(set(keys)) or set(keys)!=expected_ood:
        raise ValueError('incomplete or duplicated out-of-distribution probe grid')
    for r in ood:
        if not math.isfinite(float(r['auroc'])) or not 0<=float(r['auroc'])<=1:
            raise ValueError('invalid out-of-distribution AUROC')


if __name__ == '__main__':
    rows = read_rows(DATA)
    count = check_measurements(rows)
    check_supporting_exports(rows)
    print(f'Coverage OK: {len(rows)} measurements, {count} complete configuration groups; '
          '24 paired comparisons and 108 out-of-distribution probe measurements checked.')
