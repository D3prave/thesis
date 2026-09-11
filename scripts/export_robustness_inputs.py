#!/usr/bin/env python3
"""Read saved cluster outputs and stream a compact, hashed JSONL gzip to stdout.

Run from the historical repository root. No source/output files are modified.
Record selection follows export_data_package_v2.py at the main temperature.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

CONDITIONS = {
    'chat_0shot': ('chat', 'results/ladder', 'results/relabel', 'results/relabel_llama70b'),
    'default_5shot': ('default', 'results/ladder', 'results/relabel_default_qwen',
                      'results/relabel_default_llama70b'),
    'chat_5shot': ('chat', 'results/ladder_factorial_chat5',
                   'results/relabel_factorial_chat5_qwen',
                   'results/relabel_factorial_chat5_llama70b'),
    'default_0shot': ('default', 'results/ladder_factorial_default0',
                      'results/relabel_factorial_default0_qwen',
                      'results/relabel_factorial_default0_llama70b'),
}
CELL = re.compile(r'repl-(chat|default)-(train|eval)-(\w+)-(.+)-s(4[234])-\d+$')
LG = 'cross-encoder_nli-deberta-v3-large'


def main():
    inventory = {}

    def read(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for line in f:
                h.update(line)
                if line.strip():
                    yield json.loads(line)
        inventory[str(path)] = h.hexdigest()

    dedup = Path('results/dedup_triviaqa.json')
    excluded = set(json.loads(dedup.read_text())['excluded_eval_prompts'])
    inventory[str(dedup)] = hashlib.sha256(dedup.read_bytes()).hexdigest()
    chosen = {}
    labels_cache = {}
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode='wb', mtime=0) as out:
        def emit(value):
            out.write((json.dumps(value, separators=(',', ':')) + '\n').encode())

        def emit_cell(path, condition, backend, kind):
            match = CELL.fullmatch(path.parent.name)
            assert match, path
            regime, role, dataset, model, seed = match.groups()
            assert role == 'eval'
            maps = []
            for grader_root in CONDITIONS[condition][2:]:
                label_path = Path(grader_root) / path.parent.name / 'scored_llm.jsonl'
                if str(label_path) not in labels_cache:
                    labels_cache[str(label_path)] = {
                        r['prompt_id']: r['correctness_label'] for r in read(label_path)}
                maps.append(labels_cache[str(label_path)])
            rows, seen, provenance = [], set(), None
            for r in read(path):
                question = r['prompt']
                if dataset == 'triviaqa' and (question in excluded or question in seen):
                    continue
                seen.add(question)
                pid = r['prompt_id']
                assert all(pid in m for m in maps), (path, pid)
                answer = r.get('most_likely_answer') or ''
                if isinstance(answer, dict):
                    answer = answer.get('response', '')
                samples = r['sampled_answers']
                assert len(samples) == 10 and all(isinstance(a, str) for a in samples)
                feature = r.get('sep_feature_metadata', {})
                qid = hashlib.sha256((dataset + '\0' + question).encode()).hexdigest()
                row = dict(qid=qid, prompt_id=pid, scores=r['scores'],
                           labels={'squad_token_f1': r.get('correctness_label_squad',
                                                          r['correctness_label']),
                                   'llm_qwen2.5-72b': maps[0][pid],
                                   'llm_llama-3.1-70b': maps[1][pid]},
                           graded_chars=len(answer),
                           sampled_chars=sum(map(len, samples)) / len(samples),
                           sample_cap_count=feature.get('sampled_max_new_tokens_count'),
                           n_samples=len(samples),
                           source_artifact_sha256=r.get('source_artifact_sha256'))
                if backend == LG and seed == '42' and kind == 'cell':
                    row['clustering_input'] = dict(question=question, raw=samples,
                        normalized=r['normalized_answers'], clusters=r['semantic_clusters'])
                if kind == 'probe':
                    current = r['probe_provenance']
                    if provenance is None:
                        provenance = current
                    assert current == provenance, (path, 'varying probe provenance')
                rows.append(row)
            rows.sort(key=lambda r: r['qid'])
            expected = {'triviaqa': 378, 'nqopen': 400, 'svamp': 300}[dataset]
            assert len(rows) == expected, (path, len(rows), expected)
            if backend == LG and seed == '42' and kind == 'cell':
                ids = {r['qid'] for r in rows}
                if dataset not in chosen:
                    chosen[dataset] = sorted(ids, key=lambda q: hashlib.sha256(
                        ('20260911:' + q).encode()).hexdigest())[:30]
                assert set(chosen[dataset]) <= ids
                for row in rows:
                    if row['qid'] not in chosen[dataset]:
                        row.pop('clustering_input', None)
            emit(dict(type=kind, condition=condition, dataset=dataset, model=model,
                      seed=int(seed), backend=backend, path=str(path), records=rows,
                      probe_provenance=provenance))
            print(f'{kind} {condition} {dataset} {model} {seed} {len(rows)}', file=sys.stderr)

        for condition, (regime, ladder, _, _) in CONDITIONS.items():
            for path in sorted(Path(ladder).glob(f'*/repl-{regime}-eval-*/scored.jsonl')):
                emit_cell(path, condition, path.parent.parent.name, 'cell')
        for path in sorted(Path('results/probe_scored').glob('*/scored.jsonl')):
            regime = CELL.fullmatch(path.parent.name).group(1)
            emit_cell(path, 'chat_0shot' if regime == 'chat' else 'default_5shot', LG, 'probe')
        emit(dict(type='manifest', source_root=str(Path.cwd()), sources=inventory,
                  cluster_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                         text=True).strip(),
                  selected_question_ids=chosen, selection_seed=20260911))


if __name__ == '__main__':
    main()
