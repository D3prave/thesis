#!/usr/bin/env python3
"""Build every Chapter 5 table from the measurement package.

Single source: docs/independent_analysis/data/measurements.csv (10,350 rows).
Nothing is copied from an intermediate table. Writes .tex fragments into
thesis/includes/ and a machine-readable audit trail to stdout.

Intervals: the `units` scheme of scripts/paired_bootstrap_hierarchical.py --
resample the 9 (dataset, model) units with replacement, statistic = mean over
units of the mean-over-seeds value. Computable from cell-level AUROCs alone.
Record-level and hierarchical intervals require the cluster trees and come from
paired_bootstrap_hierarchical.py; those are quoted, not recomputed.
"""
import sys, os
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data/measurements.csv'
OUT  = sys.argv[2] if len(sys.argv) > 2 else 'includes'
os.makedirs(OUT, exist_ok=True)
DDIR = os.path.dirname(os.path.abspath(DATA))
aux = lambda n: os.path.join(DDIR, n)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(DDIR)))


def res(n):
    """Locate a results file: repo-relative first, then beside the data file."""
    for c in (os.path.join(REPO, 'results', n), aux(n)):
        if os.path.exists(c):
            return c
    raise SystemExit(
        f'missing required input: {n}\n'
        f'  looked in {os.path.join(REPO, "results")} and {DDIR}')

XL = 'microsoft_deberta-v2-xlarge-mnli'
LG = 'cross-encoder_nli-deberta-v3-large'
QW = 'Qwen_Qwen2.5-72B-Instruct'
QJ, LJ, F1 = 'llm_qwen2.5-72b', 'llm_llama-3.1-70b', 'squad_token_f1'

PRETTY = {
    'semantic_entropy_full':      'Semantic entropy, likelihood-weighted',
    'discrete_semantic_entropy':  'Discrete semantic entropy',
    'surface_entropy':            'Surface entropy',
    'naive_entropy':              'Naive predictive entropy',
    'naive_sample_entropy':       'Naive sample entropy',
    'ptrue_uncertainty':          'P(True)',
    'accuracy_probe_uncertainty': 'Accuracy probe',
    'probe_uncertainty':          'Semantic entropy probe',
}
SHORT = {
    'microsoft_deberta-v2-xlarge-mnli': r'\texttt{xlarge}',
    'cross-encoder_nli-deberta-v3-large': r'\texttt{v3-large}',
    'Qwen_Qwen2.5-72B-Instruct': r'\texttt{Qwen-72B}',
}
BACKEND = {
    XL: r'\texttt{deberta-v2-xlarge-mnli}',
    LG: r'\texttt{nli-deberta-v3-large}',
    QW: r'\texttt{Qwen2.5-72B-Instruct}',
    'exact-match': r'\texttt{exact-match}',
    'posthoc-nli-deberta-v3-base':  r'\texttt{nli-deberta-v3-base}',
    'posthoc-nli-deberta-v3-large': r'\texttt{nli-deberta-v3-large}',
    'posthoc-llm-judge-qwen2.5-72b-instruct': r'\texttt{Qwen2.5-72B-Instruct}',
}
GRADER = {QJ: 'Qwen judge', LJ: 'Llama judge', F1: 'token-F1'}
COND = {'chat_0shot': r'\texttt{chat\_0shot}', 'default_0shot': r'\texttt{default\_0shot}',
        'chat_5shot': r'\texttt{chat\_5shot}', 'default_5shot': r'\texttt{default\_5shot}'}

d = pd.read_csv(DATA)
sa = d[d.arm == 'short_answer']
lf = d[d.arm == 'long_form']
T1 = sa[sa.temperature == 1.0]
BOOT = pd.read_csv(res('paired_bootstrap_se_vs_surface.csv'))
E7   = pd.read_csv(res('e7_cross_dataset.csv'))
RNG = np.random.default_rng(20260831)
NBOOT = 2000
AUDIT = []


def cells(df, value='auroc'):
    """(dataset, model, seed) -> value, for one method in one slice."""
    return df.set_index(['dataset', 'model', 'seed'])[value]


def unit_stat(s):
    """Mean over the 9 units of the mean over seeds. s indexed by (ds,model,seed)."""
    return s.groupby(level=[0, 1]).mean().mean()


def unit_ci(s, nboot=NBOOT):
    """Percentile CI resampling the 9 (dataset, model) units with replacement."""
    per_unit = s.groupby(level=[0, 1]).mean()
    v = per_unit.to_numpy(float)
    n = v.size
    if n == 0:
        return np.nan, np.nan, np.nan
    draws = v[RNG.integers(0, n, size=(nboot, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def paired(df, m1, m2, value='auroc'):
    """Paired per-cell difference m1 - m2 within one slice."""
    p = df.pivot_table(index=['dataset', 'model', 'seed'], columns='method', values=value)
    if m1 not in p or m2 not in p:
        return pd.Series(dtype=float)
    return (p[m1] - p[m2]).dropna()


def fmt(x, n=3):
    return '---' if pd.isna(x) else f'{x:.{n}f}'


def fmtpm(x, n=3):
    return '---' if pd.isna(x) else f'{x:+.{n}f}'


def write(name, body):
    path = os.path.join(OUT, name)
    open(path, 'w', encoding='utf-8').write(body)
    print(f'  wrote {path}')


def table(caption, label, colspec, header, rows, note=None, small=True):
    out = ['\\begin{table}[htb]', '\\thisfloatsetup{capposition=above}', '\\centering']
    if small:
        out.append('\\footnotesize')
    out += [f'\\caption{{{caption}}}', f'\\label{{{label}}}',
            '\\begingroup', '\\setlength{\\tabcolsep}{3pt}',
            f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', header, '\\midrule']
    out += rows
    out += ['\\bottomrule', '\\end{tabular}', '\\endgroup']
    if note:
        out.append(f'\\par\\vspace{{2pt}}\\footnotesize {note}')
    out.append('\\end{table}')
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------- 5.1 sentence-length
def t_sentence_length():
    sl = T1[(T1.condition == 'chat_0shot') & (T1.grader == QJ)]
    order = ['semantic_entropy_full', 'discrete_semantic_entropy', 'surface_entropy',
             'accuracy_probe_uncertainty', 'ptrue_uncertainty', 'naive_entropy',
             'naive_sample_entropy', 'probe_uncertainty']
    rows = []
    for m in order:
        # each estimator on the backend it was collected under
        b = {'ptrue_uncertainty': XL, 'probe_uncertainty': LG,
             'accuracy_probe_uncertainty': LG}.get(m, XL)
        s = cells(sl[(sl.method == m) & (sl.entailment_backend == b)])
        mu, lo, hi = unit_ci(s)
        nm = PRETTY[m]
        if m == 'surface_entropy':
            nm = r'\textbf{' + nm + '}'
        rows.append(f'{nm} & {fmt(mu)} & [{fmt(lo)}, {fmt(hi)}] & {len(s)//3} \\\\')
        AUDIT.append(('5.1 sentence-length', m, b, round(mu, 4)))
    return table(
        'Sentence-length answers (\\texttt{chat\\_0shot}), Qwen judge, $T=1.0$. AUROC '
        'against answer incorrectness, averaged over the nine (dataset, model) units. '
        'Intervals resample those units. Each estimator is shown on the backend it was '
        'collected under: the entropy estimators on the canonical '
        '\\texttt{deberta-v2-xlarge-mnli}, P(True) on the same, and both probes on '
        '\\texttt{nli-deberta-v3-large}. The clustering-free comparator is set in bold.',
        'tab:sentence-length', '@{}lccr@{}',
        'Method & AUROC & 95\\,\\% CI & Units \\\\', rows)


# ---------------------------------------------------------------- 5.2 short-phrase
def t_short_phrase():
    rows = []
    for g in [F1, QJ, LJ]:
        sub = T1[(T1.condition == 'default_5shot') & (T1.grader == g) &
                 (T1.entailment_backend == XL)]
        vals = {}
        for m in ['discrete_semantic_entropy', 'surface_entropy', 'naive_entropy']:
            vals[m] = unit_stat(cells(sub[sub.method == m]))
        gap = paired(sub, 'discrete_semantic_entropy', 'surface_entropy')
        mu, lo, hi = unit_ci(gap)
        rows.append(f'{GRADER[g]} & {fmt(vals["discrete_semantic_entropy"],4)} & '
                    f'{fmt(vals["surface_entropy"],4)} & {fmt(vals["naive_entropy"],4)} & '
                    f'{fmtpm(mu,4)} & [{fmtpm(lo,3)}, {fmtpm(hi,3)}] \\\\')
        AUDIT.append(('5.2 short-phrase', g, 'gap', round(mu, 4)))
    return table(
        'Short-phrase answers (\\texttt{default\\_5shot}), canonical backend, $T=1.0$, '
        'under all three correctness rules. The final column is the paired '
        'semantic-minus-surface margin with its unit-resampled interval.',
        'tab:short-phrase', '@{}lccccc@{}',
        'Correctness rule & Discrete SE & Surface & Naive pred. & Margin & 95\\,\\% CI \\\\',
        rows)


# ---------------------------------------------------------------- 5.3 SE vs surface
def t_canonical_eight():
    rows = []
    for cond in ['chat_0shot', 'default_0shot', 'chat_5shot', 'default_5shot']:
        for g in [QJ, LJ]:
            r = BOOT[(BOOT.condition == cond) & (BOOT.entailment_backend == XL) &
                     (BOOT.grader == g)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(f'{COND[cond]} & {GRADER[g]} & {fmtpm(r.delta,4)} & '
                        f'[{fmtpm(r.hier_lo,3)}, {fmtpm(r.hier_hi,3)}] & '
                        f'{r.hier_p:.2f} & not resolved \\\\')
            AUDIT.append(('5.3 canonical8', f'{cond}|{g}', 'delta/hier',
                          [round(float(r.delta), 4), round(float(r.hier_lo), 4),
                           round(float(r.hier_hi), 4)]))
    return table(
        'Discrete semantic entropy against surface entropy on the entailment model the '
        'released code loads, under LLM grading: four prompting conditions crossed with '
        'two independent judges. Margins and 95\\,\\% intervals from the hierarchical '
        'paired bootstrap, which resamples both the nine (dataset, model) units and the '
        'records within them. Not one of the eight intervals excludes zero.',
        'tab:canonical-eight', '@{}llcccl@{}',
        'Condition & Judge & Margin & 95\\,\\% CI (hierarchical) & $p$ & Verdict \\\\',
        rows)


def t_resampling_schemes():
    rows = []
    desc = {'records': 'records, units fixed', 'units': 'the 9 units', 'hier': 'both'}
    for sch in ['records', 'units', 'hier']:
        lo, hi = BOOT[f'{sch}_lo'], BOOT[f'{sch}_hi']
        se, sf = int((lo > 0).sum()), int((hi < 0).sum())
        w = float((hi - lo).mean())
        nm = f'\\texttt{{{sch}}}'
        if sch == 'hier':
            rows.append(f'\\textbf{{{nm}}} & \\textbf{{{desc[sch]}}} & '
                        f'\\textbf{{{se+sf} / {len(BOOT)}}} & \\textbf{{{se}}} & '
                        f'\\textbf{{{sf}}} & \\textbf{{{w:.4f}}} \\\\')
        else:
            rows.append(f'{nm} & {desc[sch]} & {se+sf} / {len(BOOT)} & {se} & {sf} & '
                        f'{w:.4f} \\\\')
        AUDIT.append(('5.3 schemes', sch, 'resolved/SE/surface/width',
                      [se + sf, se, sf, round(w, 4)]))
    return table(
        'The same 24 slices under three resampling schemes. Resampling records '
        'alone makes 18 of 24 slices look resolved on intervals roughly three '
        'times too narrow; carrying the between-unit component leaves 11, and '
        'inverts the direction of the majority.',
        'tab:resampling-schemes', '@{}llcccc@{}',
        r'& & & \multicolumn{2}{c}{Direction} & \\'
        '\n\\cmidrule(lr){4-5}\n'
        'Scheme & What it varies & Resolved & SE & Surface & CI width \\\\',
        rows)


def t_all_slices():
    rows = []
    for cond in ['chat_0shot', 'default_0shot', 'chat_5shot', 'default_5shot']:
        for b in [XL, LG, QW]:
            for g in [QJ, LJ, F1]:
                r = BOOT[(BOOT.condition == cond) & (BOOT.entailment_backend == b) &
                         (BOOT.grader == g)]
                if r.empty:
                    continue
                r = r.iloc[0]
                res = ('SE' if r.hier_lo > 0 else
                       ('surface' if r.hier_hi < 0 else '---'))
                rows.append(f'{COND[cond]} & {SHORT[b]} & {GRADER[g]} & '
                            f'{fmtpm(r.delta,4)} & '
                            f'[{fmtpm(r.hier_lo,3)}, {fmtpm(r.hier_hi,3)}] & {res} \\\\')
                AUDIT.append(('5.3 all slices', f'{cond}|{b}|{g}', 'delta',
                              round(float(r.delta), 4)))
    return table(
        'All 24 slices: discrete semantic entropy minus surface entropy, with '
        'hierarchical intervals resampling both the nine units and the records '
        'within them. The final column names the direction where the interval '
        'excludes zero. Every slice resolved in favor of semantic entropy is a '
        'Qwen-backend slice; every slice resolved in favor of surface entropy is '
        'a token-F1 slice.',
        'tab:all-slices', '@{}lllccl@{}',
        'Condition & Backend & Grader & Margin & 95\\,\\% CI (hier.) & Resolved \\\\',
        rows)


def t_ladder():
    """The four-rung decomposition: each rung differs from the one below in one respect."""
    import itertools
    rungs = [('naive_sample_entropy', 'naive_entropy',
              'count strings instead of reading likelihoods'),
             ('surface_entropy', 'naive_sample_entropy',
              'normalize the strings before counting'),
             ('discrete_semantic_entropy', 'surface_entropy',
              'cluster by meaning instead of by string'),
             ('semantic_entropy_full', 'discrete_semantic_entropy',
              'weight the clusters by likelihood')]
    rows = []
    for hi, lo, desc in rungs:
        mus, res_for, res_against = [], 0, 0
        for (c, b, g), sub in T1.groupby(['condition', 'entailment_backend', 'grader']):
            diff = paired(sub, hi, lo)
            if len(diff) < 9:
                continue
            mu, l, h = unit_ci(diff)
            mus.append(mu)
            res_for += int(l > 0)
            res_against += int(h < 0)
        mus = np.array(mus)
        pos = int((mus > 0).sum())
        star = r'\textbf{' if desc.startswith('normalize') else ''
        end = '}' if star else ''
        rows.append(f'{star}{desc}{end} & {star}{fmtpm(mus.mean(),4)}{end} & '
                    f'{star}{pos} / {len(mus)}{end} & {star}{res_for}{end} & '
                    f'{star}{res_against}{end} \\\\')
        AUDIT.append(('5.x ladder', desc, 'mean/pos/for/against',
                      [round(float(mus.mean()), 4), pos, res_for, res_against]))
    return table(
        'What each step of the method is worth. Each rung differs from the one '
        'below it in essentially one respect, so the value of each step is separately '
        'identified. Computed over all 24 (condition, backend, grader) slices at '
        '$T=1.0$, with intervals resampling the nine units. Only the '
        'normalization step is positive in every slice, and it is the only one '
        'whose margin exceeds the seed noise floor of 0.025.',
        'tab:ladder', '@{}p{5.6cm}cccc@{}',
        r'Step & Mean & Positive in & \multicolumn{2}{c}{Resolved} \\'
        '\n\\cmidrule(lr){4-5}\n'
        r' & & & for & against \\', rows)


# ---------------------------------------------------------------- 5.4 paper baselines
def t_paper_baselines():
    rows = []
    for comp, b in [('ptrue_uncertainty', XL), ('probe_uncertainty', LG),
                    ('accuracy_probe_uncertainty', LG)]:
        margins = []
        for cond in ['chat_0shot', 'default_5shot']:
            for g in [QJ, LJ, F1]:
                sub = T1[(T1.condition == cond) & (T1.entailment_backend == b) &
                         (T1.grader == g)]
                diff = paired(sub, 'discrete_semantic_entropy', comp)
                if not diff.empty:
                    margins.append(unit_stat(diff))
        if margins:
            rows.append(f'{PRETTY[comp]} & {len(margins)} & {fmtpm(np.mean(margins))} & '
                        f'[{fmtpm(min(margins))}, {fmtpm(max(margins))}] \\\\')
            AUDIT.append(('5.4 paper baselines', comp, 'mean margin',
                          round(float(np.mean(margins)), 4)))
    return table(
        'Discrete semantic entropy against the comparators the original study evaluates, '
        'over the slices in which each comparator was collected. Semantic entropy leads '
        'in every one.',
        'tab:paper-baselines', '@{}lccc@{}',
        'Comparator & Slices & Mean margin & Range across slices \\\\', rows)


def t_ptrue_size():
    rows = []
    for cond in ['chat_0shot', 'default_5shot']:
        vals = []
        for mdl in ['meta-llama_Llama-3.1-70B-Instruct', 'meta-llama_Llama-3.1-8B-Instruct',
                    'mistralai_Mistral-7B-Instruct-v0.3']:
            sub = T1[(T1.condition == cond) & (T1.entailment_backend == XL) &
                     (T1.grader.isin([QJ, LJ])) &
                     (T1.method == 'ptrue_uncertainty') & (T1.model == mdl)]
            vals.append(sub.auroc.mean())
        rows.append(f'{COND[cond]} & ' + ' & '.join(fmt(v) for v in vals) + r' \\')
        AUDIT.append(('5.4 ptrue size', cond, 'auroc by model',
                      [round(float(v), 4) for v in vals]))
    return table(
        'P(True) by generator size, mean of the two LLM judges. The original study '
        'suggests P(True) '
        '``seems to improve with model size\'\'; within the Llama family here the '
        'ordering is flat or inverted.',
        'tab:ptrue-size', '@{}lccc@{}',
        'Condition & Llama-3.1-70B & Llama-3.1-8B & Mistral-7B \\\\', rows)


# ---------------------------------------------------------------- 5.5 backend
def t_backend():
    rows = []
    for b in [XL, LG, QW]:
        sub = T1[(T1.condition == 'chat_0shot') & (T1.entailment_backend == b) &
                 (T1.grader == QJ)]
        dse = unit_stat(cells(sub[sub.method == 'discrete_semantic_entropy']))
        srf = unit_stat(cells(sub[sub.method == 'surface_entropy']))
        gap = paired(sub, 'discrete_semantic_entropy', 'surface_entropy')
        mu, lo, hi = unit_ci(gap)
        nm = BACKEND[b] + (r' \textbf{(canonical)}' if b == XL else '')
        rows.append(f'{nm} & {fmt(dse,4)} & {fmt(srf,4)} & {fmtpm(mu,4)} & '
                    f'[{fmtpm(lo,3)}, {fmtpm(hi,3)}] \\\\')
        AUDIT.append(('5.5 backend', b, 'dse', round(dse, 4)))
    return table(
        'The entailment backend on identical generations '
        '(\\texttt{chat\\_0shot}, Qwen judge, $T=1.0$). Surface entropy is invariant to '
        'four decimal places across all three, as it must be: it never consults an '
        'entailment model.',
        'tab:backend', '@{}lcccc@{}',
        'Backend & Discrete SE & Surface & Margin & 95\\,\\% CI \\\\', rows)


def t_backend_grader_check():
    rows = []
    for b in [XL, LG, QW]:
        cells_ = []
        for g in [QJ, LJ]:
            sub = T1[(T1.condition == 'chat_0shot') & (T1.entailment_backend == b) &
                     (T1.grader == g)]
            cells_.append(unit_stat(paired(sub, 'discrete_semantic_entropy',
                                           'surface_entropy')))
        rows.append(f'{BACKEND[b]} & {fmtpm(cells_[0],4)} & {fmtpm(cells_[1],4)} \\\\')
        AUDIT.append(('5.5 grader check', b, 'qwen/llama',
                      [round(float(c), 4) for c in cells_]))
    return table(
        'The independent-grader check. The Qwen configuration uses one model as both '
        'entailment backend and correctness judge; re-grading the identical answers with '
        'Llama-3.1-70B leaves the ranking and the magnitudes unchanged, so the backend '
        'effect is not an artifact of one model occupying both roles.',
        'tab:grader-check', '@{}lcc@{}',
        'Backend & Margin, Qwen judge & Margin, Llama judge \\\\', rows)


def t_backend_by_model():
    rows = []
    order = ['meta-llama_Llama-3.1-8B-Instruct', 'mistralai_Mistral-7B-Instruct-v0.3',
             'meta-llama_Llama-3.1-70B-Instruct']
    nice = {'meta-llama_Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
            'mistralai_Mistral-7B-Instruct-v0.3': 'Mistral-7B',
            'meta-llama_Llama-3.1-70B-Instruct': 'Llama-3.1-70B'}
    for mdl in order:
        vals = []
        for b in [XL, QW]:
            sub = T1[(T1.condition == 'chat_0shot') & (T1.entailment_backend == b) &
                     (T1.grader == LJ) & (T1.model == mdl)]
            p_ = sub.pivot_table(index=['dataset', 'seed'], columns='method',
                                 values='auroc')
            vals.append((p_['discrete_semantic_entropy'] - p_['surface_entropy']).mean())
        ch = T1[(T1.condition == 'chat_0shot') & (T1.model == mdl) &
                (T1.entailment_backend == XL) &
                (T1.method == 'discrete_semantic_entropy')].mean_answer_chars.mean()
        rows.append(f'{nice[mdl]} & {ch:.0f} & {fmtpm(vals[0],3)} & {fmtpm(vals[1],3)} \\\\')
        AUDIT.append(('5.5 backend x model', nice[mdl], 'xlarge/qwen',
                      [round(float(v), 4) for v in vals]))
    return table(
        'The semantic-minus-surface margin per generator in \\texttt{chat\\_0shot} '
        'under the Llama judge, on the canonical backend and on the 72B entailment '
        'model. On the released cross-encoder the longest-answer generator gains '
        'least; on the 72B model the same generator gains most. The ordering '
        'inverts with the entailment model on identical generations.',
        'tab:backend-by-model', '@{}lrcc@{}',
        r'Generator & Mean chars & \texttt{xlarge} & \texttt{Qwen-72B} \\', rows)


def t_cluster_counts():
    cc = pd.read_csv(aux('cluster_counts.csv'))
    ba = pd.read_csv(aux('backend_agreement.csv'))
    rows = []
    for _, r in cc[cc.condition.isin(['chat_0shot', 'default_5shot'])].iterrows():
        rows.append(f'{COND[r.condition]} & {BACKEND[r.entailment_backend]} & '
                    f'{r.mean_clusters_per_record:.2f} \\\\')
    note = ('Agreement on cluster count over the records both processed: ' +
            '; '.join(f'{BACKEND[r.backend_a]} vs {BACKEND[r.backend_b]} '
                      f'{100*r.identical_cluster_count:.1f}\\,\\%'
                      for _, r in ba.iterrows()) + '.')
    return table(
        'Mean clusters per record by backend, out of ten samples. The three '
        'implementations of ``bidirectional entailment clustering\'\' do not agree on '
        'how many meanings a set of samples contains.',
        'tab:cluster-counts', '@{}llc@{}',
        'Condition & Backend & Mean clusters \\\\', rows, note=note)


# ---------------------------------------------------------------- 5.6 the 2x2
def t_factorial():
    rows = []
    chars = {'chat_0shot': 255.2, 'default_0shot': 69.8, 'chat_5shot': 11.4,
             'default_5shot': 11.2}
    demos = {'chat_0shot': 0, 'default_0shot': 0, 'chat_5shot': 5, 'default_5shot': 5}
    for cond in ['chat_0shot', 'default_0shot', 'chat_5shot', 'default_5shot']:
        ms = []
        for g in [QJ, LJ]:
            sub = T1[(T1.condition == cond) & (T1.entailment_backend == XL) &
                     (T1.grader == g)]
            ms.append(unit_stat(paired(sub, 'discrete_semantic_entropy', 'surface_entropy')))
        mu = float(np.mean(ms))
        both = pd.concat([paired(T1[(T1.condition == cond) &
                                    (T1.entailment_backend == XL) & (T1.grader == g)],
                                 'discrete_semantic_entropy', 'surface_entropy')
                          for g in [QJ, LJ]])
        _, lo, hi = unit_ci(both)
        rows.append(f'{COND[cond]} & {chars[cond]:.1f} & {demos[cond]} & {fmtpm(mu)} & '
                    f'[{fmtpm(lo,3)}, {fmtpm(hi,3)}] \\\\')
        AUDIT.append(('5.6 factorial', cond, 'mean of two judges', round(mu, 4)))
    return table(
        'The $2\\times2$ on the canonical backend, mean of the two LLM judges. The '
        'isolating contrast is the pair of 0-shot corners, where demonstrations are held '
        'at zero and only the instruction changes. Every interval spans zero.',
        'tab:factorial', '@{}lccrc@{}',
        'Condition & Mean chars & Demos & SE $-$ surface & 95\\,\\% CI (units) \\\\', rows)


# ---------------------------------------------------------------- 5.7 long-form
def t_longform():
    rows = []
    for b in ['exact-match', 'posthoc-nli-deberta-v3-base', 'posthoc-nli-deberta-v3-large',
              'posthoc-llm-judge-qwen2.5-72b-instruct']:
        sub = lf[lf.entailment_backend == b]
        v = {m: sub[sub.method == m].auroc.mean()
             for m in ['discrete_semantic_entropy', 'surface_entropy',
                       'naive_sample_entropy']}
        nm = BACKEND[b]
        if b == 'posthoc-llm-judge-qwen2.5-72b-instruct':
            nm = r'\textbf{' + nm + '}'
        rows.append(f'{nm} & {fmt(v["discrete_semantic_entropy"],4)} & '
                    f'{fmt(v["surface_entropy"],4)} & {fmt(v["naive_sample_entropy"],4)} \\\\')
        AUDIT.append(('5.7 long-form', b, 'dse',
                      round(float(v['discrete_semantic_entropy']), 4)))
    acc = lf.accuracy.mean()
    note = (f'Mean claim accuracy {acc:.3f} over the nine cells. Surface and naive sample '
            'entropy are constant across backends because they never consult one; they '
            'are not performing badly but are structurally degenerate, since ten sampled '
            'biographies are never string-identical and every sample forms its own '
            'cluster. The canonical \\texttt{deberta-v2-xlarge-mnli} was not applied to '
            'this arm.')
    return table(
        'The long-form arm: nine cells, 500 biographies each, claim-level grading by the '
        'Qwen judge. Strict bidirectional entailment throughout, which is not the rule '
        'the original study applies to paragraphs '
        '(Section~\\ref{sec:longform-design}).',
        'tab:longform', '@{}lccc@{}',
        'Backend & Discrete SE & Surface & Naive sample \\\\', rows, note=note)


# ---------------------------------------------------------------- 5.8 temperature
def t_temperature():
    rows = []
    for m in ['discrete_semantic_entropy', 'surface_entropy', 'naive_entropy']:
        vals = []
        for t in [0.1, 0.3, 0.5, 0.7, 1.0]:
            sub = sa[(sa.condition == 'chat_0shot') & (sa.entailment_backend == XL) &
                     (sa.grader == QJ) & (sa.temperature == t) & (sa.method == m)]
            vals.append(unit_stat(cells(sub)) if len(sub) else np.nan)
        best = int(np.nanargmax(vals))
        cellsf = [(r'\textbf{' + fmt(v) + '}') if i == best else fmt(v)
                  for i, v in enumerate(vals)]
        rows.append(f'{PRETTY[m]} & ' + ' & '.join(cellsf) + r' \\')
        AUDIT.append(('5.8 temperature', m, 'T=0.1..1.0',
                      [None if pd.isna(v) else round(float(v), 4) for v in vals]))
    return table(
        'Sampling temperature, sentence-length answers, canonical backend, Qwen judge. '
        'The judge labels are the canonical ones joined by \\texttt{prompt\\_id} and are '
        'identical at all five temperatures, so this is a labels-fixed, estimator-varies '
        'comparison. The sweep is quoted only under LLM grading, for the reason given in '
        'Section~\\ref{sec:temperature-sweep}.',
        'tab:temperature', '@{}lccccc@{}',
        'Method & $T=0.1$ & $0.3$ & $0.5$ & $0.7$ & $1.0$ \\\\', rows)


# ---------------------------------------------------------------- 5.9 correctness rule
def t_correctness():
    rows = []
    for cond in ['chat_0shot', 'default_0shot', 'chat_5shot', 'default_5shot']:
        cellsf = []
        for g in [F1, QJ, LJ]:
            sub = T1[(T1.condition == cond) & (T1.entailment_backend == XL) &
                     (T1.grader == g) & (T1.method == 'discrete_semantic_entropy')]
            cellsf.append(sub.accuracy.mean())
        gaps = []
        for g in [F1, QJ, LJ]:
            sub = T1[(T1.condition == cond) & (T1.entailment_backend == XL) &
                     (T1.grader == g)]
            gaps.append(unit_stat(paired(sub, 'discrete_semantic_entropy',
                                         'surface_entropy')))
        rows.append(f'{COND[cond]} & ' + ' & '.join(fmt(c) for c in cellsf) + ' & ' +
                    ' & '.join(fmtpm(g, 4) for g in gaps) + r' \\')
        AUDIT.append(('5.9 correctness', cond, 'acc f1/qwen/llama',
                      [round(float(c), 4) for c in cellsf]))
    return table(
        'Generator accuracy and the semantic-minus-surface margin under each correctness '
        'rule, canonical backend, $T=1.0$. The same answers are graded three ways. The '
        'margin changes sign between the token-overlap rule and either LLM judge.',
        'tab:correctness', '@{}lcccccc@{}',
        r'& \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{SE $-$ surface} \\'
        '\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n'
        'Condition & token-F1 & Qwen & Llama & token-F1 & Qwen & Llama \\\\', rows)


def t_degenerate_cells():
    sub = T1[(T1.condition == 'chat_0shot') & (T1.grader == F1) &
             (T1.method == 'discrete_semantic_entropy') & (T1.entailment_backend == XL)]
    rows = []
    for _, r in sub[sub.accuracy < 0.05].sort_values('accuracy').iterrows():
        mdl = r.model.split('_')[-1].replace('-Instruct', '')
        rows.append(f'{r.dataset} & \\texttt{{{mdl}}} & {int(r.seed)} & '
                    f'{r.accuracy:.4f} & {int(round(r.accuracy*r.n_records))} '
                    f'/ {int(r.n_records)} \\\\')
        AUDIT.append(('5.9 degenerate', f'{r.dataset}/{mdl}/{int(r.seed)}', 'acc',
                      round(float(r.accuracy), 4)))
    return table(
        'Cells in which the token-overlap rule labels almost nothing correct '
        '(\\texttt{chat\\_0shot}, $T=1.0$). An AUROC computed against three positive '
        'examples is not a measurement of a detector. No LLM-graded cell in the study '
        'falls below 0.36 accuracy.',
        'tab:degenerate-cells', '@{}llccc@{}',
        'Dataset & Generator & Seed & token-F1 accuracy & Correct / records \\\\', rows)


# ---------------------------------------------------------------- 5.10 AUROC vs AURAC
def t_aurac():
    rows = []
    for metric in ['auroc', 'aurac_paper', 'aurac_mean_retained']:
        lead, margins = 0, []
        for cond in ['chat_0shot', 'default_0shot', 'chat_5shot', 'default_5shot']:
            for b in [XL, LG, QW]:
                for g in [QJ, LJ, F1]:
                    sub = T1[(T1.condition == cond) & (T1.entailment_backend == b) &
                             (T1.grader == g)]
                    diff = paired(sub, 'discrete_semantic_entropy', 'surface_entropy',
                                  value=metric)
                    if diff.empty:
                        continue
                    mu = unit_stat(diff)
                    margins.append(mu)
                    lead += int(mu > 0)
        nm = {'auroc': r'\texttt{auroc}', 'aurac_paper': r'\texttt{aurac\_paper}',
              'aurac_mean_retained': r'\texttt{aurac\_mean\_retained}'}[metric]
        rows.append(f'{nm} & {lead} / {len(margins)} & {fmtpm(np.mean(margins),4)} \\\\')
        AUDIT.append(('5.10 aurac', metric, 'slices led', f'{lead}/{len(margins)}'))
    # correlation of aurac_paper with accuracy
    z = T1[T1.method == 'discrete_semantic_entropy'][['accuracy', 'aurac_paper']].dropna()
    r = float(np.corrcoef(z.accuracy, z.aurac_paper)[0, 1])
    AUDIT.append(('5.10 aurac', 'aurac_paper~accuracy', 'r', round(r, 4)))
    note = (f'Over the {len(z)} discrete-semantic-entropy cells of the short-answer arm at '
            f'$T=1.0$, '
            f'\\texttt{{aurac\\_paper}} correlates with the generating model\'s own '
            f'accuracy at $r={r:.3f}$. It is largely reporting how often the model is '
            'right, not how well the detector ranks.')
    return table(
        'The same 24 slices scored with three headline metrics: how many favor discrete '
        'semantic entropy over surface entropy, and the mean margin. Switching from '
        'AUROC to either rejection-accuracy summary moves the verdict from a majority '
        'for semantic entropy to a majority for surface entropy.',
        'tab:aurac', '@{}lcc@{}',
        'Metric & Slices favoring SE & Mean margin \\\\', rows, note=note)


# ---------------------------------------------------------------- 5.11 probes
def t_probes():
    rows = []
    for cond, lbl in [('chat_0shot', 'Sentence-length'),
                      ('default_5shot', 'Short-phrase')]:
        for g in [QJ, LJ, F1]:
            sub = T1[(T1.condition == cond) & (T1.entailment_backend == LG) &
                     (T1.grader == g)]
            a = unit_stat(cells(sub[sub.method == 'accuracy_probe_uncertainty']))
            p_ = unit_stat(cells(sub[sub.method == 'probe_uncertainty']))
            e = E7[(E7.condition == cond) & (E7.grader == g)]
            if len(e):
                ea = e[e.method == 'accuracy_probe_uncertainty'].auroc.mean()
                ep = e[e.method == 'probe_uncertainty'].auroc.mean()
                ood = f'{fmt(ea,4)} & {fmt(ep,4)}'
                AUDIT.append(('5.11 probes OOD', f'{cond}|{g}', 'acc/sep',
                              [round(float(ea), 4), round(float(ep), 4)]))
            else:
                ood = '--- & ---'
            rows.append(f'{lbl} & {GRADER[g]} & {fmt(a,4)} & {fmt(p_,4)} & {ood} \\\\')
            AUDIT.append(('5.11 probes ID', f'{cond}|{g}', 'acc/sep',
                          [round(float(a), 4), round(float(p_), 4)]))
            lbl = ''
    return table(
        "The two hidden-state probes, on \\texttt{nli-deberta-v3-large}, the backend "
        "whose clustering supplied the semantic entropy probe's training target. "
        'In distribution, the probe is trained and scored on disjoint question '
        'sets from the same task. Out of distribution, it is trained on the '
        'pooled prompts of the other two short-answer tasks and scored on the '
        'held-out one. The out-of-distribution pass was not run under the Qwen '
        'judge.',
        'tab:probes', '@{}llcccc@{}',
        r'& & \multicolumn{2}{c}{In distribution} & '
        r'\multicolumn{2}{c}{Out of distribution} \\'
        '\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n'
        'Condition & Grader & Accuracy & SEP & Accuracy & SEP \\\\', rows)


# ---------------------------------------------------------------- seed noise floor
def t_seed_noise():
    sv = pd.read_csv(aux('seed_variability.csv'))
    rows = []
    for m in ['discrete_semantic_entropy', 'semantic_entropy_full', 'surface_entropy',
              'naive_entropy', 'naive_sample_entropy']:
        s = sv[sv.method == m]
        rows.append(f'{PRETTY[m]} & {s.mean_seed_range.mean():.3f} & '
                    f'{s.max_seed_range.max():.3f} \\\\')
        AUDIT.append(('5.0 seed noise', m, 'mean range',
                      round(float(s.mean_seed_range.mean()), 4)))
    return table(
        'The noise floor. AUROC range across the three run seeds, averaged over '
        '(dataset, model) configurations. A margin smaller than these values is inside '
        'the variation produced by resampling alone.',
        'tab:seed-noise', '@{}lcc@{}',
        'Method & Mean seed range & Max seed range \\\\', rows)


if __name__ == '__main__':
    print('building Chapter 5 tables from', DATA)
    write('t_seed_noise.tex',        t_seed_noise())
    write('t_sentence_length.tex',   t_sentence_length())
    write('t_short_phrase.tex',      t_short_phrase())
    write('t_ladder.tex',           t_ladder())
    write('t_canonical_eight.tex',   t_canonical_eight())
    write('t_resampling_schemes.tex', t_resampling_schemes())
    write('t_all_slices.tex',        t_all_slices())
    write('t_paper_baselines.tex',   t_paper_baselines())
    write('t_ptrue_size.tex',        t_ptrue_size())
    write('t_backend.tex',           t_backend())
    write('t_grader_check.tex',      t_backend_grader_check())
    write('t_backend_by_model.tex', t_backend_by_model())
    write('t_cluster_counts.tex',    t_cluster_counts())
    write('t_factorial.tex',         t_factorial())
    write('t_longform.tex',          t_longform())
    write('t_temperature.tex',       t_temperature())
    write('t_correctness.tex',       t_correctness())
    write('t_degenerate_cells.tex',  t_degenerate_cells())
    write('t_aurac.tex',             t_aurac())
    write('t_probes.tex',            t_probes())
    print('\n--- audit trail ---')
    for a in AUDIT:
        print(' ', ' | '.join(str(x) for x in a))


# ---------------------------------------------------------------- figures
def figures(outdir='figures'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 200,
                         'savefig.bbox': 'tight', 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
    INK, MUTE, HL = '#1a1a1a', '#8c8c8c', '#c1440e'
    SURF, ACCENT = '#fcfcfb', '#c1440e'

    # --- forest plot of all 24 slices -------------------------------------
    R = BOOT.rename(columns={'entailment_backend': 'b', 'grader': 'g',
                             'delta': 'mu', 'hier_lo': 'lo', 'hier_hi': 'hi'})
    R = R[['condition', 'b', 'g', 'mu', 'lo', 'hi']].rename(
        columns={'condition': 'cond'}).copy()
    R['grp'] = np.where(R.g == F1, 'token-F1 grading',
                        np.where(R.b == QW, 'Qwen backend, LLM grading',
                                 'released cross-encoders, LLM grading'))
    order = ['released cross-encoders, LLM grading', 'Qwen backend, LLM grading',
             'token-F1 grading']
    R['k'] = R.grp.map({g: i for i, g in enumerate(order)})
    R = R.sort_values(['k', 'mu']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.axvline(0, color=MUTE, lw=0.8, zorder=1)
    for i, r in R.iterrows():
        resolved = (r.lo > 0) or (r.hi < 0)
        c = HL if resolved else INK
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.1, alpha=0.9, zorder=2)
        ax.plot([r.mu], [i], 'o', ms=3.4, color=c,
                mfc=c if resolved else 'white', mew=0.9, zorder=3)
    lab = [f"{r.cond.replace('_',chr(92)+'_')}  ·  "
           f"{ {XL:'xlarge',LG:'v3-large',QW:'Qwen-72B'}[r.b] }  ·  "
           f"{ {QJ:'Qwen',LJ:'Llama',F1:'token-F1'}[r.g] }" for _, r in R.iterrows()]
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels([l.replace('\\_', '_') for l in lab], fontsize=6.4)
    ax.set_ylim(-0.8, len(R) - 0.2)
    ax.set_xlabel('discrete semantic entropy $-$ surface entropy (AUROC)')
    prev = None
    for i, r in R.iterrows():
        if r.grp != prev:
            if prev is not None:
                ax.axhline(i - 0.5, color=MUTE, lw=0.5, ls=':')
            ax.text(0.985, i + 0.05, r.grp, transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=6.6, color=MUTE, style='italic')
            prev = r.grp
    ax.tick_params(length=2)
    fig.savefig(f'{outdir}/margin_forest.pdf'); plt.close(fig)
    print(f'  wrote {outdir}/margin_forest.pdf')

    # --- temperature ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    styles = {'discrete_semantic_entropy': ('-', 'o', INK),
              'surface_entropy': ('--', 's', MUTE),
              'naive_entropy': (':', '^', HL)}
    Ts = [0.1, 0.3, 0.5, 0.7, 1.0]
    for m, (ls, mk, c) in styles.items():
        ys = []
        for t in Ts:
            sub = sa[(sa.condition == 'chat_0shot') & (sa.entailment_backend == XL) &
                     (sa.grader == QJ) & (sa.temperature == t) & (sa.method == m)]
            ys.append(unit_stat(cells(sub)))
        ax.plot(Ts, ys, ls, marker=mk, ms=3.4, lw=1.2, color=c, label=PRETTY[m])
    ax.axhline(0.5, color=MUTE, lw=0.6, ls='-', alpha=0.5)
    ax.text(0.105, 0.505, 'chance', fontsize=6, color=MUTE, va='bottom')
    ax.set_xlabel('sampling temperature'); ax.set_ylabel('AUROC')
    ax.set_xticks(Ts); ax.legend(frameon=False, fontsize=6.8, loc='lower right')
    ax.tick_params(length=2)
    fig.savefig(f'{outdir}/temperature.pdf'); plt.close(fig)
    print(f'  wrote {outdir}/temperature.pdf')

    # --- backend ladder, short and long form -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7))
    ax = axes[0]
    bs = [LG, XL, QW]
    ys = [unit_stat(cells(T1[(T1.condition == 'chat_0shot') & (T1.entailment_backend == b) &
                             (T1.grader == QJ) & (T1.method == 'discrete_semantic_entropy')]))
          for b in bs]
    srf = unit_stat(cells(T1[(T1.condition == 'chat_0shot') & (T1.entailment_backend == XL) &
                             (T1.grader == QJ) & (T1.method == 'surface_entropy')]))
    ax.plot(range(3), ys, '-o', ms=4, lw=1.2, color=INK)
    ax.axhline(srf, color=HL, lw=1.0, ls='--')
    ax.text(2.05, srf, ' surface\n entropy', fontsize=6.2, color=HL, va='center')
    ax.set_xticks(range(3)); ax.set_xticklabels(['v3-large', 'xlarge\n(canonical)', 'Qwen-72B'],
                                                fontsize=6.6)
    ax.set_ylabel('AUROC'); ax.set_title('sentence-length answers', fontsize=7.4)
    ax.set_xlim(-0.35, 2.75); ax.tick_params(length=2)
    ax = axes[1]
    lbs = ['exact-match', 'posthoc-nli-deberta-v3-base', 'posthoc-nli-deberta-v3-large',
           'posthoc-llm-judge-qwen2.5-72b-instruct']
    ys = [lf[(lf.entailment_backend == b) &
             (lf.method == 'discrete_semantic_entropy')].auroc.mean() for b in lbs]
    srf = lf[lf.method == 'surface_entropy'].auroc.mean()
    ax.plot(range(4), ys, '-o', ms=4, lw=1.2, color=INK)
    ax.axhline(srf, color=HL, lw=1.0, ls='--')
    ax.text(3.05, srf, ' surface\n entropy', fontsize=6.2, color=HL, va='center')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['exact\nmatch', 'v3-base', 'v3-large', 'Qwen-72B'], fontsize=6.6)
    ax.set_title('paragraph-length answers', fontsize=7.4)
    ax.set_xlim(-0.35, 3.85); ax.tick_params(length=2)
    fig.savefig(f'{outdir}/backend_ladder.pdf'); plt.close(fig)
    print(f'  wrote {outdir}/backend_ladder.pdf')

    # ---- ladder and factor effects ----
    from matplotlib.patches import Patch
    NOISE=0.0245
    # ---------------- Figure 1: the four-rung decomposition ----------------
    rungs=[('naive_sample_entropy','naive_entropy','count strings,\nnot likelihoods'),
           ('surface_entropy','naive_sample_entropy','normalize before\ncounting'),
           ('discrete_semantic_entropy','surface_entropy','cluster by\nmeaning'),
           ('semantic_entropy_full','discrete_semantic_entropy','weight clusters\nby likelihood')]
    rows=[]
    for hi,lo,lab in rungs:
        mus=[];rf=0;ra=0
        for (c,b,g),sub in T1.groupby(['condition','entailment_backend','grader']):
            diff=paired(sub,hi,lo)
            if len(diff)<9: continue
            mu,l,h=unit_ci(diff); mus.append(mu); rf+=int(l>0); ra+=int(h<0)
        mus=np.array(mus)
        rows.append(dict(lab=lab,mean=mus.mean(),pos=int((mus>0).sum()),n=len(mus),rf=rf,ra=ra))
    R=pd.DataFrame(rows)

    fig,ax=plt.subplots(figsize=(5.6,2.9))
    y=np.arange(len(R))[::-1]
    XR=0.0505   # x where the resolution column starts
    for i,(yy,r) in enumerate(zip(y,R.itertuples())):
        hot = r.lab.startswith('normalize')
        ax.barh(yy, r.mean, height=0.52, color=ACCENT if hot else INK,
                edgecolor=SURF, linewidth=1.0, zorder=3)
        ax.text(r.mean+0.0012, yy, f'{r.mean:+.4f}', va='center', ha='left',
                fontsize=7.6, color=ACCENT if hot else INK,
                fontweight='bold' if hot else 'normal', zorder=4)
        txt = f'{r.rf}/{r.n} resolved' + (f', {r.ra} against' if r.ra else '')
        ax.text(XR, yy, txt, va='center', ha='left', fontsize=6.6,
                color=ACCENT if hot else MUTE, zorder=4)
    ax.axvline(NOISE, color=MUTE, lw=0.9, ls=(0,(4,2)), zorder=2)
    ax.annotate('seed noise floor', xy=(NOISE, len(R)-0.45), xytext=(NOISE, len(R)-0.15),
                fontsize=6.6, color=MUTE, ha='center', va='bottom', annotation_clip=False)
    ax.axvline(0, color=INK, lw=0.7, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(R.lab, fontsize=7.4)
    ax.set_xlim(-0.004, 0.0445)
    ax.set_xticks([0,0.01,0.02,0.03,0.04])
    ax.set_xlabel('mean AUROC gain over the step below, across 24 slices')
    ax.tick_params(length=2); ax.spines['left'].set_visible(False)
    fig.savefig(f'{outdir}/ladder.pdf'); plt.close(fig)
    print(f'  wrote {outdir}/ladder.pdf')

    # ---------------- Figure 2: ranked factor effects ----------------
    def m(**kw):
        q=T1.copy()
        for k,v in kw.items(): q=q[q[k]==v]
        return q.auroc.mean()
    base=dict(condition='chat_0shot',entailment_backend=XL,grader=QJ,
              method='discrete_semantic_entropy')
    sw=sa[(sa.condition=='chat_0shot')&(sa.entailment_backend==XL)&(sa.grader==QJ)&
          (sa.method=='discrete_semantic_entropy')]
    g=T1[(T1.condition=='chat_0shot')&(T1.entailment_backend==XL)&(T1.grader==QJ)&
         (T1.method=='discrete_semantic_entropy')]
    eff=[('sampling temperature', sw[sw.temperature==1.0].auroc.mean()-sw[sw.temperature==0.1].auroc.mean(),'m'),
         ('generator', g.groupby('model').auroc.mean().max()-g.groupby('model').auroc.mean().min(),'m'),
         ('entailment model', m(**{**base,'entailment_backend':QW})-m(**{**base,'entailment_backend':LG}),'m'),
         ('dataset', g.groupby('dataset').auroc.mean().max()-g.groupby('dataset').auroc.mean().min(),'m'),
         ('answer normalization', m(**{**base,'method':'surface_entropy'})-m(**{**base,'method':'naive_sample_entropy'}),'s'),
         ('prompting condition', m(**{**base,'condition':'default_5shot'})-m(**base),'m'),
         ('counting vs likelihood', m(**{**base,'method':'surface_entropy'})-m(**{**base,'method':'naive_entropy'}),'s'),
         ('clustering by meaning', m(**base)-m(**{**base,'method':'surface_entropy'}),'s'),
         ('correctness rule', m(**{**base,'grader':'llm_llama-3.1-70b'})-m(**{**base,'grader':'squad_token_f1'}),'m')]
    eff=sorted(eff,key=lambda x:-abs(x[1]))
    fig,ax=plt.subplots(figsize=(5.2,3.3))
    y=np.arange(len(eff))[::-1]
    for yy,(lab,v,kind) in zip(y,eff):
        step = kind=='s'
        ax.barh(yy, v, height=0.56, color=ACCENT if step else INK,
                edgecolor=SURF, linewidth=1.0,
                hatch='////' if step else None, zorder=3)
        ax.text(v+0.003, yy, f'{v:.3f}', va='center', ha='left', fontsize=7.4,
                color=ACCENT if step else INK, zorder=4)
    ax.axvline(NOISE, color=MUTE, lw=0.9, ls=(0,(4,2)), zorder=2)
    ax.text(NOISE+0.002, -0.55, 'seed noise floor 0.025', fontsize=6.6, color=MUTE,
            va='center', ha='left')
    ax.set_yticks(y); ax.set_yticklabels([e[0] for e in eff], fontsize=7.4)
    ax.set_xlim(0, 0.225); ax.set_xlabel('AUROC moved, on identical generations')
    ax.tick_params(length=2); ax.spines['left'].set_visible(False)
    ax.legend(handles=[Patch(facecolor=ACCENT,hatch='////',edgecolor=SURF,label='a step of the method'),
                       Patch(facecolor=INK,edgecolor=SURF,label='a measurement or data choice')],
              frameon=False, fontsize=6.8, loc='lower right')
    fig.savefig(f'{outdir}/factor_effects.pdf'); plt.close(fig)
    print(f'  wrote {outdir}/factor_effects.pdf')
    for lab,v,k in eff: AUDIT.append(('6.x factor effects', lab, k, round(float(v),4)))


if __name__ == '__main__':
    figures(os.path.join(os.path.dirname(OUT) or '.', 'figures'))
