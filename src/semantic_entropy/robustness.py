"""Paired robustness analyses of fixed saved outputs; no model fitting."""
from __future__ import annotations

import numpy as np


def weighted_auroc(scores, incorrect, weights):
    """AUROC for each row of nonnegative record multiplicities, including ties.

    A draw with one class has NaN AUROC. Integer weights are exactly equivalent
    to expanding the sampled records; fractional weights use the same pair sum.
    """
    scores = np.asarray(scores, float)
    incorrect = np.asarray(incorrect, bool)
    weights = np.atleast_2d(np.asarray(weights, float))
    if weights.shape[1] != len(scores) or len(incorrect) != len(scores):
        raise ValueError('scores, labels, and weight columns must align')
    if np.any(weights < 0) or not np.all(np.isfinite(scores)):
        raise ValueError('finite scores and nonnegative weights required')
    order = np.argsort(scores, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * incorrect[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * ~incorrect[order], starts, axis=1)
    numerator = (pos * (np.cumsum(neg, axis=1) - neg / 2)).sum(axis=1)
    denominator = pos.sum(axis=1) * neg.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan),
                     where=denominator > 0)


def finite_mean(values, axis):
    values = np.asarray(values, float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def paired_delta(cell, weights):
    return (weighted_auroc(cell['left'], cell['incorrect'], weights)
            - weighted_auroc(cell['right'], cell['incorrect'], weights))


def question_weights(cells, draws, rng):
    """One matched question draw per dataset across all its models and seeds."""
    by_dataset = {}
    for cell in cells:
        ids = tuple(cell['qids'])
        if cell['dataset'] in by_dataset:
            if ids != by_dataset[cell['dataset']][0]:
                raise ValueError('Question identities/order differ across matched cells')
        else:
            counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), draws)
            by_dataset[cell['dataset']] = (ids, counts)
    return {dataset: value[1] for dataset, value in by_dataset.items()}


def bootstrap(cells, scheme, draws=2000, seed=20260911):
    """Return paired draws and omission counts for the registered sensitivity.

    cells contain dataset, model, seed, sorted qids, left/right scores and
    incorrect labels. Seeds are averaged within dataset-model units first.
    """
    rng = np.random.default_rng(seed)
    units = sorted({(c['dataset'], c['model']) for c in cells})
    groups = [[c for c in cells if (c['dataset'], c['model']) == u] for u in units]
    if len(units) != 9 or any(len(g) != 3 for g in groups):
        raise ValueError('Expected complete three-dataset/model/seed grid')
    if scheme == 'independent_hierarchical':
        assignments = rng.integers(0, len(units), (draws, len(units)))
        values = np.full((draws, len(units), 3), np.nan)
        for u, group in enumerate(groups):
            rows, slots = np.where(assignments == u)
            for s, cell in enumerate(group):
                n = len(cell['left'])
                weights = rng.multinomial(n, np.full(n, 1 / n), len(rows))
                values[rows, slots, s] = paired_delta(cell, weights)
        unit_weights = np.ones((draws, 9))
    else:
        values = np.full((draws, len(units), 3), np.nan)
        matched = None if scheme == 'independent_records' else question_weights(cells, draws, rng)
        for u, group in enumerate(groups):
            for s, cell in enumerate(group):
                n = len(cell['left'])
                weights = (rng.multinomial(n, np.full(n, 1 / n), draws)
                           if matched is None else matched[cell['dataset']])
                values[:, u, s] = paired_delta(cell, weights)
        unit_weights = np.ones((draws, 9))
        if scheme == 'crossed_questions':
            datasets = sorted({u[0] for u in units})
            models = sorted({u[1] for u in units})
            dw = rng.multinomial(3, [1 / 3] * 3, draws)
            mw = rng.multinomial(3, [1 / 3] * 3, draws)
            unit_weights = np.stack([dw[:, datasets.index(d)] * mw[:, models.index(m)]
                                     for d, m in units], axis=1)
        elif scheme not in {'independent_records', 'linked_questions'}:
            raise ValueError(f'Unknown scheme: {scheme}')
    unit_values = finite_mean(values, axis=2)
    valid_weight = unit_weights * np.isfinite(unit_values)
    denom = valid_weight.sum(axis=1)
    result = np.divide((np.nan_to_num(unit_values) * valid_weight).sum(axis=1), denom,
                       out=np.full(draws, np.nan), where=denom > 0)
    active = np.broadcast_to(unit_weights[:, :, None], values.shape)
    invalid = ~np.isfinite(values)
    return result, dict(
        invalid_cell_fraction=float((invalid * active).sum() / active.sum()),
        any_invalid_cell_draw_fraction=float(np.mean((invalid * active).sum(axis=(1, 2)) > 0)),
        undefined_draw_fraction=float(np.mean(~np.isfinite(result))),
        mean_valid_unit_weight=float(denom.mean()),
    )


def anchor_clusters(entails, indices=None):
    """Recompute the published reassignment scan on an ordered answer subset."""
    entails = np.asarray(entails, bool)
    ids = list(range(len(entails))) if indices is None else list(indices)
    assigned = np.full(len(ids), -1, dtype=int)
    next_id = 0
    for i, first in enumerate(ids):
        if assigned[i] != -1:
            continue
        assigned[i] = next_id
        for j in range(i + 1, len(ids)):
            second = ids[j]
            if entails[first, second] and entails[second, first]:
                assigned[j] = next_id
        next_id += 1
    return assigned


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))
