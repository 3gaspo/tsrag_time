"""Beta-smoothed validation weight for frozen Chronos-2 / TS-RAG mixing."""

import numpy as np


def estimate_weight(chronos2, tsrag, labels, histories, references):
    """Compare MSSE per series/date after averaging supported variates."""
    groups = {}
    for c2, rag, target, history, (item, channel, origin) in zip(chronos2, tsrag, labels, histories, references):
        support = np.isfinite(target)
        if not support.any():
            continue
        if not np.all(np.isfinite(c2[support])) or not np.all(np.isfinite(rag[support])):
            continue
        scale = max(float(np.nanstd(np.asarray(history, dtype=np.float64))), 1e-8)
        if not np.isfinite(scale):
            continue
        scores = (np.mean(((c2[support] - target[support]) / scale) ** 2),
                  np.mean(((rag[support] - target[support]) / scale) ** 2))
        if np.isfinite(scores).all():
            groups.setdefault((int(item), int(origin)), []).append(scores)
    wins = 0.0
    for scores in groups.values():
        c2, rag = np.mean(scores, axis=0)
        wins += 1.0 if rag < c2 else 0.5 if rag == c2 else 0.0
    count = len(groups)
    return {'tsrag_weight': (1.0 + wins) / (2.0 + count) if count else 0.0,
            'validation_dates': count, 'tsrag_wins_with_half_ties': wins,
            'prior': [1, 1], 'score': 'per_date_mean_variate_msse',
            'status': 'estimated' if count else 'no_usable_validation_use_chronos2'}


def mix(chronos2, tsrag, weight):
    if weight == 0:
        return np.asarray(chronos2).copy()
    if weight == 1:
        return np.asarray(tsrag).copy()
    return np.asarray((1.0 - weight) * np.asarray(chronos2, dtype=np.float64)
                      + weight * np.asarray(tsrag, dtype=np.float64), dtype=np.float32)
