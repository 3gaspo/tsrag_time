"""Causal TS-RAG retrieval and rollout with explicit project ablations.

Lookback normalization, overlap-aware L2 and query-scale alignment are narrowed
adaptations of Adaptime retrieval.py / adaptime_extraction.py at 33e75400.
Calendar eligibility retains this project's growing datastore (no fitting split).
"""

import numpy as np

from timebench.data.windows import CONTEXT_LENGTH, NATIVE_HORIZON

TOP_K = 10


def instance_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    loc = np.nanmean(values, axis=-1, keepdims=True)
    scale = np.maximum(np.nanstd(values, axis=-1, keepdims=True), 1e-8)
    return (values - loc) / scale


def query_scaled_neighbors(query, sequences):
    """Estimate neighbor statistics from its lookback, never from its future."""
    lookback = sequences[..., :CONTEXT_LENGTH]
    loc = np.nanmean(lookback, axis=-1, keepdims=True)
    scale = np.maximum(np.nanstd(lookback, axis=-1, keepdims=True), 1e-8)
    query_loc = np.nanmean(query, axis=-1, keepdims=True)
    query_scale = np.maximum(np.nanstd(query, axis=-1, keepdims=True), 1e-8)
    return (sequences - loc) / scale * query_scale + query_loc


def normalized_l2(query, candidates, minimum_overlap):
    """Squared L2 on observed overlap, rescaled to the full 512 coordinates."""
    overlap = np.isfinite(candidates) & np.isfinite(query)
    count = overlap.sum(axis=-1)
    difference = np.where(overlap, candidates - query, 0)
    distances = np.sum(difference ** 2, axis=-1)
    distances *= CONTEXT_LENGTH / np.maximum(count, 1)
    distances[count < max(1, int(np.ceil(CONTEXT_LENGTH * minimum_overlap)))] = np.inf
    return distances


class CausalRetriever:
    """Share cached raw T5/IN lookbacks; encode query-scaled T5 candidates on demand."""
    def __init__(self, references, representations, maximum=None, *, windows, options,
                 encoder=None, batch_size=512, minimum_overlap=0.8):
        self.references = references
        self.representations = representations
        self.maximum = maximum
        self.windows = windows
        self.options = options
        self.encoder = encoder
        self.batch_size = batch_size
        self.minimum_overlap = minimum_overlap
        self.ticks = windows.ticks(references)
        self.positions = np.empty(0, dtype=np.int64)
        self.index = None

    def search(self, reference, representation, context):
        item, channel, _ = map(int, reference)
        query_tick = self.windows.ticks(np.asarray(reference)[None])[0]
        cutoff = query_tick - NATIVE_HORIZON * self.windows.tick_step
        # The cutoff is always the real query date, including during autoregression.
        stop = np.searchsorted(self.ticks, cutoff, side='right')
        positions = np.arange(stop, dtype=np.int64)
        if self.options['scope'] == 'same_series':
            refs = self.references[positions]
            positions = positions[(refs[:, 0] == item) & (refs[:, 1] == channel)]
        if self.options['aligned']:
            period = self.windows.task.alignment_period * self.windows.tick_step
            positions = positions[(query_tick - self.ticks[positions]) % period == 0]
        if self.maximum is not None:
            positions = positions[-self.maximum:]
        if len(positions) < TOP_K + 1:
            return None
        dynamic_t5 = self.options['representation'] == 't5' and self.options['query_scale']
        if self.options['representation'] == 't5' and not dynamic_t5:
            from .retriever import TSRAGIndex

            prefix = (self.maximum is None and len(positions) >= len(self.positions)
                      and np.array_equal(positions[:len(self.positions)], self.positions))
            if self.index is not None and prefix:
                added = positions[len(self.positions):]
                if len(added):
                    self.index.index.add(np.ascontiguousarray(self.representations[added], dtype=np.float32))
            elif self.index is None or not np.array_equal(positions, self.positions):
                self.index = TSRAGIndex(self.representations[positions])
            self.positions = positions
            distances, local = self.index.search(representation, top_k=TOP_K)
            return distances, positions[local]

        query = instance_normalize(context)[0] if not dynamic_t5 else representation[0]
        best_positions = np.empty(0, dtype=np.int64)
        best_distances = np.empty(0, dtype=np.float32)
        for start in range(0, len(positions), self.batch_size):
            block = positions[start:start + self.batch_size]
            if dynamic_t5:
                import torch

                histories = np.asarray(self.windows.histories(self.references[block], CONTEXT_LENGTH), dtype=np.float32)
                aligned = query_scaled_neighbors(context, histories)
                candidates = self.encoder.representation(torch.from_numpy(aligned[:, None])).detach().float().cpu().numpy()
                distances = np.sum((candidates - query) ** 2, axis=-1)
                distances[~np.isfinite(distances)] = np.inf
            else:
                # IN L2 is affine invariant; the query-scale axis still changes fusion.
                candidates = self.representations[block]
                distances = normalized_l2(query, candidates, self.minimum_overlap)
            all_distances = np.concatenate((best_distances, distances))
            all_positions = np.concatenate((best_positions, block))
            order = np.lexsort((all_positions, all_distances))[:TOP_K + 1]
            best_distances, best_positions = all_distances[order], all_positions[order]
        if np.isfinite(best_distances).sum() < TOP_K + 1:
            return None
        return best_distances[:TOP_K][None], best_positions[:TOP_K][None]


def forecast_query(loaded, encoder, retrieval, windows, reference, initial_representation, horizon, device):
    import torch

    history = windows.histories([reference], CONTEXT_LENGTH)[0]
    if len(history) < CONTEXT_LENGTH:
        return None, 'insufficient_query_history'
    current = np.asarray(history, dtype=np.float32)[None]
    representation = initial_representation[None] if initial_representation is not None else None
    chunks = []
    remaining = horizon
    while remaining:
        found = retrieval.search(reference, representation, current)
        if found is None:
            return None, 'insufficient_causal_datastore'
        distances, neighbors = found
        sequences = windows.sequences(retrieval.references[neighbors.reshape(-1)]).reshape(1, TOP_K, CONTEXT_LENGTH + NATIVE_HORIZON)
        if retrieval.options['query_scale']:
            sequences = query_scaled_neighbors(current[:, None], sequences)
        with torch.inference_mode():
            output = loaded.model(context=torch.from_numpy(current).to(device),
                                  retrieved_seq=torch.from_numpy(sequences).to(device),
                                  distances=torch.from_numpy(distances).to(device),
                                  neighbors_in_query_scale=retrieval.options['query_scale'])
        native = output.quantile_preds[:, loaded.median_index].detach().float().cpu().numpy()
        if native.shape != (1, NATIVE_HORIZON):
            raise ValueError(f'Unexpected native TS-RAG shape: {native.shape}')
        if not np.isfinite(native).all():
            return None, 'nonfinite_prediction'
        take = min(remaining, NATIVE_HORIZON)
        chunks.append(native[0, :take])
        remaining -= take
        if remaining:
            current = np.concatenate((current, native), axis=-1)[:, -CONTEXT_LENGTH:]
            if retrieval.options['representation'] == 't5':
                representation = encoder.representation(torch.from_numpy(current[:, None])).detach().float().cpu().numpy()
                if not np.isfinite(representation).all():
                    return None, 'nonfinite_rollout_embedding'
    return np.concatenate(chunks), None
