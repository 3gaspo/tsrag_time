"""Native ARM rollout; the datastore cutoff remains the real query origin."""

import numpy as np

from timebench.data.windows import CONTEXT_LENGTH, NATIVE_HORIZON
from .retriever import TSRAGIndex

TOP_K = 10


class CausalRetriever:
    """Increment a same-series exact index as observed history becomes available."""
    def __init__(self, references, representations, maximum=None):
        self.references = references
        self.representations = representations
        self.maximum = maximum
        self.key = None
        self.positions = np.empty(0, dtype=np.int64)
        self.index = None

    def search(self, reference, representation):
        item, channel, query_origin = map(int, reference)
        eligible = ((self.references[:, 0] == item) & (self.references[:, 1] == channel)
                    & (self.references[:, 2] + NATIVE_HORIZON <= query_origin))
        positions = np.flatnonzero(eligible)
        if self.maximum is not None:
            positions = positions[-self.maximum:]
        if len(positions) < TOP_K + 1:
            return None
        key = (item, channel)
        prefix = (key == self.key and self.maximum is None and len(positions) >= len(self.positions)
                  and np.array_equal(positions[:len(self.positions)], self.positions))
        if self.index is not None and prefix:
            added = positions[len(self.positions):]
            if len(added):
                self.index.index.add(np.ascontiguousarray(self.representations[added], dtype=np.float32))
        elif not np.array_equal(positions, self.positions) or self.index is None:
            self.index = TSRAGIndex(self.representations[positions])
        self.key, self.positions = key, positions
        distances, local = self.index.search(representation, top_k=TOP_K)
        return distances, positions[local]


def forecast_query(loaded, encoder, retrieval, windows, reference, initial_representation, horizon, device):
    import torch

    history = windows.histories([reference], CONTEXT_LENGTH)[0]
    if len(history) < CONTEXT_LENGTH:
        return None, 'insufficient_query_history'
    current = np.asarray(history, dtype=np.float32)[None]
    representation = initial_representation[None]
    chunks = []
    remaining = horizon
    while remaining:
        found = retrieval.search(reference, representation)
        if found is None:
            return None, 'insufficient_causal_datastore'
        distances, neighbors = found
        sequences = windows.sequences(retrieval.references[neighbors.reshape(-1)]).reshape(1, TOP_K, CONTEXT_LENGTH + NATIVE_HORIZON)
        with torch.inference_mode():
            output = loaded.model(context=torch.from_numpy(current).to(device),
                                  retrieved_seq=torch.from_numpy(sequences).to(device),
                                  distances=torch.from_numpy(distances).to(device))
        native = output.quantile_preds[:, loaded.median_index].detach().float().cpu().numpy()
        if native.shape != (1, NATIVE_HORIZON):
            raise ValueError(f'Unexpected native TS-RAG shape: {native.shape}')
        take = min(remaining, NATIVE_HORIZON)
        chunks.append(native[0, :take])
        remaining -= take
        if remaining:
            if not np.isfinite(native).all():
                return None, 'nonfinite_rollout'
            current = np.concatenate((current, native), axis=-1)[:, -CONTEXT_LENGTH:]
            representation = encoder.representation(torch.from_numpy(current[:, None])).detach().float().cpu().numpy()
            if not np.isfinite(representation).all():
                return None, 'nonfinite_rollout_embedding'
    return np.concatenate(chunks), None
