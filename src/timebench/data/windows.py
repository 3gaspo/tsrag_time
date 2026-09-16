"""Official TIME rows and native, causally available TS-RAG trajectories."""

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import json

import numpy as np

CONTEXT_LENGTH = 512
NATIVE_HORIZON = 64


@dataclass(frozen=True)
class Task:
    dataset: str
    term: str
    prediction_length: int
    test_length: int
    validation_length: int
    seasonality: int
    datastore_stride: int = 1
    max_datastore_windows: int | None = None

    def config(self):
        return asdict(self)


def query_origins(length, interval_length, horizon, *, stop=None):
    """Match TIME's horizon-strided instances, including a partial tail gap."""
    end = length if stop is None else stop
    return np.arange(end - interval_length, end - horizon + 1, horizon, dtype=np.int64)


def candidate_origins(query_origin, *, stride=1, maximum=None):
    """A neighbor's complete native future must be observed at the real query."""
    origins = np.arange(CONTEXT_LENGTH, query_origin - NATIVE_HORIZON + 1, stride, dtype=np.int64)
    return origins if maximum is None else origins[-maximum:]


class Windows:
    def __init__(self, task, storage):
        import datasets

        self.task = task
        self.source_path = Path(storage) / task.dataset
        self.source = datasets.load_from_disk(str(self.source_path))

    @lru_cache(maxsize=2)
    def target(self, item):
        values = np.asarray(self.source[item]['target'])
        return values[None] if values.ndim == 1 else values

    def references(self, split):
        rows = []
        for item in range(len(self.source)):
            target = self.target(item)
            end = target.shape[-1]
            interval = self.task.test_length
            if split == 'validation':
                end -= self.task.test_length
                interval = self.task.validation_length
            elif split != 'test':
                raise ValueError(split)
            origins = query_origins(target.shape[-1], interval, self.task.prediction_length, stop=end)
            for channel in range(len(target)):
                rows.extend((item, channel, int(origin)) for origin in origins)
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)

    def histories(self, references, limit):
        return [self.target(int(item))[int(channel), max(0, int(origin) - limit):int(origin)]
                for item, channel, origin in references]

    def labels(self, references):
        horizon = self.task.prediction_length
        return np.asarray([self.target(int(item))[int(channel), int(origin):int(origin) + horizon]
                           for item, channel, origin in references], dtype=np.float64).reshape(-1, horizon)

    def datastore(self, split_references):
        """Union through testing, including validation; filter at each real query."""
        rows = []
        keys = np.unique(split_references[:, :2], axis=0)
        for item, channel in keys:
            selected = np.all(split_references[:, :2] == (item, channel), axis=1)
            last_query = int(split_references[selected, 2].max())
            origins = candidate_origins(last_query, stride=self.task.datastore_stride)
            rows.extend((int(item), int(channel), int(origin)) for origin in origins)
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)

    def sequences(self, references):
        return np.asarray([self.target(int(item))[int(channel), int(origin) - CONTEXT_LENGTH:int(origin) + NATIVE_HORIZON]
                           for item, channel, origin in references], dtype=np.float32)


def write_prepared(windows, destination):
    destination = Path(destination)
    files = []
    query_rows = []
    for split in ('validation', 'test'):
        refs = windows.references(split)
        np.save(destination / f'{split}_references.npy', refs, allow_pickle=False)
        files.append(f'{split}_references.npy')
        query_rows.append(refs)
    datastore = windows.datastore(np.concatenate(query_rows))
    np.save(destination / 'datastore_references.npy', datastore, allow_pickle=False)
    files.append('datastore_references.npy')
    (destination / 'prepared.json').write_text(json.dumps({
        'schema_version': 1, 'task': windows.task.config(),
        'source_path': str(windows.source_path),
        'counts': {'validation': len(query_rows[0]), 'test': len(query_rows[1]), 'datastore': len(datastore)},
        'datastore_policy': 'same_series_complete_64_step_future_before_real_query',
    }, indent=2))
    return [*files, 'prepared.json']
