"""Stage orchestration over independent, exact-configuration TIME task runs."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import os
import random

import numpy as np

from timebench.data.windows import Task, Windows, write_prepared, CONTEXT_LENGTH, NATIVE_HORIZON
from timebench.evaluation.fallback import summarize_fallbacks
from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION, flatten_univariate_grid, load_evaluation_grid
from timebench.evaluation.timing import EvaluationTimer
from timebench.evaluation.validation import validation_window_mask
from timebench.model_loading.foundation import CONTEXT_LIMITS, CHECKPOINTS
from timebench.paths import dataset_storage_root, outputs_root, weights_root
from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
from timebench.pipeline.runs import (allocate_run, load_manifest, manifest_reference,
    select_completed_runs)

SOURCE_REVISIONS = {'adaptime': '33e75400d8c64414e4e13567c4e899803908bc44',
                    'improved_time': '541a2802cd2a35d39156aef4c37de6964d112786'}
STAGES = ('prepare', 'vanilla', 'extract', 'predict', 'mix', 'evaluate', 'report')
CONTROLS = {'chronos_t5_512': ('chronos_t5', 512), 'chronos_bolt_512': ('chronos_bolt', 512),
            'chronos_bolt_max': ('chronos_bolt', CONTEXT_LIMITS['chronos_bolt']),
            'chronos2_max': ('chronos2', CONTEXT_LIMITS['chronos2'])}


def log(message):
    print(f'[{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}] {message}', flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_run(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def retrieval_methods(config):
    base = dict(config['retrieval'])
    methods = {'tsrag': base}
    if config['ablation']:
        suffixes = {
            ('scope', 'same_series'): 'same_series',
            ('aligned', True): 'aligned',
            ('representation', 'instance_l2'): 'inst_l2',
            ('query_scale', True): 'query_scale',
        }
        for axis, value in config['retrieval_axes'].items():
            key = (axis, value)
            if axis not in base or key not in suffixes or base[axis] == value:
                raise ValueError(f'Unsupported one-axis retrieval ablation: {axis}={value}')
            methods[f'tsrag_{suffixes[key]}'] = {**base, axis: value}
    return methods


class Workflow:
    def __init__(self, config):
        from timebench.evaluation.data import load_dataset_config
        from gluonts.time_feature import get_seasonality
        import datasets

        self.config = config
        self.root = outputs_root() / 'tsrag'
        self.storage = dataset_storage_root()
        self.weights = weights_root()
        self.vanilla_predictions_path = (
            Path(config['vanilla_predictions_path']).expanduser().resolve()
            if config.get('vanilla_predictions_path') else None
        )
        self.seed = int(config['seed'])
        self.device = config['device']
        self.batch_size = int(config['batch_size'])
        self.rag_methods = retrieval_methods(config)
        for options in self.rag_methods.values():
            if options['scope'] not in ('all', 'same_series') or options['representation'] not in ('t5', 'instance_l2'):
                raise ValueError(f'Unsupported retrieval configuration: {options}')
        if not 0 < float(config['minimum_overlap_fraction']) <= 1:
            raise ValueError('minimum_overlap_fraction must be in (0, 1]')
        self.tasks = []
        settings = load_dataset_config(Path(config['dataset_config']))
        selected = config['datasets']
        names = list(settings['datasets']) if selected == ['all'] else selected
        if config['experiment_mode'] == 'test' and selected == ['all']:
            names = ['SG_Weather/D']
        for name in names:
            if name in config['excluded_datasets']:
                continue
            source = datasets.load_from_disk(str(self.storage / name))
            seasonality = int(get_seasonality(source[0]['freq']))
            ds = settings['datasets'][name]
            terms = config['terms']
            if config['experiment_mode'] == 'test' and selected == ['all']:
                terms = ['short']
            for term in terms:
                if term not in ds:
                    continue
                horizon = int(ds[term]['prediction_length'])
                configured_val = ds.get('val_length') or 0
                validation = config['validation_length']
                validation = int(configured_val if validation is None else validation)
                if 0 < validation < horizon:
                    validation = 0
                minimum_length = min(np.asarray(source[i]['target']).shape[-1] for i in range(len(source)))
                test_length = int(ds['test_length'])
                # A requested validation interval must not alter the official test interval.
                if test_length > minimum_length or test_length < horizon:
                    raise ValueError(f'Invalid official test interval: {name}/{term}')
                if minimum_length - test_length - validation < 1:
                    validation = 0
                    log(f'{name}/{term}: no room for requested validation; mixture uses Chronos-2')
                alignment = config['alignment_period']
                if alignment is None:
                    alignment = settings.get('alignment_periods', {}).get(name, seasonality)
                if int(alignment) < 1:
                    raise ValueError('alignment_period must be positive')
                self.tasks.append(Task(name, term, horizon, test_length, validation, seasonality,
                                       int(config['datastore_stride']), config['max_datastore_windows'], int(alignment)))
        if not self.tasks:
            raise ValueError('No tasks selected')
        if self.batch_size < 1 or int(config['embedding_batch_size']) < 1:
            raise ValueError('Batch sizes must be positive')
        if int(config['datastore_stride']) < 1:
            raise ValueError('datastore_stride must be positive')
        if config['max_datastore_windows'] is not None and int(config['max_datastore_windows']) < 1:
            raise ValueError('max_datastore_windows must be positive or null')
        if int(config['t5_samples']) < 1 or (config['validation_length'] is not None and int(config['validation_length']) < 0):
            raise ValueError('Invalid validation_length or t5_samples')

    def model_config(self, method):
        if method in CONTROLS:
            alias, limit = CONTROLS[method]
            result = {'method': 'vanilla', 'backbone': alias, 'context_length': limit,
                      'checkpoint': CHECKPOINTS[alias], 'point_forecast': 'median'}
            if alias == 'chronos_t5':
                result['num_samples'] = int(self.config['t5_samples'])
            return result
        if method in self.rag_methods:
            options = self.rag_methods[method]
            return {'method': 'tsrag', 'backbone': 'chronos_bolt', 'context_length': 512,
                    'native_horizon': 64, 'top_k': 10, 'retrieval': options,
                    'retrieval_encoder': 'chronos_t5_base_eos' if options['representation'] == 't5' else 'instance_normalized_512_lookback',
                    'minimum_overlap_fraction': float(self.config['minimum_overlap_fraction']),
                    'checkpoint': 'released_moe_arm', 'source_commit': '73ac807789d2e61b8a3dfc8514e3fc947fe185cc',
                    'fallback': 'chronos_bolt_max', 'point_forecast': 'median'}
        if method == 'bayes_mixture':
            return {'method': method, 'components': ['chronos2_max', 'tsrag'], 'prior': [1, 1],
                    'score': 'per_date_mean_variate_msse', 'no_validation': 'chronos2',
                    'nonfinite_fallback': 'chronos_bolt_max'}
        return {'method': 'shared', 'target_mode': 'univariate'}

    def identity(self, task, method):
        return {'model': self.model_config(method).get('backbone', method), 'target_mode': 'univariate',
                'dataset': task.dataset.rpartition('/')[0], 'frequency': task.dataset.rpartition('/')[2],
                'term': task.term, 'method': method}

    def path(self, task, phase, method):
        return self.root / phase / method / task.dataset / task.term

    def dependency_reference(self, path):
        return manifest_reference(path)

    def science(self, task, phase, method, dependencies):
        split = phase.rpartition('/')[2] if '/' in phase else None
        pipe = {
            'phase': phase,
            'method': method,
            'task': {
                'dataset': task.dataset,
                'term': task.term,
                'prediction_length': task.prediction_length,
                'test_length': task.test_length,
                'seasonality': task.seasonality,
            },
            'target_mode': 'univariate',
            'dependencies': {
                name: self.dependency_reference(path)
                for name, path in dependencies.items()
            },
        }
        model = {'component': phase, 'method': method}
        experiment = {}
        if phase.startswith('data/'):
            pipe.update(
                split=split,
                datastore_stride=task.datastore_stride,
                max_datastore_windows=task.max_datastore_windows,
                datastore_policy='calendar_causal_complete_64_step_future_before_real_query',
            )
            if split == 'validation':
                pipe.update(
                    validation_length=task.validation_length,
                    validation_stride=task.prediction_length,
                    validation_schedule='walk_backward_from_first_test_origin_at_stride_H',
                )
        elif phase.startswith('extractions/'):
            pipe.update(split=split, representation=method,
                        validation_support='finite_context_and_future' if split == 'validation' else None)
            model = {
                'method': 'retrieval_representation',
                'representation': method,
                'context_length': CONTEXT_LENGTH,
                'checkpoint': 'chronos-t5-base' if method == 't5' else None,
                'instance_normalization': 'lookback_nanmean_nanstd_eps_1e-8',
            }
        elif phase.startswith('predictions/'):
            pipe.update(split=split, covariates='none',
                        validation_support='finite_context_and_future' if split == 'validation' else None,
                        prediction_artifact_contract='vanilla_reuse_nan_counts_retrieval_provenance')
            model = self.model_config(method)
            experiment['seed'] = self.seed
        elif phase == 'selections':
            pipe.update(
                validation_support='finite_context_and_future',
                rule='beta_1_1_per_date_mean_variate_msse_win_frequency',
                no_validation_fallback='chronos2_max',
            )
        if phase == 'evaluations':
            pipe['evaluation_grid'] = EVALUATION_GRID_DEFINITION
            pipe['nan_policy'] = 'omit_nan_predictions_report_counts_reject_infinity'
        return {'model_config': model, 'pipeline_config': pipe,
                'experiment_config': experiment}

    def allocate(self, task, phase, method, dependencies=None):
        dependencies = dependencies or {}
        return allocate_run(self.path(task, phase, method), experiment='tsrag', identity=self.identity(task, method),
                            **self.science(task, phase, method, dependencies),
                            runtime_config={'device': self.device, 'batch_size': self.batch_size,
                                            'embedding_batch_size': int(self.config['embedding_batch_size'])},
                            provenance={'source_revisions': SOURCE_REVISIONS, 'dataset_path': str(self.storage / task.dataset),
                                        'task_config': task.config(),
                                        'upstream_manifests': {name: str(Path(path) / 'manifest.json') for name, path in dependencies.items()}})

    def resolve(self, task, phase, method, dependencies=None):
        expected = self.science(task, phase, method, dependencies or {})
        selected = select_completed_runs(self.path(task, phase, method), config_policy='distinct', repeat_policy='selected')
        matches = [path for path, manifest in selected if all(manifest[key] == value for key, value in expected.items())
                   and manifest['identity'] == self.identity(task, method)]
        if len(matches) != 1:
            raise ValueError(f'Expected one exact completed {phase}/{method} for {task.dataset}/{task.term}; found {len(matches)}')
        return matches[0]

    def finish(self, run, files):
        if os.getenv('TSRAG_DEFER_COMPLETION') == '1':
            write_json(run.run_dir / 'stage_ready.json', {'required_artifacts': files})
            run.compute(files)
        else:
            run.complete(files)

    def prepared(self, task, split):
        return self.resolve(task, f'data/{split}', 'shared')

    def raw(self, task, split, method):
        data = self.prepared(task, split)
        return self.resolve(task, f'predictions/{split}', method, {'data': data})

    def extraction(self, task, split, representation):
        data = self.prepared(task, split)
        return self.resolve(task, f'extractions/{split}', representation, {'data': data})

    def rag(self, task, split, method='tsrag'):
        representation = self.rag_methods[method]['representation']
        deps = {'data': self.prepared(task, split),
                'extraction': self.extraction(task, split, representation),
                'fallback': self.raw(task, split, 'chronos_bolt_max')}
        return self.resolve(task, f'predictions/{split}', method, deps)

    def mixture_weight(self, task):
        deps = {'data': self.prepared(task, 'validation'),
                'chronos2': self.raw(task, 'validation', 'chronos2_max'),
                'tsrag': self.rag(task, 'validation')}
        return self.resolve(task, 'selections', 'bayes_mixture', deps)

    def mixture(self, task):
        deps = {'data': self.prepared(task, 'test'), 'weight': self.mixture_weight(task),
                'chronos2': self.raw(task, 'test', 'chronos2_max'),
                'tsrag': self.rag(task, 'test'),
                'fallback': self.raw(task, 'test', 'chronos_bolt_max')}
        return self.resolve(task, 'predictions/test', 'bayes_mixture', deps)

    def refs(self, data, split):
        return np.load(data / f'{split}_references.npy', mmap_mode='r')

    def reuse_prediction_rows(self, task, split, method, current_run, refs, values,
                              fallback=None, provenance=None, nan_counts=None):
        """Reuse exact item/channel/origin rows from completed split or legacy runs."""
        reused = np.zeros(len(refs), dtype=bool)
        sources = []
        reused_reasons = {}
        current = current_run.manifest
        roots = (
            self.path(task, f'predictions/{split}', method),
            self.path(task, 'predictions', method),
        )
        manifests = sorted(
            {path for root in roots for path in root.glob('run_*/manifest.json')},
            key=lambda path: (path.parent.parent.as_posix(), path.parent.name),
            reverse=True,
        )
        destinations = {tuple(map(int, row)): index for index, row in enumerate(refs)}
        for manifest_path in manifests:
            if manifest_path.parent == current_run.run_dir:
                continue
            manifest = load_manifest(manifest_path)
            metadata_path = manifest_path.parent / 'prediction.json'
            if (
                manifest.get('status') != 'completed'
                or manifest.get('identity') != current.get('identity')
                or manifest.get('model_config') != current.get('model_config')
                or manifest.get('experiment_config') != current.get('experiment_config')
                or not metadata_path.is_file()
            ):
                continue
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata.get('method') != method:
                continue
            recorded_split = metadata.get('split', metadata.get('fallback_reason_split'))
            if recorded_split is not None and recorded_split != split:
                continue
            recorded = manifest.get('provenance', {}).get('upstream_manifests', {}).get('data')
            data_run = Path(recorded).parent if recorded else None
            refs_path = data_run / f'{split}_references.npy' if data_run else None
            prediction_path = manifest_path.parent / 'prediction.npy'
            if not prediction_path.is_file():
                prediction_path = manifest_path.parent / f'{split}.npy'
            if not refs_path or not refs_path.is_file() or not prediction_path.is_file():
                continue
            source_refs = np.load(refs_path, mmap_mode='r', allow_pickle=False)
            predictions = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
            provenance_path = manifest_path.parent / 'retrieval_provenance.npy'
            if provenance is not None and not provenance_path.is_file():
                continue
            nan_counts_path = manifest_path.parent / 'produced_nan_counts.npy'
            if nan_counts is not None and not nan_counts_path.is_file():
                continue
            source_provenance = (
                np.load(provenance_path, mmap_mode='r', allow_pickle=False)
                if provenance is not None else None
            )
            source_nan_counts = (
                np.load(nan_counts_path, mmap_mode='r', allow_pickle=False)
                if nan_counts is not None else None
            )
            fallback_path = manifest_path.parent / 'fallback.npy'
            if not fallback_path.is_file():
                fallback_path = manifest_path.parent / f'{split}_fallback.npy'
            source_fallback = (
                np.load(fallback_path, mmap_mode='r', allow_pickle=False)
                if fallback is not None and fallback_path.is_file()
                else None
            )
            reason_path = manifest_path.parent / 'fallback_reasons.json'
            reason_rows = {}
            if source_fallback is not None and reason_path.is_file():
                for entry in json.loads(reason_path.read_text(encoding='utf-8')):
                    if 'row' in entry:
                        reason_rows[int(entry['row'])] = entry.get('reason', 'inference_error')
            copied = 0
            for source_index, reference in enumerate(source_refs):
                destination = destinations.get(tuple(map(int, reference)))
                if destination is None or reused[destination]:
                    continue
                values[destination] = predictions[source_index]
                if source_provenance is not None:
                    provenance[destination] = source_provenance[source_index]
                if source_nan_counts is not None:
                    nan_counts[destination] = source_nan_counts[source_index]
                if source_fallback is not None:
                    fallback[destination] = source_fallback[source_index]
                    if source_fallback[source_index]:
                        reused_reasons[destination] = reason_rows.get(
                            source_index, 'reused_fallback'
                        )
                reused[destination] = True
                copied += 1
            if copied:
                sources.append({'manifest': str(manifest_path), 'rows': copied})
            if reused.all():
                break
        return reused, sources, reused_reasons

    def support(self, task, split, windows, refs):
        if split == 'test':
            targets, cells = flatten_univariate_grid(*load_evaluation_grid(
                resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
            if targets.shape != (len(refs), task.prediction_length):
                raise ValueError('Seasonal grid and official test references do not align')
            if not np.array_equal(targets, np.isfinite(windows.labels(refs))):
                raise ValueError('Seasonal grid and current labels have different finite support')
            return targets, cells
        labels = windows.labels(refs)
        targets = np.isfinite(labels)
        return targets, validation_window_mask(
            windows.histories(refs, CONTEXT_LENGTH), labels
        )

    def prepare(self):
        for task in self.tasks:
            log(f'prepare dataset={task.dataset} term={task.term} H={task.prediction_length} validation={task.validation_length}')
            windows = Windows(task, self.storage)
            for split in ('validation', 'test'):
                with self.allocate(task, f'data/{split}', 'shared') as run:
                    if run.should_run:
                        self.finish(run, write_prepared(windows, run.run_dir, split))

    def vanilla(self):
        from timebench.model_loading.foundation import Forecaster
        from timebench.pipeline.vanilla_reuse import load_vanilla_test_rows

        # One backbone at a time; Bolt's two context controls share a loaded checkpoint.
        for alias in ('chronos_bolt', 'chronos_t5', 'chronos2'):
            model = None
            for method, (backbone, limit) in CONTROLS.items():
                if alias != backbone:
                    continue
                for task in self.tasks:
                    for split in ('validation', 'test'):
                        data = self.prepared(task, split)
                        with self.allocate(task, f'predictions/{split}', method, {'data': data}) as run:
                            if not run.should_run:
                                continue
                            log(f'vanilla split={split} method={method} dataset={task.dataset} term={task.term} L={limit} H={task.prediction_length}')
                            seed_run(self.seed)
                            windows = Windows(task, self.storage)
                            refs = self.refs(data, split)
                            prediction = np.lib.format.open_memmap(run.run_dir / 'prediction.npy', mode='w+', dtype=np.float32,
                                                                  shape=(len(refs), task.prediction_length))
                            prediction[:] = np.nan
                            reused = np.zeros(len(refs), dtype=bool)
                            reuse_sources = []
                            external_vanilla = None
                            if split == 'test':
                                external_rows, external_vanilla = load_vanilla_test_rows(
                                    self.vanilla_predictions_path,
                                    backbone=alias,
                                    target_mode='univariate',
                                    dataset=task.dataset,
                                    term=task.term,
                                    context_length=limit,
                                    prediction_length=task.prediction_length,
                                    test_length=task.test_length,
                                    references=refs,
                                )
                                if external_rows is not None:
                                    prediction[:] = external_rows
                                    reused[:] = True
                                    reuse_sources.append({
                                        'manifest': external_vanilla['manifest'],
                                        'rows': int(len(refs)),
                                        'kind': 'evaluating_tsfms_canonical_vanilla',
                                    })
                            if not reused.all():
                                reused, reuse_sources, _ = self.reuse_prediction_rows(
                                    task, split, method, run, refs, prediction
                                )
                            if model is None and (~reused).any():
                                model = Forecaster(alias, self.weights, self.device)
                            timer = EvaluationTimer()
                            timer.start()
                            for start in range(0, len(refs), self.batch_size):
                                positions = np.arange(start, min(start + self.batch_size, len(refs)))
                                positions = positions[~reused[positions]]
                                if not len(positions):
                                    continue
                                rows = refs[positions]
                                histories = windows.histories(rows, limit)
                                if split == 'validation':
                                    usable = validation_window_mask(histories, windows.labels(rows))
                                    positions = positions[usable]
                                    histories = [history for history, keep in zip(histories, usable) if keep]
                                if len(positions):
                                    prediction[positions] = model.forecast(
                                        histories, task.prediction_length, limit,
                                        t5_samples=int(self.config['t5_samples'])
                                    )
                            inference_seconds = timer.stop()
                            prediction.flush()
                            targets, cells = self.support(task, split, windows, refs)
                            if np.any(cells & np.any(targets & np.isinf(prediction), axis=-1)):
                                raise ValueError(f'Infinite {method} on required {split} support')
                            np.save(run.run_dir / 'context_length.npy',
                                    np.minimum(refs[:, 2], limit), allow_pickle=False)
                            prepared = json.loads((data / 'prepared.json').read_text(encoding='utf-8'))
                            usable_ticks = np.unique(windows.ticks(refs)[cells]) if len(refs) else []
                            required = targets & cells[:, None]
                            prediction_nan_values = int((required & np.isnan(prediction)).sum())
                            prediction_values = int(required.sum())
                            write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method,
                                'split': split, 'context_limit': limit,
                                'context_policy': 'all_available_history_capped_at_limit',
                                'inference_seconds': inference_seconds,
                                'reused_rows': int(reused.sum()),
                                'newly_inferred_rows': int((~reused).sum()),
                                'reuse_sources': reuse_sources,
                                'external_vanilla': external_vanilla,
                                'prediction_nan_values': prediction_nan_values,
                                'produced_nan_values': prediction_nan_values,
                                'prediction_values': prediction_values,
                                'prediction_nan_rate': (prediction_nan_values / prediction_values
                                                        if prediction_values else None),
                                'validation_counts': ({**prepared['counts'], 'usable_rows': int(cells.sum()),
                                                       'usable_dates': int(len(usable_ticks))}
                                                      if split == 'validation' else None)})
                            self.finish(run, ['prediction.npy', 'context_length.npy', 'prediction.json'])
            del model

    def extract(self):
        import torch
        from timebench.external_models.tsrag.retriever import TSRAGRetriever
        from timebench.external_models.tsrag.inference import instance_normalize

        encoder = None
        representations = {'instance_l2'} | {options['representation'] for options in self.rag_methods.values()}
        for task in self.tasks:
            windows = Windows(task, self.storage)
            for split in ('validation', 'test'):
                data = self.prepared(task, split)
                for representation in sorted(representations):
                    with self.allocate(task, f'extractions/{split}', representation, {'data': data}) as run:
                        if not run.should_run:
                            continue
                        started = perf_counter()
                        error = None
                        files = []
                        try:
                            if representation == 't5' and encoder is None:
                                encoder = TSRAGRetriever(self.weights / 'chronos-t5-base', device_map=self.device)
                            for role, refs in (
                                ('datastore', self.refs(data, 'datastore')),
                                ('query', self.refs(data, split)),
                            ):
                                width = 768 if representation == 't5' else CONTEXT_LENGTH
                                embeddings = np.lib.format.open_memmap(
                                    run.run_dir / f'{role}.npy', mode='w+', dtype=np.float32,
                                    shape=(len(refs), width),
                                )
                                embeddings[:] = np.nan
                                batch_size = int(self.config['embedding_batch_size'])
                                for start in range(0, len(refs), batch_size):
                                    positions = np.arange(start, min(start + batch_size, len(refs)))
                                    rows = refs[positions]
                                    histories = windows.histories(rows, CONTEXT_LENGTH)
                                    if role == 'query' and split == 'validation':
                                        usable = validation_window_mask(histories, windows.labels(rows))
                                        positions = positions[usable]
                                        histories = [history for history, keep in zip(histories, usable) if keep]
                                    if not len(positions):
                                        continue
                                    values = np.full((len(histories), CONTEXT_LENGTH), np.nan, dtype=np.float32)
                                    for index, history in enumerate(histories):
                                        values[index, -len(history):] = history
                                    if representation == 't5':
                                        block = encoder.representation(torch.from_numpy(values[:, None])).detach().float().cpu().numpy()
                                        if not np.isfinite(block).all():
                                            raise ValueError(f'Non-finite {split} retrieval embedding')
                                    else:
                                        block = instance_normalize(values)
                                    embeddings[positions] = block
                                embeddings.flush()
                                files.append(f'{role}.npy')
                        except Exception as exception:
                            error = {'type': type(exception).__name__, 'message': str(exception)}
                            log(f'TS-RAG {representation} extraction fallback {task.dataset}/{task.term}: {error}')
                        write_json(run.run_dir / 'extraction.json', {
                            'schema_version': 1, 'split': split,
                            'representation': representation, 'error': error,
                            'extraction_seconds': perf_counter() - started,
                        })
                        self.finish(run, [*files, 'extraction.json'])

    def predict(self):
        from timebench.external_models.tsrag.retriever import TSRAGRetriever
        from timebench.external_models.tsrag.inference import (
            CausalRetriever, forecast_query, summarize_retrieval_provenance,
        )
        from timebench.model_loading.tsrag import load_tsrag

        loaded = encoder = None
        for method, options in self.rag_methods.items():
            for task in self.tasks:
                for split in ('validation', 'test'):
                    data = self.prepared(task, split)
                    extraction = self.extraction(task, split, options['representation'])
                    fallback = self.raw(task, split, 'chronos_bolt_max')
                    deps = {'data': data, 'extraction': extraction, 'fallback': fallback}
                    with self.allocate(task, f'predictions/{split}', method, deps) as run:
                        if not run.should_run:
                            continue
                        log(f'predict split={split} method={method} dataset={task.dataset} term={task.term} '
                            f'retrieval={options} alignment_period={task.alignment_period} fallback=chronos_bolt_max')
                        seed_run(self.seed)
                        windows = Windows(task, self.storage)
                        metadata = json.loads((extraction / 'extraction.json').read_text())
                        task_error = metadata['error']
                        refs = self.refs(data, split)
                        base = np.load(fallback / 'prediction.npy', mmap_mode='r')
                        values = np.lib.format.open_memmap(run.run_dir / 'prediction.npy', mode='w+', dtype=np.float32, shape=base.shape)
                        values[:] = base
                        mask = np.zeros(len(refs), dtype=bool)
                        provenance_rows = np.zeros((len(refs), 4), dtype=np.float64)
                        produced_nan_counts = np.zeros(len(refs), dtype=np.int64)
                        reused, reuse_sources, reused_reasons = self.reuse_prediction_rows(
                            task, split, method, run, refs, values, mask,
                            provenance_rows, produced_nan_counts
                        )
                        if task_error is None and (~reused).any():
                            try:
                                if loaded is None:
                                    loaded = load_tsrag(self.weights / 'chronos-bolt-base', self.weights / 'ts-rag', device=self.device)
                                if options['representation'] == 't5' and encoder is None:
                                    encoder = TSRAGRetriever(self.weights / 'chronos-t5-base', device_map=self.device)
                            except Exception as exception:
                                task_error = {'type': type(exception).__name__, 'message': str(exception)}
                        reasons = [
                            {'row': int(row), 'reason': reason, 'reused': True}
                            for row, reason in sorted(reused_reasons.items())
                        ]
                        reasons_by_row = dict(reused_reasons)
                        targets, cells = self.support(task, split, windows, refs)
                        retrieval = None
                        if task_error is None and (~reused).any():
                            retrieval = CausalRetriever(self.refs(data, 'datastore'), np.load(extraction / 'datastore.npy', mmap_mode='r'),
                                                        task.max_datastore_windows, windows=windows, options=options, encoder=encoder,
                                                        batch_size=int(self.config['embedding_batch_size']),
                                                        minimum_overlap=float(self.config['minimum_overlap_fraction']))
                            embeddings = (np.load(extraction / 'query.npy', mmap_mode='r')
                                          if options['representation'] == 't5' else None)
                        timer = EvaluationTimer()
                        timer.start()
                        for row, reference in enumerate(refs):
                            if reused[row]:
                                continue
                            reason = ('unusable_validation_window'
                                      if split == 'validation' and not cells[row]
                                      else 'extraction_or_model_error' if task_error else None)
                            if reason is None:
                                try:
                                    (forecast, reason, provenance_rows[row],
                                     produced_nan_counts[row]) = forecast_query(
                                        loaded, encoder, retrieval, windows, reference,
                                        embeddings[row] if embeddings is not None else None,
                                        task.prediction_length, self.device,
                                    )
                                    if reason is None:
                                        if cells[row] and not np.all(~targets[row] | np.isfinite(forecast)):
                                            reason = 'nonfinite_prediction'
                                        else:
                                            values[row] = forecast
                                except Exception as exception:
                                    reason = 'inference_error'
                                    reasons.append({'row': row, 'type': type(exception).__name__, 'message': str(exception)})
                            if reason:
                                mask[row] = True
                                reasons_by_row[row] = reason
                                if reason != 'inference_error':
                                    reasons.append({'row': row, 'reason': reason})
                        inference_seconds = timer.stop()
                        values.flush()
                        np.save(run.run_dir / 'fallback.npy', mask, allow_pickle=False)
                        np.save(run.run_dir / 'retrieval_provenance.npy', provenance_rows,
                                allow_pickle=False)
                        np.save(run.run_dir / 'produced_nan_counts.npy', produced_nan_counts,
                                allow_pickle=False)
                        write_json(run.run_dir / 'fallback_reasons.json', reasons)
                        fallback_summary = summarize_fallbacks(mask, cells, reasons_by_row)
                        counts = {'all_rows': fallback_summary['all_fallback_rows'],
                                  'eligible_rows': fallback_summary['fallback_count'],
                                  'grid_rows': fallback_summary['evaluated_rows'],
                                  'total_rows': fallback_summary['total_rows']}
                        prepared = json.loads((data / 'prepared.json').read_text(encoding='utf-8'))
                        usable_ticks = np.unique(windows.ticks(refs)[cells]) if len(refs) else []
                        write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method,
                            'split': split, 'context_limit': 512, 'retrieval': options,
                            'alignment_period': task.alignment_period,
                            'fallback_method': 'chronos_bolt_max', 'task_error': task_error,
                            'fallback_counts': {split: counts}, 'fallback_reason_split': split,
                            'fallback_reasons': fallback_summary['fallback_reasons'],
                            'reused_rows': int(reused.sum()),
                            'newly_inferred_rows': int((~reused).sum()),
                            'reuse_sources': reuse_sources,
                            'produced_nan_values': int(produced_nan_counts.sum()),
                            'retrieval_provenance': summarize_retrieval_provenance(provenance_rows),
                            'validation_counts': ({**prepared['counts'], 'usable_rows': int(cells.sum()),
                                                   'usable_dates': int(len(usable_ticks))}
                                                  if split == 'validation' else None),
                            'datastore_preprocessing_seconds': metadata['extraction_seconds'],
                            'fallback_source_inference_seconds': json.loads((fallback / 'prediction.json').read_text())['inference_seconds'],
                            'inference_seconds': inference_seconds,
                            'timing_policy': 'measured_query_work_with_precomputed_bolt_max_fallback'})
                        self.finish(run, ['prediction.npy', 'fallback.npy', 'retrieval_provenance.npy',
                                          'produced_nan_counts.npy',
                                          'fallback_reasons.json', 'prediction.json'])

    def mix(self):
        from timebench.external_models.tsrag.inference import summarize_retrieval_provenance
        from timebench.proposal.mixture import estimate_weight, mix

        for task in self.tasks:
            validation_data = self.prepared(task, 'validation')
            validation_c2 = self.raw(task, 'validation', 'chronos2_max')
            validation_rag = self.rag(task, 'validation')
            weight_deps = {'data': validation_data, 'chronos2': validation_c2, 'tsrag': validation_rag}
            with self.allocate(task, 'selections', 'bayes_mixture', weight_deps) as run:
                if run.action == 'finalize':
                    run.complete()
                    (run.run_dir / 'stage_ready.json').unlink(missing_ok=True)
                elif run.should_run:
                    windows = Windows(task, self.storage)
                    refs = self.refs(validation_data, 'validation')
                    weight = estimate_weight(
                        np.load(validation_c2 / 'prediction.npy'),
                        np.load(validation_rag / 'prediction.npy'),
                        windows.labels(refs), windows.histories(refs, 512), refs,
                    )
                    write_json(run.run_dir / 'weight.json', weight)
                    run.complete(['weight.json'])
            weight_run = self.mixture_weight(task)
            data = self.prepared(task, 'test')
            c2 = self.raw(task, 'test', 'chronos2_max')
            rag = self.rag(task, 'test')
            fallback = self.raw(task, 'test', 'chronos_bolt_max')
            deps = {'data': data, 'weight': weight_run, 'chronos2': c2, 'tsrag': rag, 'fallback': fallback}
            with self.allocate(task, 'predictions/test', 'bayes_mixture', deps) as run:
                if not run.should_run:
                    continue
                windows = Windows(task, self.storage)
                weight = json.loads((weight_run / 'weight.json').read_text(encoding='utf-8'))
                c2_test = np.load(c2 / 'prediction.npy', mmap_mode='r')
                rag_test = np.load(rag / 'prediction.npy', mmap_mode='r')
                timer = EvaluationTimer()
                timer.start()
                prediction = mix(c2_test, rag_test, weight['tsrag_weight'])
                inference_seconds = timer.stop()
                targets, cells = self.support(task, 'test', windows, self.refs(data, 'test'))
                produced_nan_values = int(
                    (targets & cells[:, None] & np.isnan(prediction)).sum()
                )
                invalid = cells & ~np.all(~targets | np.isfinite(prediction), axis=-1)
                prediction[invalid] = np.load(fallback / 'prediction.npy', mmap_mode='r')[invalid]
                np.save(run.run_dir / 'prediction.npy', prediction, allow_pickle=False)
                np.save(run.run_dir / 'fallback.npy', invalid, allow_pickle=False)
                provenance_rows = (
                    np.asarray(np.load(rag / 'retrieval_provenance.npy', mmap_mode='r')).copy()
                    if weight['tsrag_weight'] else np.zeros((len(prediction), 4), dtype=np.float64)
                )
                np.save(run.run_dir / 'retrieval_provenance.npy', provenance_rows,
                        allow_pickle=False)
                component_seconds = sum(json.loads((path / 'prediction.json').read_text())['inference_seconds'] for path in (c2, rag))
                write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': 'bayes_mixture', **weight,
                                                           'fallback_method': 'chronos_bolt_max',
                                                           'produced_nan_values': produced_nan_values,
                                                           'retrieval_provenance': summarize_retrieval_provenance(provenance_rows),
                                                           'split': 'test', 'inference_seconds': component_seconds + inference_seconds,
                                                           'mix_only_seconds': inference_seconds, 'nonfinite_fallback_count': int(invalid.sum())})
                self.finish(run, ['prediction.npy', 'fallback.npy', 'retrieval_provenance.npy',
                                  'prediction.json'])

    def methods(self):
        return [*CONTROLS, *self.rag_methods, 'bayes_mixture']

    def prediction(self, task, method):
        return (self.rag(task, 'test', method) if method in self.rag_methods
                else self.mixture(task) if method == 'bayes_mixture'
                else self.raw(task, 'test', method))

    def evaluation(self, task, method):
        return self.resolve(task, 'evaluations', method, {'prediction': self.prediction(task, method)})

    def evaluate(self):
        from timebench.evaluation.data import Dataset
        from timebench.evaluation.saver import save_window_predictions

        for task in self.tasks:
            windows = Windows(task, self.storage)
            dataset = Dataset(task.dataset, term=task.term, prediction_length=task.prediction_length,
                              test_length=task.test_length, val_length=0, storage_path=self.storage,
                              to_univariate=windows.target(0).shape[0] > 1)
            for method in self.methods():
                prediction = self.prediction(task, method)
                with self.allocate(task, 'evaluations', method, {'prediction': prediction}) as run:
                    if not run.should_run:
                        continue
                    metadata = json.loads((prediction / 'prediction.json').read_text())
                    log(f'evaluate method={method} dataset={task.dataset} term={task.term}')
                    values = np.load(prediction / 'prediction.npy', mmap_mode='r')
                    save_window_predictions(dataset, values[:, None, :], f'{task.dataset}/{task.term}', str(self.root),
                                            seasonality=task.seasonality, quantile_levels=[0.5], task_output_dir=str(run.run_dir),
                                            model_hyperparams={'model': method, 'experiment': 'tsrag', 'target_mode': 'univariate',
                                                               'forecast_type': 'point', 'prediction_manifest': str(prediction / 'manifest.json'),
                                                               'context_length': metadata.get('context_limit', CONTEXT_LIMITS['chronos2']),
                                                               'fallback_counts': metadata.get('fallback_counts'), 'tsrag_weight': metadata.get('tsrag_weight'),
                                                               'retrieval': metadata.get('retrieval'), 'alignment_period': metadata.get('alignment_period'),
                                                               'nonfinite_fallback_count': metadata.get('nonfinite_fallback_count', 0)},
                                            inference_seconds=metadata['inference_seconds'],
                                            evaluation_grid_path=str(resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
                    self.finish(run, ['predictions.npz', 'metrics.npz', 'metrics_summary.json', 'config.json'])

    def report(self):
        from timebench.results.comparison import build_report

        inputs = [(task, method, self.evaluation(task, method), self.prediction(task, method))
                  for task in self.tasks for method in self.methods()]
        launch = os.getenv('TIME_LAUNCH_ID', 'manual')
        build_report(inputs, self.root.parent / 'reports' / 'tsrag' / launch, self.config)

    def run(self, stage):
        from timebench.pipeline.runtime_resources import log_selected_device
        selected_device = self.device if stage in {'vanilla', 'extract', 'predict'} else 'cpu'
        log_selected_device(selected_device, stage=stage, component='tsrag_time')
        log(f'stage={stage} tasks={len(self.tasks)} seed={self.seed} device={self.device} Slurm={os.getenv("SLURM_JOB_ID")}')
        getattr(self, stage)()
