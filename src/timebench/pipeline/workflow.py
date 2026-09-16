"""Stage orchestration over independent, exact-configuration TIME task runs."""

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
import json
import os
import random

import numpy as np

from timebench.data.windows import Task, Windows, write_prepared, CONTEXT_LENGTH, NATIVE_HORIZON
from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION, flatten_univariate_grid, load_evaluation_grid
from timebench.evaluation.timing import EvaluationTimer
from timebench.model_loading.foundation import CONTEXT_LIMITS, CHECKPOINTS
from timebench.paths import dataset_storage_root, outputs_root, weights_root
from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
from timebench.pipeline.runs import allocate_run, load_manifest, select_completed_runs

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
    axis_names = ('scope', 'aligned', 'query_scale', 'representation')
    if config['ablation']:
        axes = config['retrieval_grid']
        for values in product(*(axes[name] for name in axis_names)):
            options = dict(zip(axis_names, values))
            if options != base:
                scope, aligned, scale, representation = values
                name = f'tsrag_{scope}_{"aligned" if aligned else "unaligned"}_{"query_scale" if scale else "raw"}_{representation}'
                methods[name] = options
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

    def science(self, task, phase, method, dependencies):
        pipe = {'task': task.config(), 'target_mode': 'univariate', 'covariates': 'none',
                'datastore_policy': 'growing_cross_variate_calendar_causal_complete_64_step_future_before_real_query',
                'dependencies': {name: {key: load_manifest(path)[key] for key in
                                       ('schema_version', 'identity', 'model_config', 'pipeline_config', 'experiment_config')}
                                 for name, path in dependencies.items()}}
        if phase == 'evaluations':
            pipe['evaluation_grid'] = EVALUATION_GRID_DEFINITION
        if phase == 'extractions':
            pipe['representations'] = sorted({'instance_l2'} | {options['representation'] for options in self.rag_methods.values()})
            pipe['t5_query_scaled_candidates'] = 'encoded_at_query_time'
        model = self.model_config(method)
        if phase == 'extractions':
            model = {'method': 'retrieval_representations', 'representations': pipe['representations'],
                     'context_length': CONTEXT_LENGTH, 't5_checkpoint': 'chronos-t5-base',
                     'instance_normalization': 'lookback_nanmean_nanstd_eps_1e-8'}
        return {'model_config': model, 'pipeline_config': pipe,
                'experiment_config': {'seed': self.seed}}

    def allocate(self, task, phase, method, dependencies=None):
        dependencies = dependencies or {}
        return allocate_run(self.path(task, phase, method), experiment='tsrag', identity=self.identity(task, method),
                            **self.science(task, phase, method, dependencies),
                            runtime_config={'device': self.device, 'batch_size': self.batch_size,
                                            'embedding_batch_size': int(self.config['embedding_batch_size'])},
                            provenance={'source_revisions': SOURCE_REVISIONS, 'dataset_path': str(self.storage / task.dataset),
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
            run._completed = True  # Owning srun must return successfully before finalization.
        else:
            run.complete(files)

    def prepared(self, task):
        return self.resolve(task, 'data', 'shared')

    def raw(self, task, method):
        data = self.prepared(task)
        return self.resolve(task, 'predictions', method, {'data': data})

    def extraction(self, task):
        data = self.prepared(task)
        return self.resolve(task, 'extractions', 'tsrag', {'data': data})

    def rag(self, task, method='tsrag'):
        deps = {'data': self.prepared(task), 'extraction': self.extraction(task), 'fallback': self.raw(task, 'chronos_bolt_max')}
        return self.resolve(task, 'predictions', method, deps)

    def mixture(self, task):
        deps = {'data': self.prepared(task), 'chronos2': self.raw(task, 'chronos2_max'), 'tsrag': self.rag(task),
                'fallback': self.raw(task, 'chronos_bolt_max')}
        return self.resolve(task, 'predictions', 'bayes_mixture', deps)

    def refs(self, data, split):
        return np.load(data / f'{split}_references.npy', mmap_mode='r')

    def support(self, task, split, windows, refs):
        if split == 'test':
            targets, cells = flatten_univariate_grid(*load_evaluation_grid(
                resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
            if targets.shape != (len(refs), task.prediction_length):
                raise ValueError('Seasonal grid and official test references do not align')
            if not np.array_equal(targets, np.isfinite(windows.labels(refs))):
                raise ValueError('Seasonal grid and current labels have different finite support')
            return targets, cells
        targets = np.isfinite(windows.labels(refs))
        return targets, targets.any(axis=-1)

    def prepare(self):
        for task in self.tasks:
            log(f'prepare dataset={task.dataset} term={task.term} H={task.prediction_length} validation={task.validation_length}')
            with self.allocate(task, 'data', 'shared') as run:
                if run.should_run:
                    self.finish(run, write_prepared(Windows(task, self.storage), run.run_dir))

    def vanilla(self):
        from timebench.model_loading.foundation import Forecaster

        # One backbone at a time; Bolt's two context controls share a loaded checkpoint.
        for alias in ('chronos_bolt', 'chronos_t5', 'chronos2'):
            model = None
            for method, (backbone, limit) in CONTROLS.items():
                if alias != backbone:
                    continue
                for task in self.tasks:
                    data = self.prepared(task)
                    with self.allocate(task, 'predictions', method, {'data': data}) as run:
                        if not run.should_run:
                            continue
                        if model is None:
                            model = Forecaster(alias, self.weights, self.device)
                        log(f'vanilla method={method} dataset={task.dataset} term={task.term} L={limit} H={task.prediction_length}')
                        seed_run(self.seed)
                        windows = Windows(task, self.storage)
                        timing = {}
                        files = []
                        for split in ('validation', 'test'):
                            refs = self.refs(data, split)
                            prediction = np.lib.format.open_memmap(run.run_dir / f'{split}.npy', mode='w+', dtype=np.float32,
                                                                  shape=(len(refs), task.prediction_length))
                            timer = EvaluationTimer()
                            timer.start()
                            for start in range(0, len(refs), self.batch_size):
                                rows = refs[start:start + self.batch_size]
                                prediction[start:start + len(rows)] = model.forecast(windows.histories(rows, limit), task.prediction_length,
                                                                                   limit, t5_samples=int(self.config['t5_samples']))
                            timing[f'{split}_inference_seconds'] = timer.stop()
                            prediction.flush()
                            targets, cells = self.support(task, split, windows, refs)
                            if np.any(cells & ~np.all(~targets | np.isfinite(prediction), axis=-1)):
                                raise ValueError(f'Non-finite {method} on required {split} support')
                            np.save(run.run_dir / f'{split}_context_length.npy',
                                    np.minimum(refs[:, 2], limit), allow_pickle=False)
                            files.append(f'{split}.npy')
                            files.append(f'{split}_context_length.npy')
                        write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'context_limit': limit,
                                                                  'context_policy': 'all_available_history_capped_at_limit', **timing})
                        self.finish(run, [*files, 'prediction.json'])
            del model

    def extract(self):
        import torch
        from timebench.external_models.tsrag.retriever import TSRAGRetriever
        from timebench.external_models.tsrag.inference import instance_normalize

        encoder = None
        representations = {'instance_l2'} | {options['representation'] for options in self.rag_methods.values()}
        for task in self.tasks:
            data = self.prepared(task)
            with self.allocate(task, 'extractions', 'tsrag', {'data': data}) as run:
                if not run.should_run:
                    continue
                started = perf_counter()
                files, errors, seconds = [], {}, {}
                windows = Windows(task, self.storage)
                for representation in sorted(representations):
                    representation_start = perf_counter()
                    errors[representation] = None
                    representation_files = []
                    try:
                        if representation == 't5' and encoder is None:
                            encoder = TSRAGRetriever(self.weights / 'chronos-t5-base', device_map=self.device)
                        for split in ('datastore', 'validation', 'test'):
                            refs = self.refs(data, split)
                            width = 768 if representation == 't5' else CONTEXT_LENGTH
                            filename = f'{split}_{representation}.npy'
                            embeddings = np.lib.format.open_memmap(run.run_dir / filename, mode='w+', dtype=np.float32,
                                                                  shape=(len(refs), width))
                            batch_size = int(self.config['embedding_batch_size'])
                            for start in range(0, len(refs), batch_size):
                                rows = refs[start:start + batch_size]
                                histories = windows.histories(rows, CONTEXT_LENGTH)
                                values = np.full((len(histories), CONTEXT_LENGTH), np.nan, dtype=np.float32)
                                for index, history in enumerate(histories):
                                    values[index, -len(history):] = history
                                if representation == 't5':
                                    block = encoder.representation(torch.from_numpy(values[:, None])).detach().float().cpu().numpy()
                                    if not np.isfinite(block).all():
                                        raise ValueError(f'Non-finite {split} retrieval embedding')
                                else:
                                    block = instance_normalize(values)
                                embeddings[start:start + len(rows)] = block
                            embeddings.flush()
                            representation_files.append(filename)
                        files.extend(representation_files)
                    except Exception as exception:
                        errors[representation] = {'type': type(exception).__name__, 'message': str(exception)}
                        log(f'TS-RAG {representation} extraction fallback {task.dataset}/{task.term}: {errors[representation]}')
                    seconds[representation] = perf_counter() - representation_start
                write_json(run.run_dir / 'extraction.json', {'schema_version': 1, 'errors': errors,
                                                          'representation_seconds': seconds,
                                                          'extraction_seconds': perf_counter() - started})
                self.finish(run, [*files, 'extraction.json'])

    def predict(self):
        from timebench.external_models.tsrag.retriever import TSRAGRetriever
        from timebench.external_models.tsrag.inference import CausalRetriever, forecast_query
        from timebench.model_loading.tsrag import load_tsrag

        loaded = encoder = None
        for method, options in self.rag_methods.items():
            for task in self.tasks:
                data, extraction, fallback = self.prepared(task), self.extraction(task), self.raw(task, 'chronos_bolt_max')
                deps = {'data': data, 'extraction': extraction, 'fallback': fallback}
                with self.allocate(task, 'predictions', method, deps) as run:
                    if not run.should_run:
                        continue
                    log(f'predict method={method} dataset={task.dataset} term={task.term} retrieval={options} '
                        f'alignment_period={task.alignment_period} fallback=chronos_bolt_max')
                    seed_run(self.seed)
                    windows = Windows(task, self.storage)
                    metadata = json.loads((extraction / 'extraction.json').read_text())
                    task_error = metadata['errors'][options['representation']]
                    if task_error is None:
                        try:
                            if loaded is None:
                                loaded = load_tsrag(self.weights / 'chronos-bolt-base', self.weights / 'ts-rag', device=self.device)
                            if options['representation'] == 't5' and encoder is None:
                                encoder = TSRAGRetriever(self.weights / 'chronos-t5-base', device_map=self.device)
                        except Exception as exception:
                            task_error = {'type': type(exception).__name__, 'message': str(exception)}
                    files = []
                    timing, counts, reason_counts = {}, {}, {}
                    for split in ('validation', 'test'):
                        refs = self.refs(data, split)
                        base = np.load(fallback / f'{split}.npy', mmap_mode='r')
                        values = np.lib.format.open_memmap(run.run_dir / f'{split}.npy', mode='w+', dtype=np.float32, shape=base.shape)
                        values[:] = base
                        mask = np.zeros(len(refs), dtype=bool)
                        reasons = []
                        targets, cells = self.support(task, split, windows, refs)
                        retrieval = None
                        if task_error is None:
                            retrieval = CausalRetriever(self.refs(data, 'datastore'), np.load(extraction / f"datastore_{options['representation']}.npy", mmap_mode='r'),
                                                        task.max_datastore_windows, windows=windows, options=options, encoder=encoder,
                                                        batch_size=int(self.config['embedding_batch_size']),
                                                        minimum_overlap=float(self.config['minimum_overlap_fraction']))
                            embeddings = (np.load(extraction / f'{split}_t5.npy', mmap_mode='r')
                                          if options['representation'] == 't5' else None)
                        timer = EvaluationTimer()
                        timer.start()
                        for row, reference in enumerate(refs):
                            reason = 'extraction_or_model_error' if task_error else None
                            if reason is None:
                                try:
                                    forecast, reason = forecast_query(loaded, encoder, retrieval, windows, reference,
                                                                      embeddings[row] if embeddings is not None else None,
                                                                      task.prediction_length, self.device)
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
                                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                                if reason != 'inference_error':
                                    reasons.append({'row': row, 'reason': reason})
                        timing[f'{split}_inference_seconds'] = timer.stop()
                        values.flush()
                        np.save(run.run_dir / f'{split}_fallback.npy', mask, allow_pickle=False)
                        write_json(run.run_dir / f'{split}_fallback_reasons.json', reasons)
                        counts[split] = {'all_rows': int(mask.sum()), 'eligible_rows': int((mask & cells).sum()),
                                         'grid_rows': int(cells.sum()), 'total_rows': len(refs)}
                        files.extend([f'{split}.npy', f'{split}_fallback.npy', f'{split}_fallback_reasons.json'])
                    write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'context_limit': 512, 'retrieval': options,
                                                               'alignment_period': task.alignment_period,
                                                               'fallback_method': 'chronos_bolt_max', 'task_error': task_error,
                                                               'fallback_counts': counts, 'fallback_reasons': reason_counts,
                                                               'datastore_preprocessing_seconds': metadata['representation_seconds'][options['representation']],
                                                               'fallback_source_test_inference_seconds': json.loads((fallback / 'prediction.json').read_text())['test_inference_seconds'],
                                                               'timing_policy': 'measured_query_work_with_precomputed_bolt_max_fallback', **timing})
                    self.finish(run, [*files, 'prediction.json'])

    def mix(self):
        from timebench.proposal.mixture import estimate_weight, mix

        for task in self.tasks:
            data, c2, rag = self.prepared(task), self.raw(task, 'chronos2_max'), self.rag(task)
            fallback = self.raw(task, 'chronos_bolt_max')
            with self.allocate(task, 'predictions', 'bayes_mixture', {'data': data, 'chronos2': c2, 'tsrag': rag, 'fallback': fallback}) as run:
                if not run.should_run:
                    continue
                windows = Windows(task, self.storage)
                refs = self.refs(data, 'validation')
                weight = estimate_weight(np.load(c2 / 'validation.npy'), np.load(rag / 'validation.npy'), windows.labels(refs),
                                         windows.histories(refs, 512), refs)
                write_json(run.run_dir / 'weight.json', weight)
                c2_test, rag_test = np.load(c2 / 'test.npy', mmap_mode='r'), np.load(rag / 'test.npy', mmap_mode='r')
                timer = EvaluationTimer()
                timer.start()
                prediction = mix(c2_test, rag_test, weight['tsrag_weight'])
                inference_seconds = timer.stop()
                targets, cells = self.support(task, 'test', windows, self.refs(data, 'test'))
                invalid = cells & ~np.all(~targets | np.isfinite(prediction), axis=-1)
                prediction[invalid] = np.load(fallback / 'test.npy', mmap_mode='r')[invalid]
                np.save(run.run_dir / 'test.npy', prediction, allow_pickle=False)
                np.save(run.run_dir / 'test_fallback.npy', invalid, allow_pickle=False)
                component_seconds = sum(json.loads((path / 'prediction.json').read_text())['test_inference_seconds'] for path in (c2, rag))
                write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': 'bayes_mixture', **weight,
                                                           'fallback_method': 'chronos_bolt_max',
                                                           'test_inference_seconds': component_seconds + inference_seconds,
                                                           'mix_only_seconds': inference_seconds, 'nonfinite_fallback_count': int(invalid.sum())})
                self.finish(run, ['test.npy', 'test_fallback.npy', 'weight.json', 'prediction.json'])

    def methods(self):
        return [*CONTROLS, *self.rag_methods, 'bayes_mixture']

    def prediction(self, task, method):
        return self.rag(task, method) if method in self.rag_methods else self.mixture(task) if method == 'bayes_mixture' else self.raw(task, method)

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
                    values = np.load(prediction / 'test.npy', mmap_mode='r')
                    save_window_predictions(dataset, values[:, None, :], f'{task.dataset}/{task.term}', str(self.root),
                                            seasonality=task.seasonality, quantile_levels=[0.5], task_output_dir=str(run.run_dir),
                                            model_hyperparams={'model': method, 'experiment': 'tsrag', 'target_mode': 'univariate',
                                                               'forecast_type': 'point', 'prediction_manifest': str(prediction / 'manifest.json'),
                                                               'context_length': metadata.get('context_limit', CONTEXT_LIMITS['chronos2']),
                                                               'fallback_counts': metadata.get('fallback_counts'), 'tsrag_weight': metadata.get('tsrag_weight'),
                                                               'retrieval': metadata.get('retrieval'), 'alignment_period': metadata.get('alignment_period'),
                                                               'nonfinite_fallback_count': metadata.get('nonfinite_fallback_count', 0)},
                                            inference_seconds=metadata['test_inference_seconds'],
                                            evaluation_grid_path=str(resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
                    self.finish(run, ['predictions.npz', 'metrics.npz', 'metrics_summary.json', 'config.json'])

    def report(self):
        from timebench.results.comparison import build_report

        inputs = [(task, method, self.evaluation(task, method), self.prediction(task, method))
                  for task in self.tasks for method in self.methods()]
        launch = os.getenv('TIME_LAUNCH_ID', 'manual')
        build_report(inputs, self.root / 'reports' / launch, self.config)

    def run(self, stage):
        log(f'stage={stage} tasks={len(self.tasks)} seed={self.seed} device={self.device} Slurm={os.getenv("SLURM_JOB_ID")}')
        getattr(self, stage)()
