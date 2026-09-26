"""One local scientific smoke: causal retrieval, mixing, fallback, and artifacts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from timebench.data.windows import Task, Windows, candidate_origins, query_origins, write_prepared
from timebench.evaluation.metrics import summarize_metric_values
from timebench.evaluation.grid import build_evaluation_grid, save_evaluation_grid
from timebench.external_models.tsrag.inference import CausalRetriever, forecast_query, instance_normalize, query_scaled_neighbors, normalized_l2
from timebench.model_loading.foundation import Forecaster
from timebench.pipeline.workflow import Workflow, CONTROLS, retrieval_methods
from timebench.pipeline.runs import load_manifest, RunHandle
from timebench.proposal.mixture import estimate_weight


OPTIONS = {'scope': 'all', 'aligned': False, 'query_scale': False, 'representation': 't5'}
AXES = {'scope': 'same_series', 'aligned': True,
        'representation': 'instance_l2', 'query_scale': True}


class ArrayIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors)
        self.index = self

    def add(self, vectors):
        self.vectors = np.concatenate((self.vectors, vectors))

    def search(self, vector, top_k):
        distances = np.sum((self.vectors[None] - vector[:, None])**2, axis=-1)
        ids = np.argsort(distances, axis=-1)[:, :top_k + 1][:, :-1]
        return np.take_along_axis(distances, ids, axis=-1), ids


class Encoder:
    def __init__(self, *args, **kwargs):
        pass

    def representation(self, tensor):
        values = torch.nan_to_num(tensor[:, 0]).mean(dim=-1)
        return values[:, None].repeat(1, 768)


class SyntheticWindows(Windows):
    def __init__(self, task, storage):
        self.task, self.source_path = task, Path(storage) / task.dataset
        t = np.arange(900, dtype=float)
        self.source = [{'target': np.stack((np.sin(t/9) + t/1000, np.cos(t/7) + t/800)), 'freq': 'D', 'start': '2000-01-01'}]
        self._targets, self.start_ticks, self.tick_step = {}, np.array([0]), 1


class SyntheticDataset:
    def __init__(self, *args, **kwargs):
        self.prediction_length = kwargs['prediction_length']
        self.windows = kwargs['test_length'] // self.prediction_length
        self.target_dim = 2
        self.freq = 'D'
        self.hf_dataset = SyntheticWindows(Task('synthetic/D', 'short', 4, 8, 8, 2), '.').source

    @property
    def test_data(self):
        values = self.hf_dataset[0]['target']
        origins = query_origins(values.shape[-1], self.windows*self.prediction_length, self.prediction_length)
        return [({'target': values[channel, :origin]}, {'target': values[channel, origin:origin+self.prediction_length]})
                for channel in range(2) for origin in origins]


class SyntheticForecaster:
    def __init__(self, alias, *args):
        self.alias = alias

    def forecast(self, histories, horizon, context_limit, **kwargs):
        return np.stack([np.repeat(history[-1], horizon) for history in histories]).astype(np.float32)


class Contract(unittest.TestCase):
    def test_test_datastore_includes_validation(self):
        task = Task('synthetic/D', 'short', 10, 100, 128, 2, datastore_stride=7,
                    max_datastore_windows=13)
        windows = SyntheticWindows(task, '.')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_root, test_root = root/'validation', root/'test'
            validation_root.mkdir()
            test_root.mkdir()
            write_prepared(windows, validation_root, 'validation')
            write_prepared(windows, test_root, 'test')
            refs = np.load(test_root/'datastore_references.npy')
            validation = np.load(validation_root/'validation_references.npy')
            test = np.load(test_root/'test_references.npy')
            # Select on validation first, then query the same observed series at test time.
            vectors = np.arange(len(refs), dtype=np.float32)[:, None]
            with patch('timebench.external_models.tsrag.retriever.TSRAGIndex', ArrayIndex):
                retrieval = CausalRetriever(refs, vectors, task.max_datastore_windows, windows=windows,
                                            options={**OPTIONS, 'scope': 'same_series'})
                retrieval.search(validation[0], np.zeros((1, 1)), np.zeros((1, 512)))
                validation_positions = retrieval.positions.copy()
                first_test = test[0]
                self.assertIsNotNone(retrieval.search(first_test, np.zeros((1, 1)), np.zeros((1, 512))))
                eligible = refs[retrieval.positions]
                self.assertEqual(len(eligible), task.max_datastore_windows)
                self.assertTrue(np.any(eligible[:, 2] >= validation[0, 2]))
                self.assertTrue(np.any(~np.isin(retrieval.positions, validation_positions)))
                expected = candidate_origins(int(first_test[2]), stride=task.datastore_stride,
                                             maximum=task.max_datastore_windows)
                np.testing.assert_array_equal(eligible[:, 2], expected)
                self.assertTrue(np.all(eligible[:, 2] + 64 <= first_test[2]))
                last_test = test[test[:, 1] == 0][-1]
                retrieval.search(last_test, np.zeros((1, 1)), np.zeros((1, 512)))
                self.assertTrue(np.any(refs[retrieval.positions, 2] >= first_test[2]))
                self.assertTrue(np.all(refs[retrieval.positions, 2] + 64 <= last_test[2]))
            # Disabling validation does not shrink the datastore available to testing.
            no_validation = root/'no_validation'
            no_validation.mkdir()
            write_prepared(SyntheticWindows(Task('synthetic/D', 'short', 10, 100, 0, 2,
                                                 datastore_stride=7, max_datastore_windows=13), '.'), no_validation, 'test')
            np.testing.assert_array_equal(refs, np.load(no_validation/'datastore_references.npy'))

    def test_causal_dates_and_rollout(self):
        self.assertEqual(candidate_origins(600).tolist(), list(range(512, 537)))
        self.assertEqual(candidate_origins(600, maximum=3).tolist(), [534, 535, 536])
        self.assertEqual(query_origins(1000, 11, 4).tolist(), [989, 993])
        windows = SyntheticWindows(Task('synthetic/D', 'short', 4, 8, 0, 2), '.')
        refs = np.array([(0, 0, date) for date in range(512, 550)] + [(0, 1, 512)])
        refs = refs[np.argsort(windows.ticks(refs), kind='stable')]
        vectors = np.arange(len(refs), dtype=np.float32)[:, None]
        with patch('timebench.external_models.tsrag.retriever.TSRAGIndex', ArrayIndex):
            retrieval = CausalRetriever(refs, vectors, windows=windows, options={**OPTIONS, 'scope': 'same_series'})
            self.assertIsNone(retrieval.search((0, 0, 585), np.zeros((1, 1)), np.zeros((1, 512))))
            found = retrieval.search((0, 0, 600), np.zeros((1, 1)), np.zeros((1, 512)))
            self.assertTrue(np.all(refs[found[1], 2] + 64 <= 600))
            self.assertEqual(len(retrieval.positions), 25)
            retrieval.search((0, 0, 604), np.zeros((1, 1)), np.zeros((1, 512)))
            self.assertEqual(len(retrieval.positions), 29)
        task = Task('synthetic/D', 'long', 130, 130, 0, 2)
        windows = SyntheticWindows(task, '.')
        lengths, cutoffs = [], []
        model = SimpleNamespace(model=lambda **kwargs: (lengths.append(kwargs['context'].shape[-1]) or
                                SimpleNamespace(quantile_preds=torch.zeros(1, 1, 64))), median_index=0)
        class Retrieval:
            options = OPTIONS
            references = np.array([(0, 0, 512)] * 10)
            def search(self, reference, representation, context):
                cutoffs.append(int(reference[2]))
                return np.zeros((1, 10), dtype=np.float32), np.arange(10)[None]
        retrieval = Retrieval()
        retrieval.ticks = windows.ticks(retrieval.references)
        prediction, reason, provenance, produced_nans = forecast_query(
            model, Encoder(), retrieval, windows, (0, 0, 700), np.zeros(768), 130, 'cpu')
        self.assertIsNone(reason)
        self.assertEqual(produced_nans, 0)
        self.assertEqual(provenance[0], 3)
        self.assertEqual(len(prediction), 130)
        self.assertEqual(lengths, [512, 512, 512])
        self.assertEqual(cutoffs, [700, 700, 700])

    def test_one_axis_ablation_and_cross_variate_calendar_alignment(self):
        methods = retrieval_methods({'retrieval': OPTIONS, 'retrieval_axes': AXES, 'ablation': True})
        self.assertEqual(methods, {
            'tsrag': OPTIONS,
            'tsrag_same_series': {**OPTIONS, 'scope': 'same_series'},
            'tsrag_aligned': {**OPTIONS, 'aligned': True},
            'tsrag_inst_l2': {**OPTIONS, 'representation': 'instance_l2'},
            'tsrag_query_scale': {**OPTIONS, 'query_scale': True},
        })
        self.assertEqual(retrieval_methods({'retrieval': OPTIONS, 'ablation': False}), {'tsrag': OPTIONS})
        task = Task('synthetic/15T', 'short', 4, 8, 0, 2, alignment_period=7)
        target = SyntheticWindows(task, '.').source[0]['target']
        source = [{'target': target, 'freq': '15min', 'start': '2000-01-01 00:00:00'},
                  {'target': target, 'freq': '15min', 'start': '2000-01-01 01:15:00'}]
        with patch.dict(sys.modules, {'datasets': SimpleNamespace(load_from_disk=lambda path: source)}):
            windows = Windows(task, '.')
        self.assertEqual(windows.tick_step, 15)
        self.assertEqual(windows.start_ticks[1] - windows.start_ticks[0], 75)
        reference = np.array([0, 0, 800])
        refs = windows.datastore(reference[None])
        query = np.asarray(windows.histories([reference], 512), dtype=np.float32)
        vectors = np.broadcast_to(instance_normalize(query), (len(refs), 512)).copy()
        # L2 cells must not initialize a T5 encoder or FAISS index.
        with patch('timebench.external_models.tsrag.retriever.TSRAGIndex', side_effect=AssertionError('L2 invoked FAISS')):
            cross = CausalRetriever(refs, vectors, windows=windows,
                                    options={**OPTIONS, 'representation': 'instance_l2', 'aligned': True},
                                    batch_size=31)
            found = cross.search(reference, None, query)
            self.assertIsNotNone(found)
            selected = refs[found[1][0]]
            tick = windows.ticks(reference[None])[0]
            self.assertTrue(np.all(windows.ticks(selected) + 64 * 15 <= tick))
            self.assertTrue(np.all((tick - windows.ticks(selected)) % (7 * 15) == 0))
            self.assertTrue(np.any((selected[:, 0] != 0) | (selected[:, 1] != 0)))
            single = CausalRetriever(refs, vectors, windows=windows,
                                     options={**OPTIONS, 'scope': 'same_series', 'representation': 'instance_l2'})
            selected = refs[single.search(reference, None, query)[1][0]]
            self.assertTrue(np.all(selected[:, :2] == (0, 0)))

    def test_query_scale_alignment_and_normalized_l2(self):
        trajectory = np.sin(np.arange(576, dtype=np.float32) / 17)
        query = (3 * trajectory[:512] - 12)[None]
        neighbors = np.stack((2 * trajectory + 50, 5 * trajectory - 7))
        aligned = query_scaled_neighbors(query, neighbors)
        np.testing.assert_allclose(aligned, np.broadcast_to(3 * trajectory - 12, aligned.shape), atol=1e-4)
        vectors = instance_normalize(neighbors[:, :512])
        distances = normalized_l2(instance_normalize(query)[0], vectors, 0.8)
        self.assertTrue(np.all(distances < 1e-6))
        damaged = vectors.copy()
        damaged[0, :200] = np.nan
        self.assertFalse(np.isfinite(normalized_l2(instance_normalize(query)[0], damaged, 0.8)[0]))

    def test_mixture_and_population_dispersion(self):
        refs = np.array([(0, 0, 10), (0, 1, 10), (0, 0, 12)])
        labels = np.zeros((3, 2))
        histories = [np.arange(512)] * 3
        result = estimate_weight(np.array([[2, 2], [0, 0], [1, 1]]),
                                 np.array([[1, 1], [0, 0], [1, 1]]), labels, histories, refs)
        self.assertEqual(result['validation_dates'], 2)
        self.assertEqual(result['tsrag_weight'], 2.5/4)
        self.assertEqual(estimate_weight([], [], [], [], [])['tsrag_weight'], 0)
        summary = summarize_metric_values(np.array([1., 3., np.nan]), 3)
        self.assertEqual(summary['variance'], 1.)
        self.assertEqual(summary['std'], 1.)
        self.assertEqual(summary['dispersion_ddof'], 0)

    def test_official_median_and_bolt_context_restoration(self):
        adapter = Forecaster.__new__(Forecaster)
        adapter.alias, adapter.supports_covariates = 'chronos_bolt', False
        config = SimpleNamespace(context_length=2048)
        lengths = []
        def predict(**kwargs):
            lengths.append((len(kwargs['inputs'][0]), config.context_length))
            return torch.ones(1, kwargs['prediction_length'], 1), torch.full((1, kwargs['prediction_length']), 99.)
        adapter.pipeline = SimpleNamespace(model=SimpleNamespace(chronos_config=config), predict_quantiles=predict)
        self.assertTrue(np.all(adapter.forecast([np.arange(3000)], 130, 512) == 1))
        adapter.forecast([np.arange(3000)], 130, 2048)
        self.assertEqual(lengths, [(512, 512), (2048, 2048)])
        self.assertEqual(config.context_length, 2048)
        with self.assertRaises(ValueError):
            adapter.forecast([np.arange(512)], 4, 512, covariates=[1])

    def test_pipeline_fallback_reuse_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = Workflow.__new__(Workflow)
            task = Task('synthetic/D', 'short', 4, 8, 8, 2)
            workflow.config = {'t5_samples': 20, 'embedding_batch_size': 16, 'minimum_overlap_fraction': 0.8,
                               'ablation': True, 'retrieval': OPTIONS, 'retrieval_axes': AXES}
            workflow.rag_methods = retrieval_methods(workflow.config)
            workflow.root, workflow.storage, workflow.weights = root/'tsrag', root/'datasets', root/'weights'
            workflow.vanilla_predictions_path = None
            workflow.tasks, workflow.seed, workflow.device, workflow.batch_size = [task], 0, 'cpu', 2
            grid = root/'seasonal'/'run_0'/'evaluation_grid.npz'
            grid.parent.mkdir(parents=True)
            # Use the maintained saver to create exactly the same support and summaries.
            from timebench.evaluation.saver import save_window_predictions
            dataset = SyntheticDataset(prediction_length=4, test_length=8)
            data_module = SimpleNamespace(Dataset=SyntheticDataset)
            forecasts = np.stack([np.repeat(inp['target'][-1], 4) for inp, label in dataset.test_data])
            save_window_predictions(dataset, forecasts[:, None], 'synthetic/D/short', str(grid.parent),
                                    seasonality=2, quantile_levels=[0.5], task_output_dir=str(grid.parent),
                                    inference_seconds=0, create_evaluation_grid=True)
            (grid.parent/'manifest.json').write_text('{}')
            def query(*args, **kwargs):
                return np.full(4, np.nan, dtype=np.float32), None, np.zeros(4), 4
            with patch.dict(os.environ, {'TIME_LAUNCH_ID': 'synthetic', 'TSRAG_DEFER_COMPLETION': '0'}), \
                 patch('timebench.pipeline.workflow.Windows', SyntheticWindows), \
                 patch('timebench.model_loading.foundation.Forecaster', SyntheticForecaster), \
                 patch('timebench.external_models.tsrag.retriever.TSRAGRetriever', Encoder), \
                 patch('timebench.external_models.tsrag.inference.forecast_query', query), \
                 patch('timebench.external_models.tsrag.retriever.TSRAGIndex', ArrayIndex), \
                 patch('timebench.model_loading.tsrag.load_tsrag', lambda *a, **kw: SimpleNamespace()), \
                 patch('timebench.pipeline.workflow.resolve_shared_evaluation_grid', lambda *a: grid), \
                 patch('timebench.results.comparison.resolve_shared_evaluation_grid', lambda *a: grid), \
                 patch.dict(sys.modules, {'timebench.evaluation.data': data_module}):
                workflow.prepare()
                workflow.vanilla()
                workflow.extract()
                workflow.predict()
                workflow.mix()
                workflow.evaluate()
                workflow.report()
                rag = workflow.rag(task, 'test')
                bolt = workflow.raw(task, 'test', 'chronos_bolt_max')
                for method in workflow.rag_methods:
                    prediction = workflow.rag(task, 'test', method)
                    np.testing.assert_array_equal(np.load(prediction/'prediction.npy'), np.load(bolt/'prediction.npy'))
                    self.assertTrue(np.load(prediction/'fallback.npy').all())
                    metadata = json.loads((prediction/'prediction.json').read_text())
                    self.assertEqual(metadata['fallback_reason_split'], 'test')
                    self.assertEqual(sum(metadata['fallback_reasons'].values()), metadata['fallback_counts']['test']['eligible_rows'])
                np.testing.assert_array_equal(np.load(rag/'prediction.npy'), np.load(bolt/'prediction.npy'))
                self.assertTrue(np.load(rag/'fallback.npy').all())
                summary = json.loads((workflow.evaluation(task, 'tsrag')/'metrics_summary.json').read_text())
                self.assertIsNotNone(summary['metrics']['MASE']['variance'])
                self.assertEqual(summary['metrics']['MASE']['dispersion_ddof'], 0)
                before = list(workflow.root.rglob('run_0/manifest.json'))
                workflow.prepare()
                workflow.vanilla()
                self.assertEqual(len(before), len(list(workflow.root.rglob('run_0/manifest.json'))))
                self.assertFalse(list(workflow.root.rglob('run_1')))
                report = json.loads((workflow.root.parent/'reports/tsrag/synthetic/report_manifest.json').read_text())
                self.assertEqual(len(report['inputs']), len(workflow.methods()))
                aggregate = json.loads((workflow.root.parent/'reports/tsrag/synthetic/comparison_summary.json').read_text())
                self.assertEqual(aggregate['tsrag']['pooled_fallback_rate'], 1.)
                self.assertEqual(aggregate['tsrag']['mean_task_fallback_rate'], 1.)
                self.assertEqual(aggregate['tsrag']['fallback_split'], 'test')
                rag_metadata = json.loads((rag/'prediction.json').read_text())
                self.assertEqual(sum(aggregate['tsrag']['fallback_reasons'].values()),
                                 rag_metadata['fallback_counts']['test']['eligible_rows'])
                self.assertEqual(load_manifest(rag)['status'], 'completed')
                # The scheduler completion contract remains deferred until after srun.
                with patch.dict(os.environ, {'TSRAG_DEFER_COMPLETION': '1'}):
                    with workflow.allocate(task, 'test_completion', 'shared') as pending:
                        (pending.run_dir/'payload.txt').write_text('ready')
                        workflow.finish(pending, ['payload.txt'])
                    self.assertEqual(load_manifest(pending.run_dir)['status'], 'computed')
                    from timebench.scripts.finalize_stage import main
                    with patch.object(sys, 'argv', ['finalize', str(workflow.root), 'synthetic']):
                        main()
                    self.assertEqual(load_manifest(pending.run_dir)['status'], 'completed')

    def test_source_and_command_scope(self):
        imported = set()
        for path in (ROOT/'src').rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            if ROOT/'src/timebench' in path.parents:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name.split('.')[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        imported.add(node.module.split('.')[0])
        distribution = {'chronos': 'chronos-forecasting', 'datasets': 'datasets',
                        'einops': 'einops', 'faiss': 'faiss-cpu', 'gluonts': 'gluonts',
                        'hydra': 'hydra-core', 'numpy': 'numpy', 'omegaconf': 'omegaconf',
                        'matplotlib': 'matplotlib',
                        'packaging': 'packaging', 'pandas': 'pandas', 'pyarrow': 'pyarrow',
                        'dotenv': 'python-dotenv', 'yaml': 'PyYAML', 'toolz': 'toolz',
                        'torch': 'torch', 'transformers': 'transformers'}
        third_party = imported - sys.stdlib_module_names - {'timebench'}
        dependencies = tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']
        declared = {re.split(r'[<>=!~\[]', dependency)[0] for dependency in dependencies}
        self.assertEqual(declared, {distribution[name] for name in third_party})
        self.assertEqual(set(CONTROLS), {'chronos_t5_512', 'chronos_bolt_512', 'chronos_bolt_max', 'chronos2_max'})
        self.assertFalse((ROOT/'src/timebench/adaptime').exists())
        self.assertFalse((ROOT/'src/timebench/models').exists())
        self.assertFalse((ROOT/'experiments').exists())
        for name in ('submit_experiment.sh', 'submit_seasonal_naive.sh', 'submit_ablation.sh'):
            self.assertTrue((ROOT/'scripts'/name).is_file())
            self.assertFalse((ROOT/name).exists())
        ablation = (ROOT/'scripts/submit_ablation.sh').read_text()
        self.assertIn('src/slurm/submit_experiment.sh', ablation)
        self.assertIn('TSRAG_EXPERIMENT_FAMILY=ablation', ablation)
        self.assertNotIn('exit 2', ablation)
        runtime = (ROOT/'src/slurm/runtime_paths.sh').read_text()
        seasonal_submit = (ROOT/'src/slurm/submit_seasonal.sh').read_text()
        self.assertIn('OUTPUTS_ROOT="${OUTPUTS_ROOT:-${TIME_OUTPUTS:-$default_outputs_root}}"', runtime)
        self.assertIn('LOGS_ROOT=$TIME_SEASONAL_LOGS_ROOT', seasonal_submit)
        required = ('--gres=gpu:1', '--partition=an', '--qos=an_preemptable', '--exclusive',
                    '--wckey=P12CU:DATASCIENCE', '--ntasks=1')
        for front in ROOT.glob('*_selena.slurm'):
            text = front.read_text()
            for directive in required:
                self.assertIn(f'#SBATCH {directive}', text)
            self.assertIn('/codes/tsrag_time/logs/%x_%j', text)
        # Bash is the installed Git shell; this requires no project environment.
        bash = Path(r'C:\Program Files\Git\bin\bash.exe')
        if bash.exists():
            files = list(ROOT.glob('*.sh')) + list(ROOT.glob('*.slurm')) + list((ROOT/'src/slurm').glob('*.sh'))
            files += list((ROOT/'scripts').glob('*.sh'))
            for path in files:
                subprocess.run([str(bash), '-n', path.as_posix()], check=True)

    def test_runtime_environment_precedence(self):
        bash = Path(r'C:\Program Files\Git\bin\bash.exe')
        if not bash.exists():
            self.skipTest('Git Bash is not available on this execution host')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.env').write_text(f'TIME_DATASET="{root.as_posix()}/configured_arrow"\n')
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(('TIME_', 'HF_', 'TRANSFORMERS_', 'TORCH_'))
                           and key not in ('OUTPUTS_ROOT', 'LOGS_ROOT', 'SELENA_NNI')}
            environment.update(TIME_OUTPUTS='/foreign/outputs', OUTPUTS_ROOT='/foreign/outputs',
                               TIME_LOGS='/foreign/logs', LOGS_ROOT='/foreign/logs', PROJECT_ROOT=root.as_posix(), TIME_STORAGE_ROOT=root.as_posix(),
                               TSRAG_RUNTIME_SCRIPT=(ROOT/'src/slurm/runtime_paths.sh').as_posix())
            command = 'source "$TSRAG_RUNTIME_SCRIPT"\nprintf "%s" "$TIME_DATASET"'
            result = subprocess.run([str(bash), '-c', command], env=environment, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines()[-1], f'{root.as_posix()}/configured_arrow')
            environment['TIME_DATASET'] = f'{root.as_posix()}/submitted_arrow'
            result = subprocess.run([str(bash), '-c', command], env=environment, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines()[-1], environment['TIME_DATASET'])
            paths_command = command.replace('printf "%s"', 'printf "%s\\n%s"').replace(
                '"$TIME_DATASET"', '"$TIME_OUTPUTS" "$TIME_LOGS"')
            result = subprocess.run([str(bash), '-c', paths_command], env=environment, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines()[-2:], ['/foreign/outputs', '/foreign/logs'])
            environment.update(SELENA_NNI='H12345', TIME_SCRATCH_ROOT='/foreign/project')
            result = subprocess.run([str(bash), '-c', 'mkdir() { :; }\n' + paths_command], env=environment,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines()[-2:], ['/foreign/outputs', '/foreign/logs'])
            for key in ('TIME_OUTPUTS', 'OUTPUTS_ROOT', 'TIME_LOGS', 'LOGS_ROOT'):
                environment.pop(key)
            result = subprocess.run([str(bash), '-c', 'mkdir() { :; }\n' + paths_command], env=environment,
                                    capture_output=True, text=True, check=True)
            scratch = f'/scratch/users/h12345/codes/{root.name}'
            self.assertEqual(result.stdout.splitlines()[-2:], [f'{scratch}/outputs', f'{scratch}/logs'])


if __name__ == '__main__':
    unittest.main()
