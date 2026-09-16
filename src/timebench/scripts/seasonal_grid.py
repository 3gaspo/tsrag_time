"""Produce the optional Seasonal store using the current Improved TIME contract."""

from pathlib import Path
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path='../conf', config_name='experiment')
def main(config):
    import os
    import numpy as np
    from timebench.pipeline.workflow import Workflow, log
    from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
    from timebench.evaluation.metrics import seasonal_naive_point_forecast
    from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION, EVALUATION_GRID_FILE
    from timebench.evaluation.saver import save_window_predictions
    from timebench.evaluation.timing import EvaluationTimer
    from timebench.pipeline.runs import allocate_run
    from timebench.data.windows import Windows

    resolved = OmegaConf.to_container(config, resolve=True)
    if resolved['dataset_config'] is None:
        resolved['dataset_config'] = str(Path(__file__).parents[1] / 'config/datasets.yaml')
    workflow = Workflow(resolved)
    settings = load_dataset_config(Path(resolved['dataset_config']))
    tasks_root = Path(os.environ['TIME_SEASONAL_TASKS_ROOT'])
    for task in workflow.tasks:
        windows = Windows(task, workflow.storage)
        dataset = Dataset(task.dataset, term=task.term, prediction_length=task.prediction_length,
                          test_length=task.test_length, val_length=0, storage_path=workflow.storage,
                          to_univariate=windows.target(0).shape[0] > 1)
        # Match the source parent's shared Seasonal scientific manifest exactly.
        val_length = get_dataset_settings(task.dataset, task.term, settings)['val_length']
        run = allocate_run(tasks_root / 'seasonal_naive/univariate' / task.dataset / task.term,
                           experiment='foundation_models',
                           identity={'model': 'seasonal_naive', 'target_mode': 'univariate',
                                     'dataset': task.dataset.rpartition('/')[0], 'frequency': dataset.freq, 'term': task.term},
                           model_config={'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                           pipeline_config={'prediction_length': task.prediction_length, 'test_length': task.test_length,
                                            'val_length': val_length, 'windows': dataset.windows, 'seasonality': task.seasonality,
                                            'evaluation_grid': EVALUATION_GRID_DEFINITION},
                           runtime_config={'device': 'cpu'}, experiment_config={'covariate_mode': 'none', 'covariate_channels': 0},
                           provenance={'dataset_config_path': resolved['dataset_config']})
        with run:
            if not run.should_run:
                continue
            log(f'Seasonal dataset={task.dataset} term={task.term} H={task.prediction_length}')
            timer = EvaluationTimer()
            timer.start()
            forecasts = np.stack([seasonal_naive_point_forecast(np.asarray(entry['target']), task.prediction_length, task.seasonality)
                                  for entry in dataset.test_data.input])
            seconds = timer.stop()
            levels = run.manifest['model_config']['quantile_levels']
            quantiles = np.repeat(forecasts[:, None, :], len(levels), axis=1)
            save_window_predictions(dataset, quantiles, f'{task.dataset}/{task.term}', str(tasks_root), seasonality=task.seasonality,
                                    quantile_levels=levels, task_output_dir=str(run.run_dir), create_evaluation_grid=True,
                                    inference_seconds=seconds, model_hyperparams={'model': 'seasonal_naive', 'experiment': 'foundation_models',
                                                                               'target_mode': 'univariate', 'season_length': task.seasonality,
                                                                               'covariate_mode': 'none', 'covariate_channels': 0})
            workflow.finish(run, ['predictions.npz', 'metrics.npz', 'metrics_summary.json', 'config.json', EVALUATION_GRID_FILE])


if __name__ == '__main__':
    main()
