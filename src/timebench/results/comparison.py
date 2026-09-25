"""Matched TIME metric, dispersion, fallback, and inference summaries."""

from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np

from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
from timebench.results.performance import write_performance_report


def build_report(inputs, destination, config):
    from timebench.pipeline.workflow import write_json, log

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    grouped = defaultdict(list)
    for task, method, evaluation, prediction in inputs:
        summary = json.loads((evaluation / 'metrics_summary.json').read_text())
        pred = json.loads((prediction / 'prediction.json').read_text())
        seasonal_root = resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate').parent
        seasonal = json.loads((seasonal_root / 'metrics_summary.json').read_text())
        row = {'dataset': task.dataset, 'term': task.term, 'method': method,
               'retrieval_scope': pred.get('retrieval', {}).get('scope'),
               'aligned_datastore': pred.get('retrieval', {}).get('aligned'),
               'neighbor_query_scale': pred.get('retrieval', {}).get('query_scale'),
               'retrieval_representation': pred.get('retrieval', {}).get('representation'),
               'alignment_period': pred.get('alignment_period'),
               'fallback_method': pred.get('fallback_method'),
               'inference_seconds': summary['inference_seconds'],
               'tsrag_weight': pred.get('tsrag_weight'), 'validation_dates': pred.get('validation_dates'),
               'fallback_count': pred.get('fallback_counts', {}).get('test', {}).get('eligible_rows', pred.get('nonfinite_fallback_count', 0)),
               'fallback_grid_rows': pred.get('fallback_counts', {}).get('test', {}).get('grid_rows', summary['evaluation_grid']['valid_values']),
               'fallback_split': pred.get('fallback_reason_split'),
               'fallback_reasons': json.dumps(pred.get('fallback_reasons', {}), sort_keys=True),
               'task_error': json.dumps(pred['task_error']) if pred.get('task_error') else '',
               'datastore_preprocessing_seconds': pred.get('datastore_preprocessing_seconds', 0)}
        row['fallback_rate'] = row['fallback_count'] / row['fallback_grid_rows'] if row['fallback_grid_rows'] else None
        if row['fallback_split'] == 'test' and sum(json.loads(row['fallback_reasons']).values()) != row['fallback_count']:
            raise ValueError(f'{task.dataset}/{task.term}/{method}: fallback reasons do not match evaluated fallbacks')
        for metric, values in summary['metrics'].items():
            expected = seasonal['metrics'][metric]['finite_values']
            if values['finite_values'] < expected:
                raise ValueError(f'{task.dataset}/{task.term}/{method}: {metric} loses Seasonal metric coverage')
            for field in ('mean', 'variance', 'std', 'dispersion_ddof', 'finite_values', 'evaluation_values', 'total_values'):
                row[f'{metric}_{field}'] = values[field]
        denominator = seasonal['metrics']['MASE']['mean']
        variance = seasonal['metrics']['MASE'].get('variance')
        mase = summary['metrics']['MASE']
        row['seasonal_MASE_mean'] = denominator
        row['seasonal_MASE_variance'] = variance
        row['scaled_MASE_mean'] = mase['mean'] / denominator if denominator and mase['mean'] is not None else None
        row['scaled_MASE_std'] = mase['std'] / denominator if denominator and mase['std'] is not None else None
        row['scaled_MASE_variance'] = mase['variance'] / denominator**2 if denominator and mase['variance'] is not None else None
        row['MASE_variance_ratio_to_seasonal'] = mase['variance'] / variance if variance and mase['variance'] is not None else None
        rows.append(row)
        grouped[method].append(row)
        sources.append({'dataset': task.dataset, 'term': task.term, 'method': method,
                        'evaluation_manifest': str(evaluation / 'manifest.json'),
                        'prediction_manifest': str(prediction / 'manifest.json'),
                        'seasonal_manifest': str(seasonal_root / 'manifest.json')})
    with (destination / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overall = {}
    for method, values in grouped.items():
        mase_values = [row['MASE_mean'] for row in values if row['MASE_mean'] is not None]
        scaled = [row['scaled_MASE_mean'] for row in values if row['scaled_MASE_mean'] is not None]
        per_dataset = defaultdict(list)
        for row in values:
            if row['scaled_MASE_mean'] is not None:
                per_dataset[row['dataset']].append(row['scaled_MASE_mean'])
        overall[method] = {'tasks': len(values), 'tasks_with_finite_MASE': len(mase_values),
                           'retrieval': {key: values[0][key] for key in
                                         ('retrieval_scope', 'aligned_datastore', 'neighbor_query_scale', 'retrieval_representation')},
                           'mean_task_MASE': float(np.mean(mase_values)) if mase_values else None,
                           'mean_task_scaled_MASE': float(np.mean(scaled)) if scaled else None,
                           'equal_dataset_scaled_MASE': float(np.mean([np.mean(v) for v in per_dataset.values()])) if per_dataset else None,
                           'summed_inference_seconds': sum(row['inference_seconds'] for row in values),
                           'summed_preprocessing_seconds': sum(row['datastore_preprocessing_seconds'] for row in values),
                           'fallback_count': sum(row['fallback_count'] for row in values),
                           'fallback_grid_rows': sum(row['fallback_grid_rows'] for row in values)}
        reason_counts = defaultdict(int)
        for row in values:
            for reason, count in json.loads(row['fallback_reasons']).items():
                reason_counts[reason] += count
        overall[method]['fallback_split'] = 'test' if any(row['fallback_split'] == 'test' for row in values) else None
        overall[method]['fallback_reasons'] = dict(sorted(reason_counts.items()))
        rates = [row['fallback_rate'] for row in values if row['fallback_rate'] is not None]
        overall[method]['mean_task_fallback_rate'] = float(np.mean(rates)) if rates else None
        count, support = overall[method]['fallback_count'], overall[method]['fallback_grid_rows']
        overall[method]['pooled_fallback_rate'] = count / support if support else None
        log(f'{method}: tasks={len(values)} mean_task_MASE={overall[method]["mean_task_MASE"]} '
            f'inference_seconds={overall[method]["summed_inference_seconds"]:.2f} fallbacks={overall[method]["fallback_count"]}')
    write_json(destination / 'comparison_summary.json', overall)
    horizons = {(task.dataset, task.term): task.prediction_length for task, *_ in inputs}
    performance_rows = [{
        'model': row['method'], 'dataset': row['dataset'].rsplit('/', 1)[0],
        'frequency': row['dataset'].rsplit('/', 1)[1], 'term': row['term'],
        'horizon_steps': horizons[row['dataset'], row['term']],
        'MASE': row['MASE_mean'], 'scaled_MASE': row['scaled_MASE_mean'],
        'MASE_std': row['MASE_std'], 'MASE_variance': row['MASE_variance'],
        'seasonal_MASE_variance': row['seasonal_MASE_variance'],
        'inference_seconds': row['inference_seconds'],
        'datastore_preprocessing_seconds': row['datastore_preprocessing_seconds'],
    } for row in rows]
    performance_artifacts = write_performance_report(
        performance_rows, destination / 'performance',
        reference='chronos2_max' if 'chronos2_max' in grouped else None,
        scaled_aggregation='arithmetic', inputs=sources)
    write_json(destination / 'report_manifest.json', {'schema_version': 1, 'status': 'completed',
                                                   'performance_artifacts': [path.relative_to(destination).as_posix()
                                                                             for path in performance_artifacts],
                                                   'experiment': 'tsrag', 'requested_config': config,
                                                   'fallback_split': 'test',
                                                   'selection': 'exact_scientific_configuration_selected_repeat', 'inputs': sources,
                                                   'task_dispersion': 'population_variance_over_finite_series_window_variate_cells'})
