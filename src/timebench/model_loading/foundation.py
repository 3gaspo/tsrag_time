"""Official Chronos invocation with explicit context and median forecasts."""

from pathlib import Path
import numpy as np

MODEL_ALIASES = ('chronos_t5', 'chronos_bolt', 'chronos2')
CONTEXT_LIMITS = {'chronos_t5': 512, 'chronos_bolt': 2048, 'chronos2': 8192}
CHECKPOINTS = {'chronos_t5': 'chronos-t5-base', 'chronos_bolt': 'chronos-bolt-base', 'chronos2': 'chronos2'}


def numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class Forecaster:
    def __init__(self, alias, weights, device):
        from chronos import BaseChronosPipeline

        if alias not in MODEL_ALIASES:
            raise ValueError(f'Expected one of {MODEL_ALIASES}')
        self.alias = alias
        self.supports_covariates = alias == 'chronos2'
        self.pipeline = BaseChronosPipeline.from_pretrained(
            str(Path(weights) / CHECKPOINTS[alias]), device_map=device, local_files_only=True)

    def forecast(self, histories, horizon, context_limit, *, t5_samples=20, covariates=None):
        import torch

        if covariates is not None and len(covariates):
            if not self.supports_covariates:
                raise ValueError(f'{self.alias} does not support covariates')
            raise ValueError('This experiment explicitly selects univariate forecasts without covariates')
        inputs = [torch.as_tensor(np.asarray(history[-context_limit:], dtype=np.float32)) for history in histories]
        kwargs = {'limit_prediction_length': False}
        if self.alias == 'chronos_t5':
            kwargs['num_samples'] = t5_samples
        if self.alias == 'chronos2':
            kwargs.update(context_length=context_limit, cross_learning=False)
        # Bolt's official autoregressive quantile rollout consults this limit at
        # every chunk. Truncating only the initial input would grow Bolt-512's
        # context beyond 512 during long-horizon prediction.
        bolt_config = self.pipeline.model.chronos_config if self.alias == 'chronos_bolt' else None
        original_limit = bolt_config.context_length if bolt_config else None
        if bolt_config:
            bolt_config.context_length = context_limit
        try:
            with torch.inference_mode():
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=inputs, prediction_length=horizon, quantile_levels=[0.5], **kwargs)
        finally:
            if bolt_config:
                bolt_config.context_length = original_limit
        if self.alias == 'chronos2':
            values = np.stack([numpy(row)[0, :, 0] for row in quantiles])
        else:
            values = numpy(quantiles)[:, :, 0]
        if values.shape != (len(histories), horizon):
            raise ValueError(f'Unexpected {self.alias} forecast shape {values.shape}')
        return np.asarray(values, dtype=np.float32)
