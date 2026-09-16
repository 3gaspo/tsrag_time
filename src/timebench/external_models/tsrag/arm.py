"""Source-adapted TS-RAG Chronos-Bolt retrieval head for local inference.

The inference architecture is adapted, not copied verbatim, from
https://github.com/UConn-DSIS/TS-RAG
at commit 73ac807789d2e61b8a3dfc8514e3fc947fe185cc. Training, dataset,
retrieval-database, and pipeline code are omitted. The released ``moe`` head
 and forward computation are retained, with the TS-RAG-local pinned
 Chronos-Bolt source providing the subclassed backbone implementation. Only the
 released MoE ARM path is kept.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Config

from .chronos_bolt import (
    ChronosBoltModelForForecasting,
    ChronosBoltOutput,
)


class InstanceNorm(nn.Module):
    """TS-RAG's released constant-aware instance normalization."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(
                torch.nanmean(x, dim=-1, keepdim=True),
                nan=0.0,
            )
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(),
                nan=1.0,
            )
            is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else:
            loc, scale = loc_scale
        normalized = (x - loc) / scale
        is_constant = (
            torch.all(x == x[..., :1], dim=-1, keepdim=True)
            if loc_scale is None
            else scale == 1
        )
        normalized = torch.where(
            is_constant,
            torch.ones_like(normalized),
            normalized,
        )
        return normalized, (loc, scale)

    def inverse(
        self,
        x: torch.Tensor,
        loc_scale: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return torch.where(scale == 1, loc, x * scale + loc)


class ChronosBoltModelForForecastingWithRetrieval(
    ChronosBoltModelForForecasting
):
    """Released TS-RAG ARM with its mixture-of-experts retrieval fusion."""

    def __init__(self, config: T5Config, augment: str = "moe") -> None:
        if augment != "moe":
            raise ValueError("the local TS-RAG evaluator supports augment=moe only")
        super().__init__(config)
        self.augment = augment
        self.instance_norm = InstanceNorm()
        self.dropout = nn.Dropout(p=0.2)
        self.encode_mlp = nn.Sequential(
            nn.Linear(self.chronos_config.prediction_length, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )
        self.post_init()

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        del distances
        if retrieved_seq is None:
            raise ValueError("TS-RAG requires retrieved sequences")
        batch_size = context.shape[0]
        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context,
            mask=mask,
        )
        retrieved_seq, _ = self.instance_norm(retrieved_seq)
        retrieved_batch, retrieved_count, retrieved_length = retrieved_seq.shape
        if retrieved_batch != batch_size:
            raise ValueError("query and retrieval batch sizes differ")
        forecast_length = 64
        if self.chronos_config.prediction_length != forecast_length:
            raise ValueError("released TS-RAG ARM requires prediction_length=64")
        if retrieved_length <= forecast_length:
            raise ValueError("retrieved sequences must concatenate lookback and future")
        _, retrieved_y = retrieved_seq.split(
            (retrieved_length - forecast_length, forecast_length),
            dim=2,
        )
        sequence_output = self.decode(
            input_embeds,
            attention_mask,
            hidden_states,
        )

        retrieved_y_enc = torch.stack(
            [
                self.encode_mlp(retrieved_y[:, index, :].to(self.dtype))
                for index in range(retrieved_count)
            ],
            dim=1,
        )
        all_enc = torch.cat([sequence_output, retrieved_y_enc], dim=1)
        attention_output, _ = self.mha(all_enc, all_enc, all_enc)
        attention_output = all_enc + attention_output
        attention_output = attention_output + self.dropout(
            self.ffn(attention_output)
        )
        scores = torch.stack(
            [
                torch.sigmoid(self.gate_layer(attention_output[:, index, :]))
                for index in range(retrieved_count + 1)
            ],
            dim=1,
        )
        alpha = F.softmax(scores, dim=1)
        fused_sequence_output = torch.sum(alpha * attention_output, dim=1)
        sequence_output = sequence_output + self.dropout(
            fused_sequence_output
        ).unsqueeze(1)

        prediction_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *prediction_shape
        )
        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            if self.chronos_config.prediction_length < target.shape[-1]:
                raise ValueError("target exceeds the checkpoint forecast horizon")
            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)],
                    dim=-1,
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)],
                    dim=-1,
                )
            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2).sum(dim=-1).mean()
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*prediction_shape)
        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)

