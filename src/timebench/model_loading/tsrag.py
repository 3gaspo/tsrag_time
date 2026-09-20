"""Compatibility loading for the pinned TS-RAG Chronos-Bolt checkpoint."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


TSRAG_ADAPTOR_PREFIXES = ("encode_mlp.", "mha.", "ffn.", "gate_layer.")


@dataclass(frozen=True)
class LoadedTSRAG:
    """The frozen upstream ARM; fallback forecasts are independent artifacts."""

    model: torch.nn.Module
    median_index: int
    adaptor_parameters: int
    checkpoint: Path


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    matches = sorted(path.rglob("best.pth")) if path.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one TS-RAG best.pth below {path}, found {len(matches)}"
        )
    return matches[0]


def load_tsrag(
    base_checkpoint: str | Path,
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> LoadedTSRAG:
    """Load the released MoE ARM and a matched frozen vanilla backbone."""

    from transformers import AutoConfig

    from timebench.external_models.tsrag.arm import (
        ChronosBoltModelForForecastingWithRetrieval,
    )

    base = Path(base_checkpoint).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Chronos-Bolt checkpoint directory not found: {base}")
    checkpoint_file = _checkpoint_file(Path(checkpoint))
    torch_device = torch.device(device)
    from timebench.pipeline.runtime_resources import log
    log("model_device", model="tsrag", device=str(torch_device))

    config = AutoConfig.from_pretrained(str(base), local_files_only=True)
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        str(base),
        config=config,
        augment="moe",
        local_files_only=True,
    )
    state = _torch_load(checkpoint_file)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint_file}")
    cleaned = OrderedDict(
        (str(key).removeprefix("module."), value) for key, value in state.items()
    )
    model.load_state_dict(cleaned, strict=True)

    for candidate in (model,):
        if int(candidate.chronos_config.context_length) < 512:
            raise ValueError("the TS-RAG backbone must accept its trained 512-point context")
        if int(candidate.chronos_config.prediction_length) != 64:
            raise ValueError("the released TS-RAG checkpoint requires prediction_length=64")
        candidate.to(torch_device).eval()
        for parameter in candidate.parameters():
            parameter.requires_grad = False

    adaptor_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(TSRAG_ADAPTOR_PREFIXES)
    )
    if adaptor_parameters <= 0:
        raise ValueError("TS-RAG checkpoint exposes no recognized ARM parameters")
    quantiles = model.quantiles.detach().float().cpu()
    median_index = int(torch.abs(quantiles - 0.5).argmin().item())
    return LoadedTSRAG(
        model=model,
        median_index=median_index,
        adaptor_parameters=int(adaptor_parameters),
        checkpoint=checkpoint_file,
    )
