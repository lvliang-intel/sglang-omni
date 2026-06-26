# SPDX-License-Identifier: Apache-2.0
"""Resolve the active quantization method's weight preprocessor for a checkpoint.

This is model-agnostic: given a model/checkpoint config, it resolves the
registered :class:`QuantizationMethod` via the quantization registry and exposes
its ``preprocess_weights`` hook so weight-loading paths exercise the unified
quantization abstraction instead of hard-coded per-model helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

    from sglang_omni.quantization.base import QuantizationMethod

# A weight preprocessor maps ``(target_name, loaded_weight) -> loaded_weight``.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]


def _resolve_active_method(config: Any) -> "QuantizationMethod | None":
    """Resolve the registered quantization method for a checkpoint config.

    Walks ``config`` and a few well-known nested sub-configs (composite models
    keep quantization metadata on the top-level config, while individual stages
    are built from sub-configs that may not carry it) looking for a
    ``quantization_config`` block, then asks the registry to resolve it.
    """
    from sglang_omni.quantization.registry import QuantizationRegistry

    if config is None:
        return None

    seen: set[int] = set()
    candidates = [config]
    for attr in ("thinker_config", "talker_config", "text_config"):
        nested = getattr(config, attr, None)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        quant_config = getattr(candidate, "quantization_config", None)
        if quant_config is None and isinstance(candidate, dict):
            quant_config = candidate.get("quantization_config")
        if quant_config is None:
            continue

        if isinstance(quant_config, dict):
            quant_dict = quant_config
        elif hasattr(quant_config, "to_dict"):
            quant_dict = quant_config.to_dict()
        elif hasattr(quant_config, "__dict__"):
            quant_dict = vars(quant_config)
        else:
            continue

        method = QuantizationRegistry.detect({"quantization_config": quant_dict})
        if method is not None:
            return method

    return None


def resolve_weight_preprocessor(config: Any = None) -> WeightPreprocessor:
    """Return the active quantization method's ``preprocess_weights`` callable."""
    method = _resolve_active_method(config)
    if method is None:
        return lambda name, weight: weight
    return method.preprocess_weights
