# SPDX-License-Identifier: Apache-2.0
"""Unified entry point for resolving a checkpoint's active quantization preprocessor.

This module is the single public contract between quantization metadata and
weight-loading code. Backend policy paths and stage weight loaders both go
through :func:`resolve_weight_preprocessor`, so they always agree on which
quantization applies to a given checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

    from sglang_omni.quantization.base import QuantizationMethod

# A weight preprocessor maps ``(target_name, loaded_weight) -> loaded_weight``.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]

# Attributes walked on composite-model configs when ``hf_config`` does not
# carry a top-level ``quantization_config``.
_NESTED_CONFIG_ATTRS: tuple[str, ...] = (
    "thinker_config",
    "talker_config",
    "text_config",
)


def _normalize_quant_config(quant_config: Any) -> dict[str, Any] | None:
    """Normalize a ``quantization_config`` value to a mutable dict."""
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config
    if hasattr(quant_config, "to_dict"):
        return quant_config.to_dict()
    if hasattr(quant_config, "__dict__"):
        return vars(quant_config)
    return None


def extract_quantization_config(config: Any) -> dict[str, Any] | None:
    """Extract a ``quantization_config`` dict from a root or sub-model config."""
    if config is None:
        return None

    seen: set[int] = set()
    candidates: list[Any] = [config]
    for attr in _NESTED_CONFIG_ATTRS:
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
        normalized = _normalize_quant_config(quant_config)
        if normalized is not None:
            return normalized

        # compressed-tensors stores its metadata under ``compression_config``.
        compression_config = getattr(candidate, "compression_config", None)
        if compression_config is None and isinstance(candidate, dict):
            compression_config = candidate.get("compression_config")
        normalized = _normalize_quant_config(compression_config)
        if normalized is not None:
            return normalized

    return None


def _get_method(
    *,
    config: Any = None,
    quant_dict: dict[str, Any] | None = None,
    method_name: str | None = None,
) -> "QuantizationMethod | None":
    """Resolve the registered quantization method from any supported input."""
    from sglang_omni.quantization.registry import QuantizationRegistry

    if method_name is not None:
        try:
            return QuantizationRegistry.get(method_name)()
        except KeyError:
            return None

    if quant_dict is None:
        quant_dict = extract_quantization_config(config)
    if quant_dict is None:
        return None
    return QuantizationRegistry.detect({"quantization_config": quant_dict})


def resolve_weight_preprocessor(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
    method_name: str | None = None,
) -> WeightPreprocessor:
    """Return the active quantization method's ``preprocess_weights`` callable."""
    method = _get_method(
        config=config,
        quant_dict=quant_dict,
        method_name=method_name,
    )
    if method is None:
        return lambda name, weight: weight
    return method.preprocess_weights


def detect_quantization_method(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
    method_name: str | None = None,
) -> "QuantizationMethod | None":
    """Return the active :class:`QuantizationMethod` instance, or ``None``."""
    return _get_method(
        config=config,
        quant_dict=quant_dict,
        method_name=method_name,
    )
