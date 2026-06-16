# SPDX-License-Identifier: Apache-2.0
"""Quantization module for SGLang-Omni.

This module provides a unified abstraction for quantization methods,
allowing easy addition of new quantization schemes without modifying
model-specific code.

Supported quantization methods:
    - FP8: Block FP8 quantization (e.g., for Qwen3-Omni)
    - AutoRound: Intel AutoRound post-training quantization

Usage:
    # Detect quantization from config
    from sglang_omni.quantization import detect_quantization_config

    config = {"quantization_config": {"quant_method": "auto-round", "bits": 4}}
    quant_config = detect_quantization_config(config)

    if quant_config:
        from sglang_omni.quantization import QuantizationRegistry
        method = QuantizationRegistry.get(quant_config.method)()
        method.configure(server_args, model_config)
"""

from sglang_omni.quantization.base import QuantizationMethod, QuantizedLinearSpec
from sglang_omni.quantization.config import QuantizationConfig
from sglang_omni.quantization.loader import (
    QuantizedWeightLoader,
    bind_weight_loaders,
    detect_quantization_config,
)
from sglang_omni.quantization.registry import QuantizationRegistry

# Built-in quantization methods are imported lazily on first use via
# ``QuantizationRegistry._ensure_builtins_registered()``.

__all__ = [
    "QuantizationMethod",
    "QuantizedLinearSpec",
    "QuantizationConfig",
    "QuantizedWeightLoader",
    "bind_weight_loaders",
    "detect_quantization_config",
    "QuantizationRegistry",
]
