# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization method implementation."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.quantization.base import QuantizationMethod
from sglang_omni.quantization.registry import QuantizationRegistry


def convert_fp8_weight_scale_inv(
    target_name: str,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    """Convert an FP8 ``weight_scale_inv`` tensor to the SGLang runtime scale."""
    if not target_name.endswith("weight_scale_inv"):
        return loaded_weight

    if not torch.is_floating_point(loaded_weight):
        raise TypeError(f"FP8 scale tensor for {target_name} must be floating point")
    if loaded_weight.numel() == 0:
        raise ValueError(f"Invalid empty FP8 scale tensor for {target_name}")
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise ValueError(f"Invalid non-finite FP8 scale tensor for {target_name}")
    if bool(torch.any(loaded_weight == 0).item()):
        raise ValueError(f"Invalid zero FP8 scale tensor for {target_name}")

    return torch.reciprocal(loaded_weight)


@QuantizationRegistry.register
class FP8Quantization(QuantizationMethod):
    """FP8 block quantization method.

    This handles the weight_scale_inv conversion needed for SGLang runtime.
    FP8 quantized checkpoints store scales as weight_scale_inv,
    so we need to convert them during loading.
    """

    name = "fp8"

    @classmethod
    def detect(cls, config: dict[str, Any]) -> bool:
        """Detect FP8 quantization from config."""
        quant_config = config.get("quantization_config", {})
        quant_method = str(quant_config.get("quant_method", "")).lower()
        weight_block_size = quant_config.get("weight_block_size")
        return quant_method == "fp8" and weight_block_size is not None

    def configure(self, server_args: Any, model_config: Any) -> None:
        """No-op: backend policy is owned by model_worker, not by this method."""

    def preprocess_weights(
        self,
        target_name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        return convert_fp8_weight_scale_inv(target_name, loaded_weight)
