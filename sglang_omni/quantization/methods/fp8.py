# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization method implementation."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.quantization.base import QuantizationMethod
from sglang_omni.quantization.registry import QuantizationRegistry

logger = logging.getLogger(__name__)


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
        """Configure SGLang for FP8 quantization."""
        # Check if model has MoE
        has_moe = self._model_has_moe(model_config)
        has_native_fp8 = self._model_has_native_fp8_block_quant(model_config)

        # Determine MoE backend
        moe_runner_backend = getattr(server_args, "moe_runner_backend", "auto")
        if moe_runner_backend == "auto":
            if has_moe and has_native_fp8 and self._is_cutlass_supported():
                server_args.moe_runner_backend = "cutlass"

        logger.info(
            f"FP8 quantization configured: moe_backend={server_args.moe_runner_backend}, "
            f"gemm_backend={server_args.fp8_gemm_runner_backend}"
        )

    def _model_has_moe(self, model_config: Any) -> bool:
        """Check if model has MoE architecture."""
        config_to_check = getattr(model_config, "hf_text_config", None)
        if config_to_check is None:
            hf_config = getattr(model_config, "hf_config", None)
            config_to_check = getattr(hf_config, "text_config", hf_config)
        return hasattr(config_to_check, "num_experts_per_tok")

    def _model_has_native_fp8_block_quant(self, model_config: Any) -> bool:
        """Check if model has native FP8 block quantization."""
        hf_config = getattr(model_config, "hf_config", None)
        quant_config = getattr(hf_config, "quantization_config", None)
        if quant_config is None:
            return False
        quant_method = getattr(quant_config, "quant_method", None)
        weight_block_size = getattr(quant_config, "weight_block_size", None)
        return (
            quant_method is not None
            and str(quant_method).lower() == "fp8"
            and weight_block_size is not None
        )

    def _is_cutlass_supported(self) -> bool:
        try:
            from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
            from sglang.srt.utils import is_sm90_supported, is_sm100_supported

            return bool(
                cutlass_fp8_supported()
                and (is_sm90_supported() or is_sm100_supported())
            )
        except ImportError:
            return False

    def preprocess_weights(
        self,
        target_name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        return convert_fp8_weight_scale_inv(target_name, loaded_weight)
