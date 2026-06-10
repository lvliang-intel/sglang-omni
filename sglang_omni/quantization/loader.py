# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware weight loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch
    from torch import nn

from sglang_omni.models.weight_loader import default_weight_loader

from .config import QuantizationConfig
from .registry import QuantizationRegistry


class QuantizedWeightLoader:
    """Model-agnostic weight loader with quantization support.

    Usage:
        loader = QuantizedWeightLoader(model, config)
        loader.load_weights(weights)
    """

    def __init__(
        self,
        module: nn.Module,
        config: QuantizationConfig | None,
    ):
        """Initialize quantized weight loader.

        Args:
            module: The model module to load weights into
            config: Quantization config, or None for no quantization
        """
        self.module = module
        self.config = config
        self.method = None

        if config is not None:
            detected = QuantizationRegistry.detect(
                {"quantization_config": config.extra}
            )
            if detected is not None:
                self.method = detected

        self._params_dict: dict[str, "torch.Tensor"] = {}
        self._module_to_params(module)

    def _module_to_params(self, module: nn.Module) -> None:
        """Build parameter name to parameter mapping."""
        for name, param in module.named_parameters(remove_duplicate=False):
            self._params_dict[name] = param

    def load_weights(
        self,
        weights: Iterable[tuple[str, "torch.Tensor"]],
    ) -> None:
        """Load weights into the module.

        Args:
            weights: Iterable of (name, tensor) pairs
        """
        if self.method is None:
            self._load_weights_default(weights)
        else:
            self._load_weights_quantized(weights)

    def _load_weights_default(
        self,
        weights: Iterable[tuple[str, "torch.Tensor"]],
    ) -> None:
        """Default weight loading (no quantization)."""
        for name, loaded_weight in weights:
            if name not in self._params_dict:
                continue
            param = self._params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def _load_weights_quantized(
        self,
        weights: Iterable[tuple[str, "torch.Tensor"]],
    ) -> None:
        """Quantization-aware weight loading."""
        for target, loaded_weight in weights:
            # Apply preprocessing if needed
            if self.method is not None:
                loaded_weight = self.method.preprocess_weights(target, loaded_weight)

            if target not in self._params_dict:
                continue
            param = self._params_dict[target]

            # Use quantization-specific weight loader
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None and weight_loader != default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                self.method.weight_loader(param, loaded_weight)


def bind_weight_loaders(
    module: nn.Module,
    config: QuantizationConfig | None = None,
) -> None:
    """Bind weight loaders to all parameters.

    This allows modules to use their own weight_loader attribute
    during load_weights() calls.

    Args:
        module: The model module
        config: Quantization config (optional)
    """
    for param in module.parameters():
        if not hasattr(param, "weight_loader"):
            param.weight_loader = default_weight_loader


def detect_quantization_config(
    config: dict[str, Any],
) -> QuantizationConfig | None:
    """Detect and parse quantization config from model config.

    Args:
        config: Model config dict (from config.json or hf_config.to_dict())

    Returns:
        QuantizationConfig or None if no quantization
    """
    return QuantizationConfig.from_checkpoint_config(config)
