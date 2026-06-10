# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for quantization methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn


@dataclass
class QuantizedLinearSpec:
    """Specification for a quantized linear layer."""

    in_features: int
    out_features: int
    bias: bool = False


class QuantizationMethod(ABC):
    """Base class for all quantization methods.

    Each quantization method (FP8, AutoRound, AWQ, etc.) must implement:

    1. name - unique identifier
    2. detect() - detect if checkpoint uses this quantization
    3. configure() - apply quantization config to runtime
    4. create_linear() - create quantized linear layer
    5. weight_loader() - custom weight loading logic (optional)
    """

    name: str = ""

    @classmethod
    @abstractmethod
    def detect(cls, config: dict[str, Any]) -> bool:
        """Detect if checkpoint uses this quantization method.

        Args:
            config: The model's config.json as a dict

        Returns:
            True if this quantization method is used
        """

    @abstractmethod
    def configure(self, server_args: Any, model_config: Any) -> None:
        """Configure server/runtime for this quantization.

        Args:
            server_args: Server arguments
            model_config: Model configuration
        """

    @abstractmethod
    def create_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a quantized linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias
            **kwargs: Additional quantization-specific arguments

        Returns:
            Quantized linear layer
        """

    def preprocess_weights(
        self,
        target_name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess weights during loading.

        Override this for quantization methods that need weight transformation.

        Args:
            target_name: Parameter name in checkpoint
            loaded_weight: Raw weight tensor from checkpoint

        Returns:
            Transformed weight tensor
        """
        return loaded_weight

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Custom weight loader for this quantization.

        Override for quantization methods with non-standard weight formats.

        Args:
            param: Model parameter to load into
            loaded_weight: Weight from checkpoint
            **kwargs: Additional loader arguments
        """
        param.data.copy_(loaded_weight)

    @classmethod
    def get_weight_block_size(cls) -> tuple[int, int] | None:
        """Return the weight block size for block quantization.

        Override for methods that use block quantization (e.g., FP8).

        Returns:
            (block_m, block_n) or None if not block quantization
        """
        return None

    def remap_block_names(
        self,
        checkpoint_names: list[str],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Remap checkpoint block names to runtime module names.

        Override for quantization methods that need name remapping
        (e.g., AutoRound).

        Args:
            checkpoint_names: List of parameter names from checkpoint
            config: Quantization config with block_name_to_quantize

        Returns:
            Mapping from checkpoint names to runtime names
        """
        del checkpoint_names, config
        return {}
