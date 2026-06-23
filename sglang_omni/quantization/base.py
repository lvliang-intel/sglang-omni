# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for quantization methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class QuantizationMethod(ABC):
    """Base class for all quantization methods.

    Each quantization method (FP8, AutoRound, AWQ, etc.) implements:

    1. ``name`` - unique identifier used by the registry.
    2. ``detect()`` - detect whether a checkpoint uses this quantization.
    3. ``configure()`` - apply runtime/backend configuration.
    4. ``preprocess_weights()`` - optional per-tensor weight transform applied
       during loading (e.g. FP8 ``weight_scale_inv`` reciprocal).

    Layer construction is handled by SGLang's native ``quant_config`` path, and
    parameter writes by each parameter's own ``weight_loader``; methods only
    provide the pieces that are genuinely quantization-specific.
    """

    name: str = ""

    @classmethod
    @abstractmethod
    def detect(cls, config: dict[str, Any]) -> bool:
        """Detect if checkpoint uses this quantization method."""

    @abstractmethod
    def configure(self, server_args: Any, model_config: Any) -> None:
        """Configure server/runtime for this quantization."""

    def preprocess_weights(
        self,
        target_name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess a weight tensor during loading.

        Override for methods that need a per-tensor transform. The default is a
        no-op so callers can invoke it unconditionally.
        """
        return loaded_weight
