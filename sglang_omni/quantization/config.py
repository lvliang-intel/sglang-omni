# SPDX-License-Identifier: Apache-2.0
"""Quantization configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class QuantizationConfig:
    """Unified quantization configuration.

    This is the model-agnostic representation of quantization settings,
    parsed from the checkpoint's quantization_config.
    """

    method: str
    bits: int = 8
    group_size: int = -1
    sym: bool = True
    packing_format: str = ""
    block_name_to_quantize: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_checkpoint_config(
        cls,
        config: dict[str, Any],
    ) -> "QuantizationConfig | None":
        """Parse quantization config from checkpoint config.json."""
        quant_config = config.get("quantization_config")
        if quant_config is None:
            return None

        quant_method = quant_config.get("quant_method", "")

        # Parse block names
        block_names = quant_config.get("block_name_to_quantize", "")
        if isinstance(block_names, str):
            block_names = tuple(b.strip() for b in block_names.split(",") if b.strip())
        elif isinstance(block_names, list):
            block_names = tuple(str(b) for b in block_names)
        else:
            block_names = ()

        return cls(
            method=quant_method,
            bits=quant_config.get("bits", 8),
            group_size=quant_config.get("group_size", -1),
            sym=quant_config.get("sym", True),
            packing_format=quant_config.get("packing_format", ""),
            block_name_to_quantize=block_names,
            extra=quant_config,
        )

    def to_backend_config(self) -> dict[str, Any]:
        """Convert to backend-specific configuration."""
        return {
            **self.extra,
            "quant_method": self.method,
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "packing_format": self.packing_format,
        }

    @property
    def is_block_quantization(self) -> bool:
        """Check if this is a block quantization method."""
        return self.group_size > 0

    @property
    def is_per_channel(self) -> bool:
        """Check if quantization is per-channel."""
        return self.group_size == -1


def detect_quantization_config(
    config: dict[str, Any],
) -> "QuantizationConfig | None":
    """Detect and parse quantization config from a model config dict."""
    return QuantizationConfig.from_checkpoint_config(config)
