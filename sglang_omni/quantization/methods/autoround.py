# SPDX-License-Identifier: Apache-2.0
"""AutoRound quantization method implementation.

AutoRound is Intel's post-training quantization algorithm that uses
signed gradient descent to optimize rounding and clipping decisions.

Reference:
    - https://github.com/intel/auto-round
    - https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/quantization/autoround/

Checkpoint format:
    The checkpoint should contain a quantization_config like:
    {
        "quantization_config": {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": true,
            "packing_format": "auto_round:auto_gptq",
            "block_name_to_quantize": "transformer_blocks,single_transformer_blocks"
        }
    }
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from sglang_omni.quantization.base import QuantizationMethod
from sglang_omni.quantization.registry import QuantizationRegistry

try:
    from sglang_omni.vendor.sglang.layers import AutoRoundLinear
except ImportError:
    AutoRoundLinear = None  # Not available in this sglang version

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Mapping from checkpoint block patterns to runtime module patterns
# These are common patterns used by various models
DEFAULT_CHECKPOINT_TO_RUNTIME_MAP = {
    # Qwen3-Omni Thinker
    "model.layers": "model.layers",
    "transformer_blocks": "model.layers",
    # Qwen3-Omni Talker
    "talker.model.layers": "model.layers",
    # Generic patterns
    "blocks": "blocks",
    "h": "blocks",  # GPT-style models
    "decoder.layers": "decoder.layers",
}


@QuantizationRegistry.register
class AutoRoundQuantization(QuantizationMethod):
    """AutoRound quantization method.

    AutoRound produces pre-quantized checkpoints using INT/FP weights
    with optimized rounding. This class handles:
    1. Detection of AutoRound quantized checkpoints
    2. Block name remapping (checkpoint -> runtime)
    3. Weight preprocessing during loading
    4. Backend configuration
    """

    name = "auto-round"

    # Common quantization-aware parameter suffixes
    QUANT_SUFFIXES = (
        ".qweight",  # GPTQ-style quantized weights
        ".g_idx",  # Group indices
        ".scales",  # Quantization scales
        ".bias",  # Bias
        "_qweight",  # Alternative suffix
        "_scales",  # Alternative suffix
    )

    def __init__(self) -> None:
        super().__init__()
        self._block_mapping: dict[str, str] = {}

    @classmethod
    def detect(cls, config: dict[str, Any]) -> bool:
        """Detect AutoRound quantization from config.

        Args:
            config: Model config dict

        Returns:
            True if AutoRound quantization is detected
        """
        quant_config = config.get("quantization_config", {})
        quant_method = str(quant_config.get("quant_method", "")).lower()

        # Match various AutoRound naming conventions
        return quant_method in (
            "auto-round",
            "autoround",
            "auto_round",
            "inc",  # Intel Neural Compressor
        )

    def configure(self, server_args: Any, model_config: Any) -> None:
        """Configure SGLang for AutoRound quantized checkpoint.

        AutoRound is checkpoint-driven, so minimal runtime configuration is needed.
        The quantization is already baked into the checkpoint weights.

        Args:
            server_args: Server arguments
            model_config: Model configuration
        """
        logger.info(
            "AutoRound quantized checkpoint detected. "
            "AutoRound is checkpoint-driven and requires no additional quantization flags."
        )

        # AutoRound checkpoints for composite Omni models prefix the quantized
        # block names with the owning sub-model (e.g. ``thinker.model.layers``).
        # Each stage builds its sub-model with that prefix stripped (the thinker
        # stage exposes modules as ``model.layers.N`` rather than
        # ``thinker.model.layers.N``).
        self._normalize_block_names_for_stage(model_config)

    # Map a sub-model architecture to the checkpoint weight-name prefix that the
    # corresponding stage strips when loading weights.
    _ARCH_CHECKPOINT_PREFIX = {
        "Qwen3OmniThinkerForCausalLM": "thinker.",
        "Qwen3ASRForConditionalGeneration": "thinker.",
        "Qwen3OmniTalker": "talker.",
    }

    def _normalize_block_names_for_stage(self, model_config: Any) -> None:
        """Strip the active stage's checkpoint prefix from the quant config.

        Mutates ``model_config.hf_config.quantization_config`` in place so SGLang's
        AutoRoundConfig targets the runtime module names of the current stage.
        Entries belonging to other sub-models are left untouched so they keep
        failing to match (those stages stay unquantized, as the checkpoint intends).
        """
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return
        quant_config = getattr(hf_config, "quantization_config", None)
        if not isinstance(quant_config, dict):
            return

        arch_list = getattr(hf_config, "architectures", None) or []
        arch = arch_list[0] if arch_list else None
        stage_prefix = self._ARCH_CHECKPOINT_PREFIX.get(arch)
        if not stage_prefix:
            return

        blocks = quant_config.get("block_name_to_quantize")
        if isinstance(blocks, str):
            block_list = [b.strip() for b in blocks.split(",") if b.strip()]
        elif isinstance(blocks, list):
            block_list = [str(b) for b in blocks]
        else:
            block_list = []

        if not block_list:
            return

        normalized_blocks = [
            entry[len(stage_prefix) :] if entry.startswith(stage_prefix) else entry
            for entry in block_list
        ]
        if normalized_blocks == block_list:
            return

        quant_config["block_name_to_quantize"] = ",".join(normalized_blocks)

        # extra_config keys are regex patterns that reference the same prefixed
        # module paths; strip the prefix in both its plain and regex-escaped forms.
        extra_config = quant_config.get("extra_config")
        if isinstance(extra_config, dict) and extra_config:
            escaped_prefix = stage_prefix.replace(".", r"\.")
            quant_config["extra_config"] = {
                key.replace(escaped_prefix, "").replace(stage_prefix, ""): value
                for key, value in extra_config.items()
            }

        logger.info(
            "Normalized AutoRound block_name_to_quantize for stage %s: %s -> %s",
            arch,
            block_list,
            normalized_blocks,
        )

    def create_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """Create AutoRound quantized linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias
            **kwargs: Additional arguments (e.g., group_size, bits)

        Returns:
            AutoRound quantized linear layer, or fallback linear
        """
        if AutoRoundLinear is not None:
            return AutoRoundLinear(
                in_features,
                out_features,
                bias=bias,
                **kwargs,
            )
        else:
            logger.warning(
                "AutoRoundLinear not available in SGLang. "
                "Please ensure AutoRound kernels are installed. "
                "Using standard linear as fallback."
            )
            return nn.Linear(in_features, out_features, bias=bias)

    def remap_block_names(
        self,
        checkpoint_names: list[str],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Remap checkpoint block names to runtime module names.

        AutoRound checkpoints may use different naming conventions than
        the runtime model. This method builds a mapping based on the
        block_name_to_quantize config.

        Args:
            checkpoint_names: List of parameter names from checkpoint
            config: Quantization config dict

        Returns:
            Mapping from checkpoint names to runtime names
        """
        self._block_mapping = {}

        block_patterns = config.get("block_name_to_quantize", "")
        if isinstance(block_patterns, str):
            block_patterns = [p.strip() for p in block_patterns.split(",") if p.strip()]
        elif not isinstance(block_patterns, list):
            block_patterns = []

        if not block_patterns:
            logger.debug("No block_name_to_quantize specified, skipping remapping")
            return {}

        for name in checkpoint_names:
            for pattern in block_patterns:
                if pattern in name:
                    # Extract the layer/block index and remaining path
                    mapped_name = self._remap_single_name(name, pattern)
                    if mapped_name:
                        self._block_mapping[name] = mapped_name
                    break

        logger.debug(
            f"AutoRound block name mapping: {len(self._block_mapping)} entries"
        )
        return self._block_mapping

    def _remap_single_name(self, name: str, pattern: str) -> str | None:
        """Remap a single parameter name based on the pattern.

        Args:
            name: Full parameter name from checkpoint
            pattern: Block pattern to match

        Returns:
            Remapped name, or None if no mapping needed
        """
        # Get the runtime prefix from our mapping
        runtime_prefix = DEFAULT_CHECKPOINT_TO_RUNTIME_MAP.get(pattern, pattern)

        # Find the position of the pattern in the name
        idx = name.find(pattern)
        if idx == -1:
            return None

        # Extract everything after the pattern (layer index, etc.)
        remaining = name[idx + len(pattern) :]

        # Extract layer index (e.g., ".0." -> "0")
        layer_match = re.match(r"\.(\d+)", remaining)
        if layer_match:
            layer_idx = layer_match.group(1)
            rest = remaining[len(layer_match.group(0)) :]
            return f"{runtime_prefix}.{layer_idx}{rest}"

        # No layer index, just append the rest
        return f"{runtime_prefix}{remaining}"

    def preprocess_weights(
        self,
        target_name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess AutoRound quantized weights during loading.

        For AutoRound, the weights are already quantized in the checkpoint.
        This method handles any necessary format conversions.

        Args:
            target_name: Parameter name in checkpoint
            loaded_weight: Raw weight tensor from checkpoint

        Returns:
            Processed weight tensor
        """
        # AutoRound weights are typically in GPTQ-compatible format
        # No preprocessing needed unless there's a specific format difference
        return loaded_weight

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Load AutoRound weights."""
        param.data.copy_(loaded_weight)

    def get_quantized_param_names(self) -> tuple[str, ...]:
        """Get the suffixes that indicate quantized parameters.

        Returns:
            Tuple of quantized parameter suffixes
        """
        return self.QUANT_SUFFIXES

    def extract_checkpoint_block_mapping(
        self, config: dict[str, Any]
    ) -> dict[str, str]:
        """Extract block mapping from quantization config.

        Args:
            config: Quantization config dict

        Returns:
            Mapping from checkpoint patterns to runtime patterns
        """
        mapping = {}

        block_patterns = config.get("block_name_to_quantize", "")
        if isinstance(block_patterns, str):
            block_patterns = [p.strip() for p in block_patterns.split(",") if p.strip()]

        for pattern in block_patterns:
            if pattern in DEFAULT_CHECKPOINT_TO_RUNTIME_MAP:
                mapping[pattern] = DEFAULT_CHECKPOINT_TO_RUNTIME_MAP[pattern]
            else:
                mapping[pattern] = pattern

        return mapping