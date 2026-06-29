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
from typing import Any

from sglang_omni.quantization.base import QuantizationMethod
from sglang_omni.quantization.registry import QuantizationRegistry

logger = logging.getLogger(__name__)


@QuantizationRegistry.register
class AutoRoundQuantization(QuantizationMethod):
    """AutoRound quantization method.

    AutoRound produces pre-quantized checkpoints using INT/FP weights with
    optimized rounding. This class only needs to:
    1. Detect AutoRound quantized checkpoints.
    2. Normalize the per-stage block names in ``configure()`` so SGLang's
       native ``AutoRoundConfig`` targets the runtime module names.

    Layer construction and weight loading are handled by SGLang's native
    ``quant_config`` path and the model's own weight loaders.
    """

    name = "auto-round"

    @classmethod
    def detect(cls, config: dict[str, Any]) -> bool:
        """Detect AutoRound quantization from config."""
        quant_config = config.get("quantization_config", {})
        quant_method = str(quant_config.get("quant_method", "")).lower()
        return quant_method == "auto-round"

    def configure(self, server_args: Any, model_config: Any) -> None:
        """Configure SGLang for AutoRound quantized checkpoint."""
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

    @staticmethod
    def _to_mutable_dict(quant_config: Any) -> dict[str, Any] | None:
        """Convert a quantization_config to a mutable dict if possible."""
        if isinstance(quant_config, dict):
            return quant_config
        if hasattr(quant_config, "to_dict"):
            return quant_config.to_dict()
        if hasattr(quant_config, "__dict__"):
            return vars(quant_config)
        return None

    def _normalize_block_names_for_stage(self, model_config: Any) -> None:
        """Strip the active stage's checkpoint prefix from the quant config."""
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return
        quant_config_raw = getattr(hf_config, "quantization_config", None)
        if quant_config_raw is None:
            return

        quant_config = self._to_mutable_dict(quant_config_raw)
        if quant_config is None:
            raise TypeError(
                f"AutoRound was detected but quantization_config has an "
                f"unsupported type {type(quant_config_raw).__name__!r}. "
                f"Expected dict or object with to_dict()/__dict__."
            )

        # If we created a new dict from a non-dict object, we must write it
        # back after mutation so downstream consumers see the normalized names.
        needs_writeback = quant_config is not quant_config_raw

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

        blocks_changed = False
        normalized_blocks: list[str] = []
        if block_list:
            normalized_blocks = [
                entry[len(stage_prefix) :] if entry.startswith(stage_prefix) else entry
                for entry in block_list
            ]
            if normalized_blocks != block_list:
                quant_config["block_name_to_quantize"] = ",".join(normalized_blocks)
                blocks_changed = True

        extra_changed = self._normalize_extra_config_keys(quant_config, stage_prefix)

        if not blocks_changed and not extra_changed:
            return

        if needs_writeback:
            setattr(hf_config, "quantization_config", quant_config)

        if blocks_changed:
            logger.info(
                "Normalized AutoRound block_name_to_quantize for stage %s: %s -> %s",
                arch,
                block_list,
                normalized_blocks,
            )

    @staticmethod
    def _strip_stage_prefix(
        pattern: str, plain_prefix: str, escaped_prefix: str
    ) -> str:
        """Strip the stage prefix from the start of a regex pattern."""
        if pattern.startswith(escaped_prefix):
            return pattern[len(escaped_prefix) :]
        if pattern.startswith(plain_prefix):
            return pattern[len(plain_prefix) :]
        return pattern

    def _normalize_extra_config_keys(
        self, quant_config: dict[str, Any], stage_prefix: str
    ) -> bool:
        """Strip ``stage_prefix`` from the leading edge of every regex key."""
        extra_config = quant_config.get("extra_config")
        if not (isinstance(extra_config, dict) and extra_config):
            return False

        escaped_prefix = stage_prefix.replace(".", r"\.")
        normalized_extra = {
            self._strip_stage_prefix(key, stage_prefix, escaped_prefix): value
            for key, value in extra_config.items()
        }
        if normalized_extra == extra_config:
            return False

        quant_config["extra_config"] = normalized_extra
        return True
