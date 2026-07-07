# SPDX-License-Identifier: Apache-2.0
"""Omni compatibility layer between model weight loading and SGLang's native
quantization stack.

SGLang (pinned `v0.5.12.post1`) owns quantization end-to-end: it registers
`fp8` and `auto-round`, parses `quantization_config`, builds the quantized
layers, and runs the post-load hooks. This module only adds the pieces
SGLang cannot infer on its own. It is model-agnostic by default, with
composite models opting in through small data tables below.

Model-agnostic:

* `resolve_quant_config` -- discover the active `quantization_config`,
  including composite checkpoints that nest it under a per-stage sub-config.
* `get_weight_preprocessor` -- the single entry point every custom weight
  loader calls. Defaults to identity, so standard block-FP8 (dequantized
  natively by SGLang) and AutoRound both pass through untouched.

Opt-in per model (data-driven):

* `fp8_scale_inverted=True` (-> `convert_fp8_weight_scale_inv`) -- reciprocate
  a `weight_scale_inv` tensor for a checkpoint that stores it as the literal
  inverse of the runtime scale. Qwen3-Omni is the current instance.
* `normalize_quant_config` -- strip a composite checkpoint's stage prefix
  (e.g. `thinker.model.layers` -> `model.layers`) from AutoRound's
  `block_name_to_quantize` / `extra_config` keys so SGLang matches them
  against runtime module names. A new composite model opts in by adding its
  architecture -> prefix to `_ARCH_CHECKPOINT_PREFIX`, and its nested
  sub-config attribute to `_NESTED_CONFIG_ATTRS` if the metadata is nested.

Adding another quantization method (e.g. AWQ) is a single-place change here
-- a dispatch entry and, if needed, a preprocessing function -- with no
changes in model directories, which only ever call `get_weight_preprocessor`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# A weight preprocessor maps ``(target_name, loaded_weight) -> loaded_weight``.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]

# Sub-config attributes walked to find a nested ``quantization_config`` when
# the root config does not carry one. Composite models add their per-stage
# sub-config attribute names here.
_NESTED_CONFIG_ATTRS: tuple[str, ...] = (
    "thinker_config",
    "talker_config",
    "text_config",
)

# Checkpoint weight-name prefix that the corresponding stage strips when
# re-rooting its sub-model. Methods whose per-block quant names are matched
# against runtime module names (e.g. AutoRound) need these stripped before
# SGLang builds its quant config. A new composite model that re-roots stages
# adds its architecture here.
_ARCH_CHECKPOINT_PREFIX: dict[str, str] = {
    "Qwen3OmniThinkerForCausalLM": "thinker.",
    "Qwen3ASRForConditionalGeneration": "thinker.",
    "Qwen3OmniTalker": "talker.",
}

# Methods whose checkpoint stores stage-local per-block quant names that must be
# normalized before SGLang matches them against runtime module names.
_STAGE_NORMALIZED_METHODS: frozenset[str] = frozenset({"auto-round"})


__all__ = [
    "resolve_quant_config",
    "get_weight_preprocessor",
    "needs_quant_config_normalization",
    "normalize_quant_config",
]

def _to_mutable_dict(quant_config: Any) -> dict[str, Any] | None:
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


def resolve_quant_config(config: Any) -> dict[str, Any] | None:
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
        normalized = _to_mutable_dict(quant_config)
        if normalized is not None:
            return normalized

        # compressed-tensors stores its metadata under ``compression_config``.
        compression_config = getattr(candidate, "compression_config", None)
        if compression_config is None and isinstance(candidate, dict):
            compression_config = candidate.get("compression_config")
        normalized = _to_mutable_dict(compression_config)
        if normalized is not None:
            return normalized

    return None


def quant_method_name(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
) -> str | None:
    """Return the checkpoint's normalized quantization method name, or ``None``."""
    if quant_dict is None:
        quant_dict = resolve_quant_config(config)
    if not quant_dict:
        return None
    method = quant_dict.get("quant_method")
    if method is None:
        return None
    return str(method).lower().replace("_", "-")


def is_fp8_block_quant(quant_dict: dict[str, Any] | None) -> bool:
    """True when the checkpoint is native block-FP8 (``weight_block_size`` set)."""
    if not quant_dict:
        return False
    method = quant_dict.get("quant_method")
    if method is None or str(method).lower() != "fp8":
        return False
    return quant_dict.get("weight_block_size") is not None


def convert_fp8_weight_scale_inv(
    target_name: str,
    loaded_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Reciprocate a ``weight_scale_inv`` tensor into the SGLang runtime scale."""
    if not target_name.endswith("weight_scale_inv"):
        return loaded_weight

    import torch

    if not torch.is_floating_point(loaded_weight):
        raise TypeError(f"FP8 scale tensor for {target_name} must be floating point")
    if loaded_weight.numel() == 0:
        raise ValueError(f"Invalid empty FP8 scale tensor for {target_name}")
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise ValueError(f"Invalid non-finite FP8 scale tensor for {target_name}")
    if bool(torch.any(loaded_weight == 0).item()):
        raise ValueError(f"Invalid zero FP8 scale tensor for {target_name}")

    return torch.reciprocal(loaded_weight)


def _identity_preprocessor(
    target_name: str, loaded_weight: "torch.Tensor"
) -> "torch.Tensor":
    return loaded_weight


def get_weight_preprocessor(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
    fp8_scale_inverted: bool = False,
) -> WeightPreprocessor:
    """Return the per-tensor weight transform for a checkpoint's quantization."""
    if quant_dict is None:
        quant_dict = resolve_quant_config(config)
    if fp8_scale_inverted and is_fp8_block_quant(quant_dict):
        return convert_fp8_weight_scale_inv
    return _identity_preprocessor


def needs_quant_config_normalization(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
) -> bool:
    """True when the checkpoint's method uses stage-local per-block quant names."""
    return quant_method_name(config, quant_dict=quant_dict) in _STAGE_NORMALIZED_METHODS


def _strip_stage_prefix(pattern: str, plain_prefix: str, escaped_prefix: str) -> str:
    """Strip the stage prefix from the start of a regex pattern."""
    if pattern.startswith(escaped_prefix):
        return pattern[len(escaped_prefix) :]
    if pattern.startswith(plain_prefix):
        return pattern[len(plain_prefix) :]
    leading_wildcard_escaped = r".*" + escaped_prefix
    if pattern.startswith(leading_wildcard_escaped):
        # Drop only the prefix part; keep the leading ".*" wildcard so the
        # normalized regex still matches stage-local module names.
        return (
            leading_wildcard_escaped[: -len(escaped_prefix)]
            + pattern[len(leading_wildcard_escaped) :]
        )
    return pattern


def _normalize_extra_config_keys(
    quant_config: dict[str, Any], stage_prefix: str
) -> bool:
    """Strip ``stage_prefix`` from the leading edge of every regex key."""
    extra_config = quant_config.get("extra_config")
    if not (isinstance(extra_config, dict) and extra_config):
        return False

    escaped_prefix = stage_prefix.replace(".", r"\.")
    normalized_extra = {
        _strip_stage_prefix(key, stage_prefix, escaped_prefix): value
        for key, value in extra_config.items()
    }
    if normalized_extra == extra_config:
        return False

    quant_config["extra_config"] = normalized_extra
    return True


def normalize_quant_config(model_config: Any) -> None:
    """Strip the active stage's checkpoint prefix from the quant config."""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return
    quant_config_raw = getattr(hf_config, "quantization_config", None)
    if quant_config_raw is None:
        return

    quant_config = _to_mutable_dict(quant_config_raw)
    if quant_config is None:
        raise TypeError(
            f"Stage-local normalization was requested but quantization_config "
            f"has an unsupported type {type(quant_config_raw).__name__!r}. "
            f"Expected dict or object with to_dict()/__dict__."
        )

    # If we created a new dict from a non-dict object, we must write it back
    # after mutation so downstream consumers see the normalized names.
    needs_writeback = quant_config is not quant_config_raw

    arch_list = getattr(hf_config, "architectures", None) or []
    arch = arch_list[0] if arch_list else None
    stage_prefix = _ARCH_CHECKPOINT_PREFIX.get(arch)
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

    extra_changed = _normalize_extra_config_keys(quant_config, stage_prefix)

    if not blocks_changed and not extra_changed:
        return

    if needs_writeback:
        setattr(hf_config, "quantization_config", quant_config)

    if blocks_changed:
        logger.info(
            "Normalized stage-local block_name_to_quantize for stage %s: %s -> %s",
            arch,
            block_list,
            normalized_blocks,
        )
