# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni quantization layer built on top of SGLang's native quantization stack.

SGLang owns quantization end-to-end: it parses `quantization_config`,
constructs quantized layers, and executes post-load hooks. This module only
provides model-specific glue that SGLang cannot infer automatically, while
remaining model-agnostic by dispatching entirely on the resolved
`quantization_config`.

The module is driven by two registries:

* `QuantMethodSpec` / `register_quant_method` define method-specific behavior,
  including weight preprocessing and stage-local quant-config normalization.
* `CompositeModelSpec` / `register_composite_model` define metadata for
  multi-stage checkpoints, including nested config attributes and checkpoint
  weight-name prefixes.

Core APIs:

* `resolve_quant_config` discovers the active `quantization_config` (or
  `compression_config`) from the root or nested stage configs.
* `get_weight_preprocessor` returns the per-tensor preprocessing function for
  the resolved quantization method, defaulting to the identity transform.
* `normalize_quant_config` rewrites stage-local checkpoint names into runtime
  module names for quantization methods that require normalization.

Adding support for a new quantization method or composite model only requires
registering a new spec; the core dispatch logic remains unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# A weight preprocessor maps `(target_name, loaded_weight) -> loaded_weight`.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]


@dataclass(frozen=True)
class PreprocessorContext:
    """Context passed to `build_preprocessor`."""

    fp8_scale_inverted: bool = False


# Quant-method registry: Adding a new method is a single `register_quant_method` call
@dataclass(frozen=True)
class QuantMethodSpec:
    """Extra, method-specific handling this module layers on top of SGLang."""

    name: str
    needs_stage_normalization: bool = False
    build_preprocessor: (
        Callable[[dict[str, Any], PreprocessorContext], WeightPreprocessor] | None
    ) = None


_QUANT_METHOD_REGISTRY: dict[str, QuantMethodSpec] = {}


def register_quant_method(spec: QuantMethodSpec) -> None:
    """Register (or replace) the handling for a quantization method."""
    existing = _QUANT_METHOD_REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        logger.warning(
            "Overriding existing quant method spec for %r: %r -> %r",
            spec.name,
            existing,
            spec,
        )
    _QUANT_METHOD_REGISTRY[spec.name] = spec


# Composite-model registry: per-architecture metadata for multi-stage checkpoints.
@dataclass(frozen=True)
class CompositeModelSpec:
    """Per-architecture metadata for a multi-stage checkpoint."""

    arch: str
    checkpoint_prefix: str | None = None
    nested_config_attr: str | None = None


_COMPOSITE_MODEL_REGISTRY: dict[str, CompositeModelSpec] = {}

# Sub-config attributes walked to find a nested `quantization_config` when
# the root config does not carry one. Composite models with other
# stage-config attribute names register them via `register_composite_model`.
_NESTED_QUANT_CONFIG_ATTRS: list[str] = ["text_config"]


def register_composite_model(
    arch: str,
    *,
    checkpoint_prefix: str | None = None,
    nested_config_attr: str | None = None,
) -> None:
    """Register (or replace) a multi-stage model architecture's quant metadata."""
    existing = _COMPOSITE_MODEL_REGISTRY.get(arch)
    new_spec = CompositeModelSpec(
        arch=arch,
        checkpoint_prefix=checkpoint_prefix,
        nested_config_attr=nested_config_attr,
    )
    if existing is not None and existing != new_spec:
        logger.warning(
            "Overriding existing composite model spec for %r: %r -> %r",
            arch,
            existing,
            new_spec,
        )
    _COMPOSITE_MODEL_REGISTRY[arch] = new_spec

    if nested_config_attr and nested_config_attr not in _NESTED_QUANT_CONFIG_ATTRS:
        _NESTED_QUANT_CONFIG_ATTRS.append(nested_config_attr)


__all__ = [
    "QuantMethodSpec",
    "register_quant_method",
    "PreprocessorContext",
    "CompositeModelSpec",
    "register_composite_model",
    "resolve_quant_config",
    "quant_method_name",
    "is_fp8_block_quant",
    "convert_fp8_weight_scale_inv",
    "get_weight_preprocessor",
    "needs_quant_config_normalization",
    "normalize_quant_config",
]


def _to_mutable_dict(quant_config: Any) -> dict[str, Any] | None:
    """Normalize a `quantization_config` value to a mutable dict."""
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config
    if hasattr(quant_config, "to_dict"):
        return quant_config.to_dict()
    if hasattr(quant_config, "__dict__"):
        return vars(quant_config)
    return None


def _read_metadata(node: Any, key: str) -> Any:
    """Read `key` off an object- or dict-shaped config node, or `None`."""
    if isinstance(node, dict):
        return node.get(key)
    return getattr(node, key, None)


# Ordered by priority.
_QUANT_METADATA_KEYS: tuple[str, ...] = ("quantization_config", "compression_config")


def resolve_quant_config(config: Any) -> dict[str, Any] | None:
    """Extract a `quantization_config` dict from a root or sub-model config."""
    visited: set[int] = set()

    def _search(node: Any) -> dict[str, Any] | None:
        if node is None or id(node) in visited:
            return None
        visited.add(id(node))

        for key in _QUANT_METADATA_KEYS:
            found = _to_mutable_dict(_read_metadata(node, key))
            if found is not None:
                return found

        for attr in _NESTED_QUANT_CONFIG_ATTRS:
            found = _search(_read_metadata(node, attr))
            if found is not None:
                return found
        return None

    return _search(config)


def quant_method_name(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
) -> str | None:
    """Return the checkpoint's normalized quantization method name, or `None`."""
    if quant_dict is None:
        quant_dict = resolve_quant_config(config)
    if not quant_dict:
        return None
    method = quant_dict.get("quant_method")
    if method is None:
        return None
    return str(method).lower().replace("_", "-")


def is_fp8_block_quant(quant_dict: dict[str, Any] | None) -> bool:
    """True when the checkpoint is native block-FP8."""
    if not quant_dict:
        return False
    if quant_method_name(quant_dict=quant_dict) != "fp8":
        return False
    return quant_dict.get("weight_block_size") is not None


def convert_fp8_weight_scale_inv(
    target_name: str,
    loaded_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Reciprocate a `weight_scale_inv` tensor into the SGLang runtime scale."""
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


def _build_fp8_preprocessor(
    quant_dict: dict[str, Any], context: PreprocessorContext
) -> WeightPreprocessor:
    """Return the FP8 weight preprocessor."""
    if not context.fp8_scale_inverted or not is_fp8_block_quant(quant_dict):
        return _identity_preprocessor
    return convert_fp8_weight_scale_inv


def get_weight_preprocessor(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
    fp8_scale_inverted: bool = False,
) -> WeightPreprocessor:
    """Return the per-tensor weight transform for a checkpoint's quantization."""
    if quant_dict is None:
        quant_dict = resolve_quant_config(config)

    method = quant_method_name(quant_dict=quant_dict)
    spec = _QUANT_METHOD_REGISTRY.get(method) if method else None
    if spec is None or spec.build_preprocessor is None:
        return _identity_preprocessor

    context = PreprocessorContext(fp8_scale_inverted=fp8_scale_inverted)
    return spec.build_preprocessor(quant_dict or {}, context)


def needs_quant_config_normalization(
    config: Any = None,
    *,
    quant_dict: dict[str, Any] | None = None,
) -> bool:
    """True when the checkpoint's method uses stage-local per-block quant names."""
    method = quant_method_name(config, quant_dict=quant_dict)
    spec = _QUANT_METHOD_REGISTRY.get(method) if method else None
    return bool(spec and spec.needs_stage_normalization)


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
    """Strip `stage_prefix` from the leading edge of every regex key."""
    extra_config = quant_config.get("extra_config")
    if not (isinstance(extra_config, dict) and extra_config):
        return False

    escaped_prefix = stage_prefix.replace(".", r"\.")
    normalized_extra: dict[str, Any] = {}
    changed = False
    for key, value in extra_config.items():
        normalized_key = _strip_stage_prefix(key, stage_prefix, escaped_prefix)
        changed = changed or normalized_key != key
        normalized_extra[normalized_key] = value

    if not changed:
        return False

    quant_config["extra_config"] = normalized_extra
    return True


def _normalize_block_name_to_quantize(
    quant_config: dict[str, Any], stage_prefix: str
) -> bool:
    """Strip `stage_prefix` from every entry of `block_name_to_quantize`."""
    blocks = quant_config.get("block_name_to_quantize")
    if isinstance(blocks, str):
        block_list = [b.strip() for b in blocks.split(",") if b.strip()]
        was_list = False
    elif isinstance(blocks, list):
        block_list = [str(b) for b in blocks]
        was_list = True
    else:
        return False
    if not block_list:
        return False

    normalized_blocks = [
        entry[len(stage_prefix) :] if entry.startswith(stage_prefix) else entry
        for entry in block_list
    ]
    if normalized_blocks == block_list:
        return False

    quant_config["block_name_to_quantize"] = (
        normalized_blocks if was_list else ",".join(normalized_blocks)
    )
    return True


def _load_writable_quant_config(
    hf_config: Any,
) -> tuple[str, dict[str, Any], bool] | None:
    """Return `(metadata_key, quant_config, needs_writeback)` for `hf_config`'s own
    quant metadata, or `None` if it doesn't carry one."""
    for metadata_key in _QUANT_METADATA_KEYS:
        quant_config_raw = getattr(hf_config, metadata_key, None)
        if quant_config_raw is None:
            continue

        quant_config = _to_mutable_dict(quant_config_raw)
        if quant_config is None:
            raise TypeError(
                f"Stage-local normalization was requested but {metadata_key} "
                f"has an unsupported type {type(quant_config_raw).__name__!r}. "
                f"Expected dict or object with to_dict()/__dict__."
            )
        # If we created a new dict from a non-dict object, we must write it
        # back after mutation so downstream consumers see the normalized names.
        needs_writeback = quant_config is not quant_config_raw
        return metadata_key, quant_config, needs_writeback

    return None


def _get_architecture(hf_config: Any) -> str | None:
    """Return the first HF `architectures` entry, or `None`."""
    architectures: Sequence[str] = getattr(hf_config, "architectures", None) or []
    return architectures[0] if architectures else None


def _resolve_stage_prefix(hf_config: Any) -> tuple[str | None, str | None]:
    """Return `(arch, checkpoint_prefix)` for the active stage architecture."""
    arch = _get_architecture(hf_config)
    spec = _COMPOSITE_MODEL_REGISTRY.get(arch) if arch else None
    return arch, (spec.checkpoint_prefix if spec else None)


def normalize_quant_config(model_config: Any) -> None:
    """Strip the active stage's checkpoint prefix from the quant config."""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return

    loaded = _load_writable_quant_config(hf_config)
    if loaded is None:
        return
    metadata_key, quant_config, needs_writeback = loaded

    arch, stage_prefix = _resolve_stage_prefix(hf_config)
    if not stage_prefix:
        return

    blocks_changed = _normalize_block_name_to_quantize(quant_config, stage_prefix)
    extra_changed = _normalize_extra_config_keys(quant_config, stage_prefix)
    if not (blocks_changed or extra_changed):
        return

    if needs_writeback:
        setattr(hf_config, metadata_key, quant_config)


# Built-in quant-method specs. A new quantization method (e.g. AWQ, GPTQ) is a single `register_quant_method` call.
register_quant_method(
    QuantMethodSpec(
        name="fp8",
        needs_stage_normalization=False,
        build_preprocessor=_build_fp8_preprocessor,
    )
)
register_quant_method(
    QuantMethodSpec(
        name="auto-round",
        needs_stage_normalization=True,
    )
)


# Built-in registrations for composite models already integrated in this
# repo. New composite models call `register_composite_model` to register.
register_composite_model(
    "Qwen3OmniThinkerForCausalLM",
    checkpoint_prefix="thinker.",
    nested_config_attr="thinker_config",
)
register_composite_model(
    "Qwen3ASRForConditionalGeneration",
    checkpoint_prefix="thinker.",
    nested_config_attr="thinker_config",
)
register_composite_model(
    "Qwen3OmniTalker",
    checkpoint_prefix="talker.",
    nested_config_attr="talker_config",
)
