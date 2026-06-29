# SPDX-License-Identifier: Apache-2.0
"""Tests for weight_preprocess module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import sglang_omni.quantization.weight_preprocess  # noqa: F401


@pytest.fixture(autouse=True)
def _reset_quantization_registry():
    """Ensure built-in quantization methods are registered before each test.

    Other tests in this directory may clear ``QuantizationRegistry._methods``,
    which would break detection for FP8 / AutoRound here. The auto-register
    path only fires ``@register`` once per process (the modules are already
    imported), so we manually re-register the built-in classes.
    """
    from sglang_omni.quantization.methods.autoround import AutoRoundQuantization
    from sglang_omni.quantization.methods.fp8 import FP8Quantization
    from sglang_omni.quantization.registry import QuantizationRegistry

    QuantizationRegistry._methods.clear()
    QuantizationRegistry._initialized = True  # Avoid re-importing modules
    QuantizationRegistry.register(FP8Quantization)
    QuantizationRegistry.register(AutoRoundQuantization)
    yield


class TestResolveWeightPreprocessor:
    """Tests for resolve_weight_preprocessor()."""

    def test_returns_identity_when_no_quantization(self) -> None:
        """No quantization config returns an identity function."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        config = SimpleNamespace(model_type="qwen3")
        preprocessor = resolve_weight_preprocessor(config)

        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = preprocessor("model.layers.0.weight", weight)

        assert torch.equal(result, weight)
        assert result is weight  # Should be the same tensor, not a copy

    def test_returns_identity_when_config_is_none(self) -> None:
        """None config returns an identity function."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        preprocessor = resolve_weight_preprocessor(None)

        weight = torch.tensor([1.0, 2.0])
        result = preprocessor("some.weight", weight)

        assert torch.equal(result, weight)

    def test_returns_identity_when_no_quantization_config(self) -> None:
        """Config without quantization_config returns identity."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        config = {"model_type": "qwen3", "hidden_size": 4096}
        preprocessor = resolve_weight_preprocessor(config)

        weight = torch.tensor([1.0])
        result = preprocessor("weight", weight)

        assert torch.equal(result, weight)

    def test_returns_fp8_preprocessor_for_fp8_checkpoint(self) -> None:
        """FP8 config returns FP8-specific preprocessor."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "weight_block_size": [128, 128],
            }
        }
        preprocessor = resolve_weight_preprocessor(config)

        # Regular weights should pass through unchanged
        weight = torch.tensor([[1.0, 2.0]])
        result = preprocessor("model.weight", weight)
        assert torch.equal(result, weight)

        # FP8 scale should be converted
        scale = torch.tensor([2.0, 4.0])
        result = preprocessor("model.layers.0.weight_scale_inv", scale)
        expected = torch.tensor([0.5, 0.25])
        assert torch.allclose(result, expected)

    def test_returns_identity_for_autoround(self) -> None:
        """AutoRound returns identity (no weight preprocessing needed)."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
            }
        }
        preprocessor = resolve_weight_preprocessor(config)

        weight = torch.tensor([[1.0, 2.0]])
        result = preprocessor("model.layers.0.weight", weight)

        # AutoRound doesn't modify weights
        assert torch.equal(result, weight)

    def test_nested_config_with_quantization_in_thinker(self) -> None:
        """Nested config with quantization in thinker_config is detected."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        thinker_quant = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "weight_block_size": [128, 128],
            }
        }
        config = SimpleNamespace(
            model_type="qwen3-omni",
            thinker_config=SimpleNamespace(**thinker_quant),
        )
        preprocessor = resolve_weight_preprocessor(config)

        # Should use FP8 preprocessor from nested config
        scale = torch.tensor([2.0])
        result = preprocessor("layers.0.weight_scale_inv", scale)
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_object_shaped_quantization_config(self) -> None:
        """Object-shaped quantization config is converted to dict."""
        from sglang_omni.quantization.weight_preprocess import (
            resolve_weight_preprocessor,
        )

        quant_config = SimpleNamespace(
            quant_method="fp8",
            bits=8,
            weight_block_size=[128, 128],
        )
        config = SimpleNamespace(quantization_config=quant_config)

        preprocessor = resolve_weight_preprocessor(config)

        scale = torch.tensor([4.0])
        result = preprocessor("layers.0.weight_scale_inv", scale)
        assert torch.allclose(result, torch.tensor([0.25]))


class TestDetectQuantizationMethod:
    """Tests for the unified ``detect_quantization_method`` entry point."""

    def test_returns_none_when_no_quantization(self) -> None:
        """Returns None when config has no quantization."""
        from sglang_omni.quantization import detect_quantization_method

        config = SimpleNamespace(model_type="qwen3")
        result = detect_quantization_method(config=config)

        assert result is None

    def test_returns_none_for_fp8_without_block_size(self) -> None:
        """FP8 without weight_block_size returns None."""
        from sglang_omni.quantization import detect_quantization_method

        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                # Missing weight_block_size
            }
        }
        result = detect_quantization_method(config=config)

        # Should not detect FP8 without required block_size
        assert result is None

    def test_returns_method_for_valid_fp8(self) -> None:
        """Valid FP8 config returns FP8Quantization."""
        from sglang_omni.quantization import detect_quantization_method
        from sglang_omni.quantization.methods.fp8 import FP8Quantization

        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "weight_block_size": [128, 128],
            }
        }
        result = detect_quantization_method(config=config)

        assert result is not None
        assert isinstance(result, FP8Quantization)
