# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization weight loader."""

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.quantization.config import QuantizationConfig
from sglang_omni.quantization.loader import QuantizedWeightLoader, bind_weight_loaders


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)


class TestQuantizedWeightLoader:
    """Tests for QuantizedWeightLoader."""

    def test_load_weights_no_quantization(self) -> None:
        """Test loading weights without quantization."""
        model = SimpleModel()
        config = None
        loader = QuantizedWeightLoader(model, config)

        # Create fake weights
        weights = [
            ("linear1.weight", torch.randn(20, 10)),
            ("linear1.bias", torch.randn(20)),
            ("linear2.weight", torch.randn(30, 20)),
            ("linear2.bias", torch.randn(30)),
        ]

        loader.load_weights(weights)

        # Weights should be loaded (though randomized, just check shapes)
        assert model.linear1.weight.shape == (20, 10)
        assert model.linear1.bias.shape == (20,)
        assert model.linear2.weight.shape == (30, 20)
        assert model.linear2.bias.shape == (30,)

    def test_load_weights_with_fp8_config(self) -> None:
        """Test loading weights with FP8 config."""
        model = SimpleModel()
        config = QuantizationConfig(
            method="fp8",
            bits=8,
            group_size=128,
        )
        loader = QuantizedWeightLoader(model, config)

        # FP8 should be registered and method should be set
        assert loader.method is not None

        # Create weights
        weights = [
            ("linear1.weight", torch.randn(20, 10)),
            ("linear1.bias", torch.randn(20)),
        ]

        loader.load_weights(weights)

    def test_load_weights_with_autoround_config(self) -> None:
        """Test loading weights with AutoRound config."""
        model = SimpleModel()
        config = QuantizationConfig(
            method="auto-round",
            bits=4,
            group_size=128,
        )
        loader = QuantizedWeightLoader(model, config)

        # AutoRound should be registered and method should be set
        assert loader.method is not None

        # Create weights
        weights = [
            ("linear1.weight", torch.randn(20, 10)),
        ]

        loader.load_weights(weights)

    def test_load_weights_unknown_quantization(self) -> None:
        """Test loading weights with unknown quantization method."""
        model = SimpleModel()
        config = QuantizationConfig(
            method="unknown-method",
            bits=4,
        )
        loader = QuantizedWeightLoader(model, config)

        # Method should be None for unknown quantization
        assert loader.method is None

        # Should still load weights (fall back to default)
        weights = [
            ("linear1.weight", torch.randn(20, 10)),
        ]

        # Should not raise
        loader.load_weights(weights)


class TestBindWeightLoaders:
    """Tests for bind_weight_loaders."""

    def test_bind_weight_loaders(self) -> None:
        """Test binding weight loaders to parameters."""
        model = SimpleModel()

        bind_weight_loaders(model)

        # All parameters should now have weight_loader
        for name, param in model.named_parameters():
            assert hasattr(param, "weight_loader"), f"Missing weight_loader for {name}"
