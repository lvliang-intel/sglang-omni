# SPDX-License-Identifier: Apache-2.0
"""Tests for FP8 quantization method."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.quantization.methods.fp8 import FP8Quantization


class TestFP8Quantization:
    """Tests for FP8Quantization."""

    def test_detect_fp8(self) -> None:
        """Test FP8 detection from config."""
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "weight_block_size": [128, 128],
            }
        }

        assert FP8Quantization.detect(config) is True

    def test_detect_fp8_missing_block_size(self) -> None:
        """Test that FP8 without block size is not detected."""
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
            }
        }

        assert FP8Quantization.detect(config) is False

    def test_detect_not_fp8(self) -> None:
        """Test that non-FP8 methods are not detected."""
        config = {
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
            }
        }

        assert FP8Quantization.detect(config) is False

    def test_preprocess_weights_regular(self) -> None:
        """Test preprocessing regular weights (no conversion)."""
        method = FP8Quantization()
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = method.preprocess_weights("model.weight", weight)

        assert torch.equal(result, weight)

    def test_preprocess_weights_scale_inv_conversion(self) -> None:
        """Test converting weight_scale_inv to runtime scale."""
        method = FP8Quantization()
        # HF stores inverse, we need reciprocal
        weight_scale_inv = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)

        result = method.preprocess_weights(
            "model.layers.0.mlp.gate_proj.weight_scale_inv", weight_scale_inv
        )

        expected = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_preprocess_weights_scale_inv_preserves_original(self) -> None:
        """Test that original weight_scale_inv is not modified."""
        method = FP8Quantization()
        weight_scale_inv = torch.tensor([2.0, 4.0], dtype=torch.float32)

        _ = method.preprocess_weights(
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv", weight_scale_inv
        )

        assert torch.equal(weight_scale_inv, torch.tensor([2.0, 4.0]))

    def test_preprocess_weights_scale_inv_empty_raises(self) -> None:
        """Test that empty scale tensor raises ValueError."""
        method = FP8Quantization()
        weight_scale_inv = torch.tensor([], dtype=torch.float32)

        with pytest.raises(ValueError, match="Invalid empty FP8 scale tensor"):
            method.preprocess_weights(
                "model.layers.0.mlp.gate_up_proj.weight_scale_inv", weight_scale_inv
            )

    def test_preprocess_weights_scale_inv_zero_raises(self) -> None:
        """Test that zero scale tensor raises ValueError."""
        method = FP8Quantization()
        weight_scale_inv = torch.tensor([2.0, 0.0], dtype=torch.float32)

        with pytest.raises(ValueError, match="Invalid zero FP8 scale tensor"):
            method.preprocess_weights(
                "model.layers.0.mlp.down_proj.weight_scale_inv", weight_scale_inv
            )

    def test_preprocess_weights_scale_inv_inf_raises(self) -> None:
        """Test that infinite scale tensor raises ValueError."""
        method = FP8Quantization()
        weight_scale_inv = torch.tensor([2.0, float("inf")], dtype=torch.float32)

        with pytest.raises(ValueError, match="Invalid non-finite FP8 scale tensor"):
            method.preprocess_weights(
                "model.layers.0.self_attn.o_proj.weight_scale_inv", weight_scale_inv
            )

    def test_preprocess_weights_scale_inv_nan_raises(self) -> None:
        """Test that NaN scale tensor raises ValueError."""
        method = FP8Quantization()
        weight_scale_inv = torch.tensor([2.0, float("nan")], dtype=torch.float32)

        with pytest.raises(ValueError, match="Invalid non-finite FP8 scale tensor"):
            method.preprocess_weights(
                "model.layers.0.self_attn.o_proj.weight_scale_inv", weight_scale_inv
            )

    def test_preprocess_weights_scale_inv_non_float_raises(self) -> None:
        """Test that non-float scale tensor raises TypeError."""
        method = FP8Quantization()
        weight_scale_inv = torch.tensor([1, 2, 3], dtype=torch.int32)

        with pytest.raises(TypeError, match="FP8 scale tensor.*must be floating point"):
            method.preprocess_weights(
                "model.layers.0.mlp.gate_up_proj.weight_scale_inv", weight_scale_inv
            )
