# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization config parsing."""

from __future__ import annotations

from sglang_omni.quantization.config import QuantizationConfig


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_from_checkpoint_config_fp8(self) -> None:
        """Test parsing FP8 quantization config."""
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "group_size": 128,
            }
        }

        result = QuantizationConfig.from_checkpoint_config(config)

        assert result is not None
        assert result.method == "fp8"
        assert result.bits == 8
        assert result.group_size == 128

    def test_from_checkpoint_config_autoround(self) -> None:
        """Test parsing AutoRound quantization config."""
        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "block_name_to_quantize": "transformer_blocks,single_transformer_blocks",
            }
        }

        result = QuantizationConfig.from_checkpoint_config(config)

        assert result is not None
        assert result.method == "auto-round"
        assert result.bits == 4
        assert result.group_size == 128
        assert result.sym is True
        assert result.packing_format == "auto_round:auto_gptq"
        assert result.block_name_to_quantize == (
            "transformer_blocks",
            "single_transformer_blocks",
        )

    def test_from_checkpoint_config_no_quantization(self) -> None:
        """Test when no quantization config exists."""
        config = {"model_type": "qwen3"}

        result = QuantizationConfig.from_checkpoint_config(config)

        assert result is None

    def test_from_checkpoint_config_empty_quantization(self) -> None:
        """Test with empty quantization_config."""
        config = {"quantization_config": {}}

        result = QuantizationConfig.from_checkpoint_config(config)

        # Should return a config with empty method
        assert result is not None
        assert result.method == ""

    def test_from_checkpoint_config_block_name_as_list(self) -> None:
        """Test block_name_to_quantize as list."""
        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "block_name_to_quantize": ["blocks", "h"],
            }
        }

        result = QuantizationConfig.from_checkpoint_config(config)

        assert result is not None
        assert result.block_name_to_quantize == ("blocks", "h")

    def test_from_checkpoint_config_block_name_as_string(self) -> None:
        """Test block_name_to_quantize as comma-separated string."""
        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "block_name_to_quantize": "layer.0,layer.1,layer.2",
            }
        }

        result = QuantizationConfig.from_checkpoint_config(config)

        assert result is not None
        assert result.block_name_to_quantize == ("layer.0", "layer.1", "layer.2")
