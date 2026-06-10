# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoRound quantization method."""

from __future__ import annotations

import torch

from sglang_omni.quantization.methods.autoround import AutoRoundQuantization


class TestAutoRoundQuantization:
    """Tests for AutoRoundQuantization."""

    def test_detect_auto_round(self) -> None:
        """Test detecting 'auto-round' quantization."""
        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
            }
        }

        assert AutoRoundQuantization.detect(config) is True

    def test_detect_autoround(self) -> None:
        """Test detecting 'autoround' quantization."""
        config = {
            "quantization_config": {
                "quant_method": "autoround",
                "bits": 4,
            }
        }

        assert AutoRoundQuantization.detect(config) is True

    def test_detect_auto_round_with_underscore(self) -> None:
        """Test detecting 'auto_round' quantization."""
        config = {
            "quantization_config": {
                "quant_method": "auto_round",
                "bits": 4,
            }
        }

        assert AutoRoundQuantization.detect(config) is True

    def test_detect_inc(self) -> None:
        """Test detecting 'inc' (Intel Neural Compressor) quantization."""
        config = {
            "quantization_config": {
                "quant_method": "inc",
                "bits": 4,
            }
        }

        assert AutoRoundQuantization.detect(config) is True

    def test_detect_fp8(self) -> None:
        """Test that FP8 is not detected as AutoRound."""
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
            }
        }

        assert AutoRoundQuantization.detect(config) is False

    def test_detect_awq(self) -> None:
        """Test that AWQ is not detected as AutoRound."""
        config = {
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
            }
        }

        assert AutoRoundQuantization.detect(config) is False

    def test_detect_no_quantization(self) -> None:
        """Test that missing quantization config returns False."""
        config = {"model_type": "qwen3"}

        assert AutoRoundQuantization.detect(config) is False

    def test_remap_block_names_qwen3_omni(self) -> None:
        """Test block name remapping for Qwen3-Omni checkpoints."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.1.mlp.gate_up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
        config = {
            "block_name_to_quantize": "model.layers",
        }

        mapping = method.remap_block_names(checkpoint_names, config)

        assert len(mapping) == 5
        # The checkpoint names should remain unchanged since they already match runtime names
        assert (
            mapping["model.layers.0.self_attn.q_proj.weight"]
            == "model.layers.0.self_attn.q_proj.weight"
        )

    def test_remap_block_names_transformer_blocks(self) -> None:
        """Test block name remapping for transformer_blocks pattern."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "transformer_blocks.0.attn.qkv_proj.weight",
            "transformer_blocks.1.attn.qkv_proj.weight",
        ]
        config = {
            "block_name_to_quantize": "transformer_blocks",
        }

        mapping = method.remap_block_names(checkpoint_names, config)

        assert len(mapping) == 2
        # Should map transformer_blocks -> model.layers
        assert (
            mapping["transformer_blocks.0.attn.qkv_proj.weight"]
            == "model.layers.0.attn.qkv_proj.weight"
        )
        assert (
            mapping["transformer_blocks.1.attn.qkv_proj.weight"]
            == "model.layers.1.attn.qkv_proj.weight"
        )

    def test_remap_block_names_no_pattern_match(self) -> None:
        """Test remapping when no patterns match."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "embed_tokens.weight",
            "lm_head.weight",
        ]
        config = {
            "block_name_to_quantize": "model.layers",
        }

        mapping = method.remap_block_names(checkpoint_names, config)

        # No mappings since embed_tokens and lm_head don't match model.layers
        assert len(mapping) == 0

    def test_remap_block_names_empty_config(self) -> None:
        """Test remapping with empty block_name_to_quantize."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "model.layers.0.self_attn.q_proj.weight",
        ]
        config = {}

        mapping = method.remap_block_names(checkpoint_names, config)

        assert len(mapping) == 0

    def test_remap_block_names_comma_separated(self) -> None:
        """Test remapping with comma-separated block names."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "transformer_blocks.0.attn.qkv_proj.weight",
            "single_transformer_blocks.0.mlp.weight",
        ]
        config = {
            "block_name_to_quantize": "transformer_blocks,single_transformer_blocks",
        }

        mapping = method.remap_block_names(checkpoint_names, config)

        assert len(mapping) == 2
        assert (
            mapping["transformer_blocks.0.attn.qkv_proj.weight"]
            == "model.layers.0.attn.qkv_proj.weight"
        )
        assert (
            mapping["single_transformer_blocks.0.mlp.weight"]
            == "model.layers.0.mlp.weight"
        )

    def test_extract_checkpoint_block_mapping(self) -> None:
        """Test extracting block mapping from config."""
        method = AutoRoundQuantization()
        config = {
            "block_name_to_quantize": "transformer_blocks,single_transformer_blocks",
        }

        mapping = method.extract_checkpoint_block_mapping(config)

        # transformer_blocks is in DEFAULT_CHECKPOINT_TO_RUNTIME_MAP
        assert mapping["transformer_blocks"] == "model.layers"
        # single_transformer_blocks is not in the map, so it maps to itself
        assert mapping["single_transformer_blocks"] == "single_transformer_blocks"

    def test_get_quantized_param_names(self) -> None:
        """Test getting quantized parameter suffixes."""
        method = AutoRoundQuantization()

        suffixes = method.get_quantized_param_names()

        assert ".qweight" in suffixes
        assert ".g_idx" in suffixes
        assert ".scales" in suffixes

    def test_preprocess_weights(self) -> None:
        """Test that preprocess_weights returns input unchanged."""
        method = AutoRoundQuantization()
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = method.preprocess_weights("model.layers.0.weight", weight)

        assert torch.equal(result, weight)

    def test_weight_loader(self) -> None:
        """Test weight loader copies data correctly."""
        method = AutoRoundQuantization()
        param = torch.zeros(2, 2)
        loaded_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        method.weight_loader(param, loaded_weight)

        assert torch.equal(param, loaded_weight)


class TestAutoRoundQuantizationNameRemapping:
    """Tests for AutoRound name remapping edge cases."""

    def test_remap_h_pattern_gpt_models(self) -> None:
        """Test remapping 'h' pattern used in GPT-style models."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "h.0.attn.qkv_proj.weight",
            "h.1.mlp.weight",
        ]
        config = {
            "block_name_to_quantize": "h",
        }

        mapping = method.remap_block_names(checkpoint_names, config)

        assert len(mapping) == 2
        assert mapping["h.0.attn.qkv_proj.weight"] == "blocks.0.attn.qkv_proj.weight"
        assert mapping["h.1.mlp.weight"] == "blocks.1.mlp.weight"

    def test_remap_decoder_layers_pattern(self) -> None:
        """Test remapping decoder.layers pattern."""
        method = AutoRoundQuantization()
        checkpoint_names = [
            "decoder.layers.0.attention.q_proj.weight",
            "decoder.layers.1.mlp.weight",
        ]
        config = {
            "block_name_to_quantize": "decoder.layers",
        }

        mapping = method.remap_block_names(checkpoint_names, config)

        assert len(mapping) == 2
        assert (
            mapping["decoder.layers.0.attention.q_proj.weight"]
            == "decoder.layers.0.attention.q_proj.weight"
        )
