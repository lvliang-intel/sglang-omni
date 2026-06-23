# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoRound quantization method."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.quantization.methods.autoround import AutoRoundQuantization


def _make_model_config(
    architecture: str | None,
    quantization_config: object,
) -> SimpleNamespace:
    """Build a minimal ``model_config`` stub for ``configure()`` tests."""
    hf_config = SimpleNamespace(
        architectures=[architecture] if architecture is not None else [],
        quantization_config=quantization_config,
    )
    return SimpleNamespace(hf_config=hf_config)


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

    def test_preprocess_weights(self) -> None:
        """Test that preprocess_weights returns input unchanged."""
        method = AutoRoundQuantization()
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = method.preprocess_weights("model.layers.0.weight", weight)

        assert torch.equal(result, weight)


class TestAutoRoundConfigureStagePrefix:
    """Tests for ``configure()`` normalizing stage-local block names."""

    def test_strips_thinker_prefix_from_block_names(self) -> None:
        """Thinker stage strips the ``thinker.`` prefix from block names."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_talker_prefix_from_block_names(self) -> None:
        """Talker stage strips the ``talker.`` prefix from block names."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "talker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniTalker", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_asr_architecture_strips_thinker_prefix(self) -> None:
        """ASR architecture also maps to the ``thinker.`` checkpoint prefix."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config(
            "Qwen3ASRForConditionalGeneration", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_prefix_from_comma_separated_list(self) -> None:
        """Each entry of a comma-separated block list is normalized."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers,thinker.model.experts",
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert (
            quant_config["block_name_to_quantize"] == "model.layers,model.experts"
        )

    def test_normalizes_block_name_list_input(self) -> None:
        """A list-valued ``block_name_to_quantize`` is normalized and serialized."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": ["thinker.model.layers", "model.shared"],
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers,model.shared"

    def test_strips_prefix_from_extra_config_keys(self) -> None:
        """``extra_config`` regex keys have the prefix stripped (plain + escaped)."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_no_change_when_block_names_lack_prefix(self) -> None:
        """Already prefix-less block names are left untouched (idempotent)."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_unknown_architecture_leaves_block_names_unchanged(self) -> None:
        """Architectures without a known prefix mapping are not rewritten."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("SomeOtherForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "thinker.model.layers"

    def test_missing_hf_config_is_noop(self) -> None:
        """A ``model_config`` without ``hf_config`` does not raise."""
        model_config = SimpleNamespace(hf_config=None)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

    def test_non_dict_quantization_config_is_noop(self) -> None:
        """A non-dict ``quantization_config`` is ignored without raising."""
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", "not-a-dict"
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

    def test_missing_block_name_to_quantize_is_noop(self) -> None:
        """No ``block_name_to_quantize`` key leaves the config unchanged."""
        quant_config = {"quant_method": "auto-round"}
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config == {"quant_method": "auto-round"}

