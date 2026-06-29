# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoRound quantization method."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

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
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers,model.experts"

    def test_normalizes_block_name_list_input(self) -> None:
        """A list-valued ``block_name_to_quantize`` is normalized and serialized."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": ["thinker.model.layers", "model.shared"],
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

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
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_extra_config_normalized_when_block_names_already_stripped(
        self,
    ) -> None:
        """``extra_config`` is rewritten even when block names are already stripped."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        # block_name_to_quantize must remain unchanged (idempotent)
        assert quant_config["block_name_to_quantize"] == "model.layers"
        # but extra_config must still be normalized
        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_extra_config_prefix_anchored_at_pattern_start(self) -> None:
        """Only a leading stage prefix is stripped from ``extra_config`` keys."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                # leading prefix → should be stripped
                r"thinker\.model\.layers\.0": {"bits": 8},
                # prefix appears inside an alternation → must be preserved
                r"(?:thinker|decoder)\.model\.layers\.1": {"bits": 4},
                # prefix appears as a substring of another name → must be preserved
                r"thinker_audio\.model\.layers\.2": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            r"(?:thinker|decoder)\.model\.layers\.1": {"bits": 4},
            r"thinker_audio\.model\.layers\.2": {"bits": 4},
        }

    def test_object_config_extra_config_normalized_when_blocks_already_stripped(
        self,
    ) -> None:
        """Object-shaped configs: extra_config is normalized even when blocks are
        already stripped, and the converted dict is written back to
        ``hf_config.quantization_config``.
        """
        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="model.layers",
            extra_config={r"thinker\.model\.layers\.0": {"bits": 8}},
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert (
            model_config.hf_config.quantization_config["block_name_to_quantize"]
            == "model.layers"
        )
        assert model_config.hf_config.quantization_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8}
        }

    def test_no_change_when_block_names_lack_prefix(self) -> None:
        """Already prefix-less block names are left untouched (idempotent)."""
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

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

    def test_non_dict_quantization_config_raises(self) -> None:
        """A non-dict ``quantization_config`` raises TypeError.

        AutoRound's ``block_name_to_quantize`` normalization is required for
        correctness, so we must fail loudly when the config shape is unsupported.
        """
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", "not-a-dict")

        with pytest.raises(TypeError, match="unsupported type"):
            AutoRoundQuantization().configure(
                server_args=None, model_config=model_config
            )

    def test_missing_block_name_to_quantize_is_noop(self) -> None:
        """No ``block_name_to_quantize`` key leaves the config unchanged."""
        quant_config = {"quant_method": "auto-round"}
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config == {"quant_method": "auto-round"}


class TestAutoRoundObjectShapedConfig:
    """Tests for AutoRound handling object-shaped quantization configs."""

    def test_object_quant_config_converted_and_written_back(self) -> None:
        """Object-shaped config (__dict__) is converted and written back."""
        from types import SimpleNamespace

        # Create an object-shaped quantization config with __dict__
        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="thinker.model.layers",
            bits=4,
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        # After writeback, hf_config.quantization_config should be a dict
        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_object_with_to_dict_converted_and_written_back(self) -> None:
        """Object with to_dict() is converted to a dict and written back."""

        class HasToDict:
            def __init__(self):
                self.quant_method = "auto-round"
                self.block_name_to_quantize = "thinker.model.layers"
                self.bits = 4

            def to_dict(self):
                return {
                    "quant_method": self.quant_method,
                    "block_name_to_quantize": self.block_name_to_quantize,
                    "bits": self.bits,
                }

        quant_config = HasToDict()
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        # to_dict() objects should be converted and the dict written back
        # to hf_config.quantization_config
        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_unsupported_quant_config_type_raises(self) -> None:
        """Unsupported quantization config type raises TypeError."""

        # A list is not a supported type
        quant_config = ["not", "a", "dict"]
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        with pytest.raises(TypeError, match="unsupported type"):
            AutoRoundQuantization().configure(
                server_args=None, model_config=model_config
            )

    def test_extra_config_with_object_quant_config(self) -> None:
        """extra_config keys are normalized even for object-shaped configs."""
        from types import SimpleNamespace

        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="thinker.model.layers",
            extra_config={
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        AutoRoundQuantization().configure(server_args=None, model_config=model_config)

        assert quant_config.extra_config == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }
