# SPDX-License-Identifier: Apache-2.0
"""Integration tests for quantization detection in model_worker."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class FakeQuantizationConfig:
    """Fake quantization config for testing."""

    quant_method: str = "fp8"
    weight_block_size: list[int] | None = None
    bits: int = 8
    group_size: int = 128


class TestModelWorkerQuantizationDetection:
    """Integration tests for quantization detection in model_worker."""

    def test_detect_fp8_from_hf_config(self) -> None:
        """Test detecting FP8 quantization from hf_config."""
        # Import the module
        from sglang_omni.model_runner import model_worker

        # Create mock model config
        quant_config = FakeQuantizationConfig(
            quant_method="fp8",
            weight_block_size=[128, 128],
        )
        hf_config = SimpleNamespace(
            quantization_config=quant_config,
            text_config=None,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=None,
        )

        # Detect quantization
        method_name, quant_config_result = model_worker._detect_quantization_method(
            model_config
        )

        assert method_name == "fp8"
        assert quant_config_result is not None
        assert quant_config_result.method == "fp8"
        assert quant_config_result.bits == 8

    def test_detect_autoround_from_hf_config(self) -> None:
        """Test detecting AutoRound quantization from hf_config."""
        from sglang_omni.model_runner import model_worker

        quant_config = FakeQuantizationConfig(
            quant_method="auto-round",
            weight_block_size=None,
            bits=4,
        )
        hf_config = SimpleNamespace(
            quantization_config=quant_config,
            text_config=None,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=None,
        )

        method_name, quant_config_result = model_worker._detect_quantization_method(
            model_config
        )

        assert method_name == "auto-round"
        assert quant_config_result is not None
        assert quant_config_result.method == "auto-round"
        assert quant_config_result.bits == 4

    def test_detect_no_quantization(self) -> None:
        """Test when no quantization is present."""
        from sglang_omni.model_runner import model_worker

        hf_config = SimpleNamespace(
            quantization_config=None,
            text_config=None,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=None,
        )

        method_name, quant_config_result = model_worker._detect_quantization_method(
            model_config
        )

        assert method_name is None
        assert quant_config_result is None

    def test_detect_from_nested_text_config(self) -> None:
        """Test detecting quantization from nested text_config."""
        from sglang_omni.model_runner import model_worker

        # Simulate Qwen3-Omni style nested config
        quant_config = FakeQuantizationConfig(
            quant_method="fp8",
            weight_block_size=[128, 128],
        )
        text_config = SimpleNamespace(
            quantization_config=quant_config,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=4096,
            num_hidden_layers=32,
        )
        talker_config = SimpleNamespace(
            text_config=text_config,
        )
        hf_config = SimpleNamespace(
            quantization_config=None,  # Main config has no quantization
            text_config=text_config,
            talker_config=talker_config,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=text_config,
        )

        method_name, quant_config_result = model_worker._detect_quantization_method(
            model_config
        )

        assert method_name == "fp8"
        assert quant_config_result is not None

    def test_apply_quantization_method_config_fp8(self) -> None:
        """Test applying FP8 quantization configuration."""
        from sglang_omni.model_runner import model_worker

        # Create mock server args
        server_args = SimpleNamespace(
            moe_runner_backend="auto",
            fp8_gemm_runner_backend="auto",
            quantization=None,
        )

        # Create mock model config without native FP8 block quant
        # This will test the basic flow without cutlass requirements
        hf_config = SimpleNamespace(
            quantization_config=None,
            text_config=None,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=None,
        )

        # Apply FP8 configuration - this should not crash
        model_worker._apply_quantization_method_config(
            server_args,
            model_config,
            "fp8",
        )

        # Just verify it runs without error (backends may or may not change
        # depending on hardware support)
        assert hasattr(server_args, "moe_runner_backend")

    def test_build_config_dict_from_object(self) -> None:
        """Test building config dict from model config object."""
        from sglang_omni.model_runner import model_worker

        quant_config = FakeQuantizationConfig(
            quant_method="auto-round",
            bits=4,
        )
        hf_config = SimpleNamespace(
            quantization_config=quant_config,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=None,
        )

        config_dict = model_worker._build_config_dict(model_config)

        assert config_dict is not None
        assert "quantization_config" in config_dict

    def test_build_config_dict_no_quantization(self) -> None:
        """Test building config dict when no quantization present."""
        from sglang_omni.model_runner import model_worker

        hf_config = SimpleNamespace(
            quantization_config=None,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=None,
        )

        config_dict = model_worker._build_config_dict(model_config)

        # No quantization config found, should return None
        assert config_dict is None


class TestQuantizationIntegrationE2E:
    """End-to-end tests for quantization workflow."""

    def test_full_quantization_detection_flow_fp8(self) -> None:
        """Test complete flow from config to detection."""
        from sglang_omni.quantization import QuantizationConfig, QuantizationRegistry

        # This tests the integration between all components
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "weight_block_size": [128, 128],
            }
        }

        # Step 1: Parse config
        quant_config = QuantizationConfig.from_checkpoint_config(config)
        assert quant_config is not None
        assert quant_config.method == "fp8"

        # Step 2: Detect method
        detected = QuantizationRegistry.detect(config)
        assert detected is not None

        # Step 3: Verify it's the correct type
        fp8_cls = QuantizationRegistry.get("fp8")
        assert isinstance(detected, fp8_cls)

    def test_full_quantization_detection_flow_autoround(self) -> None:
        """Test complete flow from config to detection for AutoRound."""
        from sglang_omni.quantization import QuantizationConfig, QuantizationRegistry

        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "packing_format": "auto_round:auto_gptq",
                "block_name_to_quantize": "transformer_blocks",
            }
        }

        # Step 1: Parse config
        quant_config = QuantizationConfig.from_checkpoint_config(config)
        assert quant_config is not None
        assert quant_config.method == "auto-round"
        assert quant_config.bits == 4
        assert quant_config.packing_format == "auto_round:auto_gptq"

        # Step 2: Detect method
        detected = QuantizationRegistry.detect(config)
        assert detected is not None
        assert detected.name == "auto-round"

    def test_unified_abstraction_all_methods(self) -> None:
        """Test that all registered methods work with unified abstraction."""
        from sglang_omni.quantization import QuantizationRegistry

        methods = QuantizationRegistry.list_supported()

        assert "fp8" in methods
        assert "auto-round" in methods

        # Test that each method can be instantiated and has required methods
        for method_name in methods:
            method_cls = QuantizationRegistry.get(method_name)
            method_instance = method_cls()

            # All methods should have these attributes
            assert hasattr(method_instance, "name")
            assert hasattr(method_instance, "detect")
            assert hasattr(method_instance, "configure")
            assert hasattr(method_instance, "preprocess_weights")
