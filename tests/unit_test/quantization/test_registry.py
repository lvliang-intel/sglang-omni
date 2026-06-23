# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization registry."""

from __future__ import annotations

import importlib
import sys

import pytest

from sglang_omni.quantization.base import QuantizationMethod
from sglang_omni.quantization.registry import QuantizationRegistry


# Create a test quantization method for testing the registry
class TestQuantizationMethod(QuantizationMethod):
    """Test implementation of QuantizationMethod."""

    name = "test-quant"

    @classmethod
    def detect(cls, config):
        return config.get("quantization_config", {}).get("quant_method") == "test-quant"

    def configure(self, server_args, model_config):
        pass


class AnotherTestMethod(QuantizationMethod):
    """Another test implementation."""

    name = "another-test"

    @classmethod
    def detect(cls, config):
        return (
            config.get("quantization_config", {}).get("quant_method") == "another-test"
        )

    def configure(self, server_args, model_config):
        pass


class TestQuantizationRegistry:
    """Tests for QuantizationRegistry."""

    def test_register_method(self) -> None:
        """Test registering a quantization method."""
        # Clear any existing registration
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False

        # Register should work
        QuantizationRegistry.register(TestQuantizationMethod)

        assert "test-quant" in QuantizationRegistry.list_supported()
        assert QuantizationRegistry.get("test-quant") == TestQuantizationMethod

    def test_register_decorator(self) -> None:
        """Test registering via decorator."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False

        # Using as decorator
        @QuantizationRegistry.register
        class DecoratorTestMethod(QuantizationMethod):
            name = "decorator-test"

            @classmethod
            def detect(cls, config):
                return False

            def configure(self, server_args, model_config):
                pass

        assert "decorator-test" in QuantizationRegistry.list_supported()

    def test_register_decorator_with_parens(self) -> None:
        """Test registering via decorator with parentheses."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False

        # Using as decorator with parens
        @QuantizationRegistry.register()
        class DecoratorParensTestMethod(QuantizationMethod):
            name = "decorator-parens-test"

            @classmethod
            def detect(cls, config):
                return False

            def configure(self, server_args, model_config):
                pass

        assert "decorator-parens-test" in QuantizationRegistry.list_supported()

    def test_get_unknown_method_raises(self) -> None:
        """Test that getting an unknown method raises KeyError."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False

        with pytest.raises(KeyError, match="Unknown quantization method"):
            QuantizationRegistry.get("unknown-method")

    def test_detect_fp8(self) -> None:
        """Test detecting FP8 quantization."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(TestQuantizationMethod)

        # Import the actual FP8 method to test detection
        from sglang_omni.quantization.methods.fp8 import FP8Quantization

        QuantizationRegistry.register(FP8Quantization)

        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                "weight_block_size": [128, 128],
            }
        }

        result = QuantizationRegistry.detect(config)

        assert result is not None
        assert isinstance(result, FP8Quantization)

    def test_detect_autoround(self) -> None:
        """Test detecting AutoRound quantization."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False

        from sglang_omni.quantization.methods.autoround import AutoRoundQuantization

        QuantizationRegistry.register(AutoRoundQuantization)

        config = {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
            }
        }

        result = QuantizationRegistry.detect(config)

        assert result is not None
        assert isinstance(result, AutoRoundQuantization)

    def test_detect_no_quantization(self) -> None:
        """Test when no quantization is present."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False

        config = {"model_type": "qwen3"}

        result = QuantizationRegistry.detect(config)

        assert result is None

    def test_detect_unknown_method(self) -> None:
        """Test detecting unknown quantization method."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(TestQuantizationMethod)

        config = {
            "quantization_config": {
                "quant_method": "unknown-quant",
            }
        }

        result = QuantizationRegistry.detect(config)

        assert result is None

    def test_detect_by_name(self) -> None:
        """Test getting method by name."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(TestQuantizationMethod)

        result = QuantizationRegistry.detect_by_name("test-quant")

        assert isinstance(result, TestQuantizationMethod)

    def test_auto_register_all(self) -> None:
        """Test auto-registering all built-in methods."""
        module_names = [
            "sglang_omni.quantization.registry",
            "sglang_omni.quantization.methods",
            "sglang_omni.quantization.methods.fp8",
            "sglang_omni.quantization.methods.autoround",
            "sglang_omni.quantization",
        ]
        original_modules = {
            name: sys.modules.get(name) for name in module_names if name in sys.modules
        }

        try:
            registry_module = importlib.import_module(
                "sglang_omni.quantization.registry"
            )
            registry_module.QuantizationRegistry._methods.clear()
            registry_module.QuantizationRegistry._initialized = False

            for module_name in module_names[1:]:
                sys.modules.pop(module_name, None)

            QuantizationRegistry = registry_module.QuantizationRegistry
            QuantizationRegistry.auto_register_all()

            supported = QuantizationRegistry.list_supported()
            assert "fp8" in supported
            assert "auto-round" in supported
        finally:
            for module_name in module_names:
                original_module = original_modules.get(module_name)
                if original_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = original_module

    def test_register_without_name_raises(self) -> None:
        """Test that registering without a name raises ValueError."""

        class NoNameMethod(QuantizationMethod):
            @classmethod
            def detect(cls, config):
                return False

            def configure(self, server_args, model_config):
                pass

        with pytest.raises(ValueError, match="must define a non-empty 'name'"):
            QuantizationRegistry.register(NoNameMethod)
