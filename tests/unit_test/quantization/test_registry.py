# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization registry."""

from __future__ import annotations

import importlib
import sys

import pytest

from sglang_omni.quantization.base import QuantizationMethod
from sglang_omni.quantization.registry import QuantizationRegistry


@pytest.fixture(autouse=True)
def _snapshot_quantization_registry():
    """Snapshot and restore the QuantizationRegistry state around each test."""
    registry_module = importlib.import_module("sglang_omni.quantization.registry")
    registry_cls = registry_module.QuantizationRegistry

    original_methods = dict(registry_cls._methods)
    original_initialized = registry_cls._initialized
    try:
        yield
    finally:
        registry_cls._methods.clear()
        registry_cls._methods.update(original_methods)
        registry_cls._initialized = original_initialized


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

    def test_exact_name_match_requires_detect_confirmation(self) -> None:
        """Exact name match is rejected if detect() returns False.

        This prevents FP8-like names from matching without the required
        weight_block_size field.
        """
        from sglang_omni.quantization.methods.fp8 import FP8Quantization

        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(FP8Quantization)

        # FP8 without weight_block_size should NOT be detected
        # even though the name matches
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
                # Missing weight_block_size
            }
        }

        result = QuantizationRegistry.detect(config)

        assert result is None

    def test_exact_name_match_succeeds_when_detect_passes(self) -> None:
        """Exact name match succeeds when detect() confirms."""
        from sglang_omni.quantization.methods.fp8 import FP8Quantization

        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(FP8Quantization)

        # FP8 with weight_block_size should be detected
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

    def test_detect_warns_when_name_matches_but_detection_fails(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning is logged when method name matches but detection fails."""
        from sglang_omni.quantization.methods.fp8 import FP8Quantization

        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(FP8Quantization)

        config = {
            "quantization_config": {
                "quant_method": "fp8",
                # Missing weight_block_size
            }
        }

        QuantizationRegistry.detect(config)

        assert any(
            "did not confirm support" in record.message for record in caplog.records
        )

    def test_detect_warns_when_no_method_matches(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning is logged when no registered method matches."""
        QuantizationRegistry._methods.clear()
        QuantizationRegistry._initialized = False
        QuantizationRegistry.register(TestQuantizationMethod)

        config = {
            "quantization_config": {
                "quant_method": "completely-unknown-method",
            }
        }

        QuantizationRegistry.detect(config)

        assert any(
            "No registered quantization method matches" in record.message
            for record in caplog.records
        )

    def test_auto_register_all_fails_fast_on_builtin_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken built-in import is propagated, never silently skipped."""
        registry_module = importlib.import_module("sglang_omni.quantization.registry")
        original_methods = dict(registry_module.QuantizationRegistry._methods)
        original_initialized = registry_module.QuantizationRegistry._initialized

        def _raise_import_error(*args, **kwargs):
            raise ImportError("simulated builtin quantization import failure")

        try:
            registry_module.QuantizationRegistry._methods.clear()
            registry_module.QuantizationRegistry._initialized = False
            monkeypatch.setattr(
                registry_module.importlib, "import_module", _raise_import_error
            )

            with pytest.raises(ImportError, match="simulated builtin quantization"):
                registry_module.QuantizationRegistry.auto_register_all()

            assert registry_module.QuantizationRegistry._initialized is False
        finally:
            registry_module.QuantizationRegistry._methods.clear()
            registry_module.QuantizationRegistry._methods.update(original_methods)
            registry_module.QuantizationRegistry._initialized = original_initialized
