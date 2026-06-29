# SPDX-License-Identifier: Apache-2.0
"""Quantization method registry."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Type, overload

if TYPE_CHECKING:
    from .base import QuantizationMethod

logger = logging.getLogger(__name__)


# Built-in method modules to attempt to import during auto-registration.
_BUILTIN_METHOD_MODULES: tuple[str, ...] = (
    "autoround",
    "fp8",
)


class QuantizationRegistry:
    """Registry for quantization methods.

    Registration is lazy.

    Usage:
        # Register a new quantization method
        @QuantizationRegistry.register
        class AutoRoundQuantization(QuantizationMethod):
            name = "auto-round"
            ...

        # Detect quantization from config
        method = QuantizationRegistry.detect(config)
    """

    _methods: dict[str, Type["QuantizationMethod"]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_builtins_registered(cls) -> None:
        """Import built-in method modules on first use.

        Safe to call repeatedly: a guard short-circuits after the first run.
        """
        if cls._initialized:
            return
        cls.auto_register_all()

    @overload
    @classmethod
    def register(
        cls,
        method_cls: Type["QuantizationMethod"],
    ) -> Type["QuantizationMethod"]: ...
    @overload
    @classmethod
    def register(
        cls,
        method_cls: None = ...,
    ) -> Callable[[Type["QuantizationMethod"]], Type["QuantizationMethod"]]: ...
    @classmethod
    def register(
        cls,
        method_cls: Type["QuantizationMethod"] | None = None,
    ) -> (
        Type["QuantizationMethod"]
        | Callable[[Type["QuantizationMethod"]], Type["QuantizationMethod"]]
    ):
        """Register a quantization method.
        Can be used as a decorator or called directly.
        """

        def decorator(
            impl_cls: Type["QuantizationMethod"],
        ) -> Type["QuantizationMethod"]:
            if not impl_cls.name:
                raise ValueError(f"{impl_cls.__name__} must define a non-empty 'name'")
            if impl_cls.name in cls._methods:
                logger.warning(
                    f"Overriding existing quantization method: {impl_cls.name}"
                )
            cls._methods[impl_cls.name] = impl_cls
            logger.debug(f"Registered quantization method: {impl_cls.name}")
            return impl_cls

        if method_cls is not None:
            # Called directly: @QuantizationRegistry.register
            return decorator(method_cls)
        else:
            # Called as decorator: @QuantizationRegistry.register()
            return decorator

    @classmethod
    def get(cls, name: str) -> Type["QuantizationMethod"]:
        """Get a registered quantization method."""
        cls._ensure_builtins_registered()
        if name not in cls._methods:
            raise KeyError(
                f"Unknown quantization method: {name!r}. "
                f"Available: {list(cls._methods.keys())}"
            )
        return cls._methods[name]

    @classmethod
    def detect(cls, config: dict[str, Any]) -> "QuantizationMethod | None":
        """Detect quantization method from model config."""
        cls._ensure_builtins_registered()
        quant_config = config.get("quantization_config")
        if quant_config is None:
            return None

        quant_method = quant_config.get("quant_method")
        if quant_method is None:
            return None

        # Normalize method name for matching
        normalized = str(quant_method).lower().replace("-", "_").replace(" ", "_")

        name_matched_method: Type["QuantizationMethod"] | None = None
        for name, method_cls in cls._methods.items():
            name_normalized = name.lower().replace("-", "_").replace(" ", "_")
            if name_normalized == normalized:
                name_matched_method = method_cls
                if cls._detect(method_cls, config):
                    return method_cls()
                break

        # Fall back to detection by class
        for method_cls in cls._methods.values():
            if method_cls is name_matched_method:
                continue
            if cls._detect(method_cls, config):
                return method_cls()

        if name_matched_method is not None:
            logger.warning(
                "Quantization method %r is known but %r did not confirm support "
                "for this checkpoint variant.",
                quant_method,
                name_matched_method.name,
            )
        else:
            logger.warning(f"No registered quantization method matches: {quant_method}")
        return None

    @staticmethod
    def _detect(method_cls: Type["QuantizationMethod"], config: dict[str, Any]) -> bool:
        """Call a method's detector."""
        return bool(method_cls.detect(config))

    @classmethod
    def detect_by_name(cls, method_name: str) -> "QuantizationMethod":
        """Get a quantization method by name."""
        return cls.get(method_name)()

    @classmethod
    def list_supported(cls) -> list[str]:
        """List all registered quantization methods."""
        cls._ensure_builtins_registered()
        return list(cls._methods.keys())

    @classmethod
    def auto_register_all(cls) -> None:
        """Auto-register all built-in quantization methods.

        Safe to call repeatedly: a guard short-circuits after the first run.
        """
        if cls._initialized:
            return

        package = "sglang_omni.quantization.methods"
        for module_name in _BUILTIN_METHOD_MODULES:
            importlib.import_module(f".{module_name}", package=package)

        cls._initialized = True
        logger.info(f"Registered quantization methods: {list(cls._methods.keys())}")
