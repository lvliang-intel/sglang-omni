# SPDX-License-Identifier: Apache-2.0
"""Built-in quantization methods.

This module auto-registers all supported quantization methods when imported.
"""

# Import all methods to trigger registration via @QuantizationRegistry.register
from sglang_omni.quantization.methods import autoround  # noqa: F401
from sglang_omni.quantization.methods import fp8  # noqa: F401

__all__ = [
    "fp8",
    "autoround",
]
