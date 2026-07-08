# SPDX-License-Identifier: Apache-2.0
"""Tests for the quant-method / composite-model registries introduced by the
registry-based refactor: `QuantMethodSpec`, `PreprocessorContext`,
`CompositeModelSpec`, `register_quant_method`, `register_composite_model`, and
the depth-first `resolve_quant_config` traversal.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang_omni import quantization as quant


@pytest.fixture(autouse=True)
def _isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against copies of the module-level registries so test
    registrations never leak into other tests or the built-in defaults."""
    monkeypatch.setattr(
        quant, "_QUANT_METHOD_REGISTRY", dict(quant._QUANT_METHOD_REGISTRY)
    )
    monkeypatch.setattr(
        quant, "_COMPOSITE_MODEL_REGISTRY", dict(quant._COMPOSITE_MODEL_REGISTRY)
    )
    monkeypatch.setattr(
        quant, "_NESTED_QUANT_CONFIG_ATTRS", list(quant._NESTED_QUANT_CONFIG_ATTRS)
    )


class TestPreprocessorContext:
    """`PreprocessorContext` replaces the untyped options dict."""

    def test_defaults(self) -> None:
        assert quant.PreprocessorContext().fp8_scale_inverted is False

    def test_is_frozen(self) -> None:
        context = quant.PreprocessorContext(fp8_scale_inverted=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.fp8_scale_inverted = False  # type: ignore[misc]


class TestQuantMethodSpec:
    """`QuantMethodSpec` is a pure name -> behavior mapping (no `matches`)."""

    def test_defaults(self) -> None:
        spec = quant.QuantMethodSpec(name="unit-test-defaults")
        assert spec.needs_stage_normalization is False
        assert spec.build_preprocessor is None

    def test_has_no_matches_field(self) -> None:
        # Per-method predicates live inside build_preprocessor now, not as a
        # separate ``matches`` callback on the spec.
        field_names = {f.name for f in dataclasses.fields(quant.QuantMethodSpec)}
        assert "matches" not in field_names


class TestRegisterQuantMethod:
    def test_registers_new_method(self) -> None:
        quant.register_quant_method(quant.QuantMethodSpec(name="unit-test-method"))
        assert "unit-test-method" in quant._QUANT_METHOD_REGISTRY

    def test_overwrites_rather_than_merges(self) -> None:
        quant.register_quant_method(
            quant.QuantMethodSpec(
                name="unit-test-method", needs_stage_normalization=True
            )
        )
        quant.register_quant_method(quant.QuantMethodSpec(name="unit-test-method"))

        spec = quant._QUANT_METHOD_REGISTRY["unit-test-method"]
        assert spec.needs_stage_normalization is False

    def test_unregistered_method_is_identity_and_no_normalization(self) -> None:
        preprocess = quant.get_weight_preprocessor(
            quant_dict={"quant_method": "unregistered-method"}
        )
        weight = torch.tensor([1.0, 2.0])
        assert torch.equal(preprocess("layer.weight", weight), weight)
        assert not quant.needs_quant_config_normalization(
            quant_dict={"quant_method": "unregistered-method"}
        )

    def test_new_method_drives_weight_preprocessor_and_normalization(self) -> None:
        """A hypothetical future method (e.g. AWQ) needs only one registration
        call -- no branching added anywhere else in the module."""

        def _build_awq_preprocessor(
            quant_dict: dict, context: quant.PreprocessorContext
        ) -> quant.WeightPreprocessor:
            if context.fp8_scale_inverted:
                return quant.convert_fp8_weight_scale_inv
            return quant._identity_preprocessor

        quant.register_quant_method(
            quant.QuantMethodSpec(
                name="awq",
                needs_stage_normalization=True,
                build_preprocessor=_build_awq_preprocessor,
            )
        )

        assert quant.needs_quant_config_normalization(
            quant_dict={"quant_method": "awq"}
        )

        preprocess = quant.get_weight_preprocessor(
            quant_dict={"quant_method": "awq"}, fp8_scale_inverted=True
        )
        scale = torch.tensor([2.0])
        assert torch.allclose(
            preprocess("layer.weight_scale_inv", scale), torch.tensor([0.5])
        )


class TestCompositeModelSpec:
    def test_defaults(self) -> None:
        spec = quant.CompositeModelSpec(arch="UnitTestArch")
        assert spec.checkpoint_prefix is None
        assert spec.nested_config_attr is None


class TestRegisterCompositeModel:
    def test_registers_new_architecture(self) -> None:
        quant.register_composite_model(
            "UnitTestArch",
            checkpoint_prefix="stage.",
            nested_config_attr="unit_test_stage_config",
        )

        spec = quant._COMPOSITE_MODEL_REGISTRY["UnitTestArch"]
        assert spec.checkpoint_prefix == "stage."
        assert spec.nested_config_attr == "unit_test_stage_config"
        assert "unit_test_stage_config" in quant._NESTED_QUANT_CONFIG_ATTRS

    def test_overwrites_rather_than_merges(self) -> None:
        quant.register_composite_model(
            "UnitTestArch",
            checkpoint_prefix="stage.",
            nested_config_attr="unit_test_stage_config",
        )
        # Re-registering without nested_config_attr must drop it, not keep the
        # previous value -- registration replaces, it does not merge.
        quant.register_composite_model("UnitTestArch", checkpoint_prefix="other.")

        spec = quant._COMPOSITE_MODEL_REGISTRY["UnitTestArch"]
        assert spec.checkpoint_prefix == "other."
        assert spec.nested_config_attr is None

    def test_does_not_duplicate_nested_config_attr(self) -> None:
        quant.register_composite_model(
            "ArchA", nested_config_attr="shared_config"
        )
        quant.register_composite_model(
            "ArchB", nested_config_attr="shared_config"
        )

        assert quant._NESTED_QUANT_CONFIG_ATTRS.count("shared_config") == 1


class TestResolveQuantConfigDeepNesting:
    """`resolve_quant_config` depth-first searches to arbitrary nesting depth."""

    def test_finds_config_nested_three_levels_deep(self) -> None:
        quant.register_composite_model(
            "UnitTestDeepArch", nested_config_attr="deep_stage_config"
        )

        leaf = SimpleNamespace(
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
        mid = SimpleNamespace(quantization_config=None, deep_stage_config=leaf)
        root = SimpleNamespace(quantization_config=None, text_config=mid)

        result = quant.resolve_quant_config(root)
        assert result is not None
        assert result["quant_method"] == "fp8"

    def test_cyclic_config_does_not_infinite_loop(self) -> None:
        node = SimpleNamespace(quantization_config=None)
        node.text_config = node  # self-reference

        assert quant.resolve_quant_config(node) is None
