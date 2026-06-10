# SPDX-License-Identifier: Apache-2.0
"""CI tests for Qwen3-Omni AutoRound quantization.

Usage:
    pytest tests/test_model/test_qwen3_omni_quantization_autoround.py -s -x

Requirements:
    - CUDA GPU
    - AutoRound checkpoint: Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound (or env var)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest

from tests.test_model.conftest import QWEN3_OMNI_MODEL_NAME, _model_cache_present
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    print_router_diagnostics,
    print_worker_snapshot,
    router_get_json,
)
from tests.utils import MetricCheckCollector

# AutoRound model path (Intel's quantized checkpoint)
QWEN3_OMNI_AUTOROUND_MODEL_PATH = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
QWEN3_OMNI_AUTOROUND_TEST_MODEL_PATH = os.environ.get(
    "SGLANG_OMNI_TEST_QWEN3_AUTOROUND_MODEL", QWEN3_OMNI_AUTOROUND_MODEL_PATH
)

# Benchmark settings
MAX_SAMPLES = 10
CONCURRENCY = 4
BENCHMARK_TIMEOUT = 300


@dataclass
class QuantizationBenchmarkResult:
    """Results from a quantization benchmark run."""

    summary: dict
    per_sample: list[dict]
    output_dir: str


def _run_text_benchmark(
    port: int,
    output_dir: str,
    max_samples: int = MAX_SAMPLES,
    concurrency: int = CONCURRENCY,
    repo_id: str | None = "zhaochenyang20/mmmu-ci-50",
) -> dict:
    """Run a simple text benchmark and return results."""
    import asyncio

    from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval

    config = MMMUEvalConfig(
        model=QWEN3_OMNI_MODEL_NAME,
        port=port,
        output_dir=output_dir,
        max_samples=max_samples,
        max_concurrency=concurrency,
        repo_id=repo_id,
    )
    return asyncio.run(run_mmmu_eval(config, compute_wer=False))


def _assert_benchmark_results(
    results: dict,
    label: str,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Assert benchmark results meet basic sanity checks."""
    checks = collector or MetricCheckCollector(label)

    summary = results.get("summary", {})
    per_sample = results.get("per_sample", [])

    # Check summary-level invariants
    checks.check(
        summary.get("failed", 1) == 0,
        f"Expected 0 failed requests, got {summary.get('failed')}",
    )

    # Check per-request invariants
    for req in per_sample:
        rid = req.get("sample_id", "<missing>")
        checks.check(
            req.get("is_success") is True,
            f"Request {rid} failed: {req.get('error')}",
        )

    if collector is None:
        checks.assert_all()


def _launch_quantized_router(
    tmp_path_factory: pytest.TempPathFactory,
    model_path: str,
    model_name: str = QWEN3_OMNI_MODEL_NAME,
    worker_args: str = "",
    num_workers: int = 1,
) -> ManagedRouterHandle:
    """Launch router for a quantized Qwen3-Omni model."""
    return launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=model_path,
        model_name=model_name,
        worker_extra_args=worker_args,
        num_workers=num_workers,
        wait_timeout=300,
    )


def _check_model_cache(model_path: str, label: str) -> bool:
    """Check if model is in cache, skip test if not."""
    if not _model_cache_present(model_path):
        pytest.skip(
            f"{label} checkpoint {model_path!r} is not in the local HF cache. "
            f"Pre-populate the cache or set SGLANG_OMNI_TEST_QWEN3_*_MODEL env var to a local path."
        )
    return True


@pytest.fixture(scope="module")
def qwen3_omni_autoround_server(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Launch Qwen3-Omni AutoRound router for testing."""
    _check_model_cache(QWEN3_OMNI_AUTOROUND_TEST_MODEL_PATH, "AutoRound")

    worker_args = "--config examples/configs/qwen3_omni_colocated_h20.yaml --colocate "

    with _launch_quantized_router(
        tmp_path_factory,
        model_path=QWEN3_OMNI_AUTOROUND_TEST_MODEL_PATH,
        worker_args=worker_args,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def autoround_benchmark_results(
    qwen3_omni_autoround_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> QuantizationBenchmarkResult:
    """Run text benchmark on AutoRound model."""
    output_dir = str(tmp_path_factory.mktemp("autoround_benchmark"))

    try:
        results = _run_text_benchmark(
            qwen3_omni_autoround_server.port,
            output_dir,
            max_samples=MAX_SAMPLES,
            concurrency=CONCURRENCY,
        )
    except Exception:
        print_router_diagnostics(qwen3_omni_autoround_server)
        raise

    return QuantizationBenchmarkResult(
        summary=results["summary"],
        per_sample=results["per_sample"],
        output_dir=output_dir,
    )


@pytest.mark.benchmark
def test_autoround_model_loads_and_responds(
    qwen3_omni_autoround_server: ManagedRouterHandle,
) -> None:
    """Test that AutoRound model loads and responds to requests."""
    # Check workers are healthy
    workers = router_get_json(qwen3_omni_autoround_server.port, "/workers")
    print_worker_snapshot("AutoRound /workers snapshot", workers)

    checks = MetricCheckCollector("AutoRound model load")
    checks.check(
        workers["total_workers"] >= 1,
        f"Expected at least 1 worker, got {workers['total_workers']}",
    )
    checks.check(
        workers["healthy_workers"] >= 1,
        f"Expected at least 1 healthy worker, got {workers['healthy_workers']}",
    )
    checks.check(
        workers["routable_workers"] >= 1,
        f"Expected at least 1 routable worker, got {workers['routable_workers']}",
    )

    # Check models endpoint
    models = router_get_json(qwen3_omni_autoround_server.port, "/v1/models")
    model_ids = {card["id"] for card in models.get("data", [])}
    checks.check(
        QWEN3_OMNI_MODEL_NAME in model_ids,
        f"Expected model {QWEN3_OMNI_MODEL_NAME!r} in {model_ids}",
    )

    checks.assert_all()


@pytest.mark.benchmark
def test_quantization_unified_abstraction_autoround() -> None:
    """Test that the unified quantization abstraction works for AutoRound."""
    from sglang_omni.quantization import QuantizationConfig, QuantizationRegistry

    # Test config parsing
    config_dict = {
        "quantization_config": {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "packing_format": "auto_round:auto_gptq",
            "block_name_to_quantize": "transformer_blocks",
        }
    }

    quant_config = QuantizationConfig.from_checkpoint_config(config_dict)
    assert quant_config is not None
    assert quant_config.method == "auto-round"
    assert quant_config.bits == 4
    assert quant_config.group_size == 128
    assert quant_config.sym is True
    assert quant_config.packing_format == "auto_round:auto_gptq"

    # Test registry detection
    detected = QuantizationRegistry.detect(config_dict)
    assert detected is not None
    assert detected.name == "auto-round"

    # Test block name remapping
    checkpoint_names = [
        "transformer_blocks.0.attn.qkv.weight",
        "transformer_blocks.1.mlp.weight",
        "embed_tokens.weight",
    ]
    mapping = detected.remap_block_names(
        checkpoint_names, config_dict["quantization_config"]
    )
    assert len(mapping) == 2  # Only transformer_blocks entries
    assert "embed_tokens.weight" not in mapping


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
