# SPDX-License-Identifier: Apache-2.0
"""CI tests for Qwen3-Omni FP8 quantization.

Usage:
    pytest tests/test_model/test_qwen3_omni_quantization_fp8.py -s -x

Requirements:
    - CUDA GPU
    - FP8 checkpoint: marksverdhei/Qwen3-Omni-30B-A3B-FP8 (or env var)
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

# FP8 model path (QuantKitchen quantized checkpoint)
QWEN3_OMNI_FP8_MODEL_PATH = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
QWEN3_OMNI_FP8_TEST_MODEL_PATH = os.environ.get(
    "SGLANG_OMNI_TEST_QWEN3_FP8_MODEL", QWEN3_OMNI_FP8_MODEL_PATH
)

# Benchmark settings
MAX_SAMPLES = 10
CONCURRENCY = 4
BENCHMARK_TIMEOUT = 300


@dataclass
class FP8QuantizationBenchmarkResult:
    """Results from an FP8 quantization benchmark run."""

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


def _launch_fp8_router(
    tmp_path_factory: pytest.TempPathFactory,
    model_path: str,
    model_name: str = QWEN3_OMNI_MODEL_NAME,
    worker_args: str = "",
    num_workers: int = 1,
) -> ManagedRouterHandle:
    """Launch router for an FP8 quantized Qwen3-Omni model."""
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
            f"Pre-populate the cache or set SGLANG_OMNI_TEST_QWEN3_FP8_MODEL env var to a local path."
        )
    return True


@pytest.fixture(scope="module")
def qwen3_omni_fp8_server(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Launch Qwen3-Omni FP8 router for testing."""
    _check_model_cache(QWEN3_OMNI_FP8_TEST_MODEL_PATH, "FP8")

    # Use FP8 colocated config
    # FP8 quantized checkpoints include quantization metadata (weight_scale_inv)
    # which increases memory usage. We use the FP8-specific colocated config
    # that tunes memory fractions for the FP8 workload.
    worker_args = "--config examples/configs/qwen3_omni_fp8_colocated.yaml"

    with _launch_fp8_router(
        tmp_path_factory,
        model_path=QWEN3_OMNI_FP8_TEST_MODEL_PATH,
        worker_args=worker_args,
    ) as router:
        yield router


@pytest.fixture(scope="module")
def fp8_benchmark_results(
    qwen3_omni_fp8_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> FP8QuantizationBenchmarkResult:
    """Run text benchmark on FP8 model."""
    output_dir = str(tmp_path_factory.mktemp("fp8_benchmark"))

    try:
        results = _run_text_benchmark(
            qwen3_omni_fp8_server.port,
            output_dir,
            max_samples=MAX_SAMPLES,
            concurrency=CONCURRENCY,
        )
    except Exception:
        print_router_diagnostics(qwen3_omni_fp8_server)
        raise

    return FP8QuantizationBenchmarkResult(
        summary=results["summary"],
        per_sample=results["per_sample"],
        output_dir=output_dir,
    )


@pytest.mark.benchmark
def test_fp8_model_loads_and_responds(
    qwen3_omni_fp8_server: ManagedRouterHandle,
) -> None:
    """Test that FP8 model loads and responds to requests."""
    # Check workers are healthy
    workers = router_get_json(qwen3_omni_fp8_server.port, "/workers")
    print_worker_snapshot("FP8 /workers snapshot", workers)

    checks = MetricCheckCollector("FP8 model load")
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
    models = router_get_json(qwen3_omni_fp8_server.port, "/v1/models")
    model_ids = {card["id"] for card in models.get("data", [])}
    checks.check(
        QWEN3_OMNI_MODEL_NAME in model_ids,
        f"Expected model {QWEN3_OMNI_MODEL_NAME!r} in {model_ids}",
    )

    checks.assert_all()


@pytest.mark.benchmark
def test_quantization_unified_abstraction_fp8() -> None:
    """Test that the unified quantization abstraction works for FP8."""
    from sglang_omni.quantization import QuantizationConfig, QuantizationRegistry

    # Test config parsing
    config_dict = {
        "quantization_config": {
            "quant_method": "fp8",
            "bits": 8,
            "group_size": 128,
            "weight_block_size": [128, 128],
        }
    }

    quant_config = QuantizationConfig.from_checkpoint_config(config_dict)
    assert quant_config is not None
    assert quant_config.method == "fp8"
    assert quant_config.bits == 8
    assert quant_config.group_size == 128
    assert quant_config.is_block_quantization is True

    # Test registry detection
    detected = QuantizationRegistry.detect(config_dict)
    assert detected is not None
    assert detected.name == "fp8"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
