"""
Latency and memory benchmarking for FIDES pipeline.

Measures inference latency (ms/slice) and peak GPU memory (MB)
for each pipeline configuration.

Usage:
    python benchmark/benchmark_latency.py
    python benchmark/benchmark_latency.py --device cuda --n_warmup 10 --n_runs 100
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from fides.configs import get_config, CONFIG_NAMES
from fides.pipeline import FIDESPipeline


def measure_latency(
    pipeline: FIDESPipeline,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """
    Measure inference latency.

    Args:
        pipeline: FIDES pipeline instance
        input_tensor: Input image tensor (1, 1, H, W)
        n_warmup: Number of warmup iterations
        n_runs: Number of timed iterations

    Returns:
        Dictionary with latency statistics (ms)
    """
    device = pipeline.device

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = pipeline.predict_slice(input_tensor)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = pipeline.predict_slice(input_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "n_runs": int(n_runs),
    }


def measure_memory(
    pipeline: FIDESPipeline,
    input_tensor: torch.Tensor,
) -> dict:
    """
    Measure peak GPU memory usage.

    Args:
        pipeline: FIDES pipeline instance
        input_tensor: Input image tensor (1, 1, H, W)

    Returns:
        Dictionary with memory statistics (MB)
    """
    device = pipeline.device

    if device != "cuda":
        return {
            "peak_memory_mb": 0,
            "note": "Memory measurement requires CUDA device",
        }

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        _ = pipeline.predict_slice(input_tensor)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB

    return {
        "peak_memory_mb": float(peak_memory),
        "device": "cuda",
    }


def benchmark_all_configs(
    input_size: int = 224,
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """
    Benchmark latency and memory for all pipeline configurations.

    Args:
        input_size: Model input size
        device: Device to run on
        n_warmup: Warmup iterations
        n_runs: Timed iterations

    Returns:
        Dictionary with benchmark results for each config
    """
    # Create a dummy input tensor
    input_tensor = torch.randn(1, 1, input_size, input_size)

    results = {}
    key_configs = ["fides_optimal", "corseg_original", "fides_clinical",
                   "arp_only", "arp_psr_lbu", "arp_psr_lbu_htta"]

    for config_name in key_configs:
        try:
            config = get_config(config_name)
        except ValueError:
            continue

        print(f"\n  Benchmarking: {config_name}...")
        pipeline = FIDESPipeline(config, checkpoint_path=None, device=device)

        latency = measure_latency(pipeline, input_tensor, n_warmup, n_runs)
        memory = measure_memory(pipeline, input_tensor)

        results[config_name] = {
            "latency": latency,
            "memory": memory,
            "config": {
                "arp": config.arp,
                "psr": config.psr,
                "lbu": config.lbu,
                "tta": config.tta,
            },
        }

        print(f"    Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
        if memory.get("peak_memory_mb", 0) > 0:
            print(f"    Peak Memory: {memory['peak_memory_mb']:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Latency & Memory Benchmark")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--output", type=str, default="./benchmark_results/latency.json")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  FIDES Latency & Memory Benchmark")
    print(f"{'='*60}")
    print(f"  Device: {args.device}")
    print(f"  Input: {args.input_size}x{args.input_size}")
    print(f"  Warmup: {args.n_warmup} | Runs: {args.n_runs}")

    results = benchmark_all_configs(
        input_size=args.input_size,
        device=args.device,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
    )

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {'Config':<25} {'Latency (ms)':<20} {'Memory (MB)':<15}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        lat = f"{r['latency']['mean_ms']:.2f} ± {r['latency']['std_ms']:.2f}"
        mem = f"{r['memory'].get('peak_memory_mb', 0):.1f}" if r['memory'].get('peak_memory_mb', 0) > 0 else "N/A"
        print(f"  {name:<25} {lat:<20} {mem:<15}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
