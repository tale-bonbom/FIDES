"""
FLOPs analysis for FIDES pipeline components.

Measures floating-point operations (FLOPs) for each pipeline component
using fvcore's FlopCountAnalysis.

Usage:
    python benchmark/benchmark_flops.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")


def measure_model_flops(model: nn.Module, input_shape: tuple, device: str = "cpu") -> dict:
    """
    Measure FLOPs for a model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        device: Device to run on

    Returns:
        Dictionary with FLOPs count
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(1, *input_shape).to(device)

    if FVCORE_AVAILABLE:
        try:
            flops = FlopCountAnalysis(model, dummy_input)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total_flops = flops.total()
            return {
                "total_flops": int(total_flops),
                "total_gflops": float(total_flops / 1e9),
                "total_mflops": float(total_flops / 1e6),
            }
        except Exception as e:
            print(f"  fvcore analysis failed: {e}")

    # Fallback: estimate FLOPs from conv layers
    return _estimate_flops_manual(model, input_shape)


def _estimate_flops_manual(model: nn.Module, input_shape: tuple) -> dict:
    """Manual FLOPs estimation by counting conv linear operations."""
    total_flops = 0
    model = model.eval()
    dummy = torch.randn(1, *input_shape)

    hooks = []
    layer_flops = []

    def conv_hook(module, input, output):
        # FLOPs for Conv2d: 2 * H_out * W_out * C_in * C_out * K_h * K_w / groups
        batch_size = output.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size[0] * module.kernel_size[1]
        groups = module.groups
        flops = 2 * batch_size * out_h * out_w * in_channels * out_channels * kernel_size / groups
        layer_flops.append(("conv", int(flops)))
        nonlocal total_flops
        total_flops += flops

    def linear_hook(module, input, output):
        batch_size = output.shape[0]
        flops = 2 * batch_size * module.in_features * module.out_features
        layer_flops.append(("linear", int(flops)))
        nonlocal total_flops
        total_flops += flops

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(dummy)

    for hook in hooks:
        hook.remove()

    return {
        "total_flops": int(total_flops),
        "total_gflops": float(total_flops / 1e9),
        "total_mflops": float(total_flops / 1e6),
        "estimated": True,
        "layer_count": len(layer_flops),
    }


def measure_pipeline_flops(input_size: int = 224) -> dict:
    """
    Measure FLOPs for the full FIDES pipeline (model + transforms).

    The pipeline FLOPs are dominated by the model forward pass.
    ARP, PSR, LBU, and TTA add negligible FLOPs (interpolation only).
    """
    from fides.model import MedNeXtL

    model = MedNeXtL(in_channels=1, num_classes=4, input_size=input_size, kernel_size=5)
    input_shape = (1, input_size, input_size)

    model_flops = measure_model_flops(model, input_shape)

    # TTA multiplies FLOPs by the number of augmentations
    tta_flops = {
        "tta_none": model_flops["total_gflops"] * 1,
        "tta_h": model_flops["total_gflops"] * 2,
        "tta_hv": model_flops["total_gflops"] * 3,
        "tta_hvr": model_flops["total_gflops"] * 4,
    }

    return {
        "model_flops": model_flops,
        "tta_flops": tta_flops,
        "input_size": input_size,
    }


def main():
    parser = argparse.ArgumentParser(description="FLOPs Benchmark")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--output", type=str, default="./benchmark_results/flops.json")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  FIDES FLOPs Analysis")
    print(f"{'='*60}\n")

    results = measure_pipeline_flops(args.input_size)

    print(f"  Model: MedNeXt-L (input: {args.input_size}x{args.input_size})")
    print(f"  Total FLOPs:  {results['model_flops']['total_flops']:,}")
    print(f"  Total GFLOPs: {results['model_flops']['total_gflops']:.2f}")
    if results['model_flops'].get('estimated'):
        print(f"  (Estimated from conv/linear layers)")

    print(f"\n  TTA FLOPs scaling:")
    for tta_name, gflops in results['tta_flops'].items():
        print(f"    {tta_name}: {gflops:.2f} GFLOPs")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
