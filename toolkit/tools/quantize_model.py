import os
import sys
import argparse
from typing import Optional, List, Dict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from toolkit.print import print_acc


DEFAULT_EXCLUDE_PATTERNS = [
    "first",
    "tmlp",
    "tproj",
    "txtmlp",
    "txtfusion",
    "last",
    "norm",
    "embed",
    "bias",
    "scale",
    "time_in",
    "vector_in",
    "guidance_in",
    "final_layer",
]


def should_exclude(tensor_name: str, exclude_patterns: List[str]) -> bool:
    name_lower = tensor_name.lower()
    for pattern in exclude_patterns:
        if pattern.lower() in name_lower:
            return True
    return False


def quantize_state_dict_to_float8(
    input_path: str,
    output_path: str,
    exclude_patterns: Optional[List[str]] = None,
    target_dtype: torch.dtype = torch.float8_e4m3fn,
    fallback_dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    """
    Streamingly quantizes weights from an input safetensors file to float8,
    keeping non-linear/sensitive weights in fallback_dtype.
    Minimal memory footprint: processes tensors one by one.
    """
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS

    print_acc(f"[Quantizer] Loading weights from: {input_path}")
    print_acc(f"[Quantizer] Target quantization dtype: {target_dtype}")
    print_acc(f"[Quantizer] Output path: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    quantized_dict: Dict[str, torch.Tensor] = {}
    quantized_count = 0
    kept_count = 0

    with safe_open(input_path, framework="pt", device=device) as f:
        keys = list(f.keys())
        pbar = tqdm(keys, desc="Quantizing tensors to float8")

        for k in pbar:
            tensor = f.get_tensor(k)
            # Only quantize 2D linear weights that are not in exclude patterns
            is_linear_weight = tensor.ndim == 2 and k.endswith(".weight")
            exclude = should_exclude(k, exclude_patterns)

            if is_linear_weight and not exclude:
                # Convert to float8
                max_val = torch.finfo(target_dtype).max
                clamped = tensor.to(torch.float32).clamp(min=-max_val, max=max_val)
                q_tensor = clamped.to(target_dtype)
                quantized_dict[k] = q_tensor
                quantized_count += 1
            else:
                # Keep in fallback floating point or original format
                if tensor.is_floating_point():
                    quantized_dict[k] = tensor.to(fallback_dtype)
                else:
                    quantized_dict[k] = tensor
                kept_count += 1

    metadata = {
        "quantization_type": "float8_e4m3fn",
        "quantized_by": "ai-toolkit",
        "quantized_tensors": str(quantized_count),
        "kept_tensors": str(kept_count),
    }

    print_acc(f"[Quantizer] Saving {len(quantized_dict)} tensors ({quantized_count} quantized to float8, {kept_count} kept in {fallback_dtype})...")
    save_file(quantized_dict, output_path, metadata=metadata)
    print_acc(f"[Quantizer] Done! Quantized model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize model weights to float8 or other formats with low RAM overhead.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input safetensors file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output safetensors file path")
    parser.add_argument("--qtype", "-q", type=str, default="float8", choices=["float8"], help="Quantization type (default: float8)")
    parser.add_argument("--exclude", "-e", nargs="*", default=None, help="Additional substring patterns of layer names to exclude from quantization")
    parser.add_argument("--device", "-d", type=str, default="cpu", help="Device for conversion math ('cpu' or 'cuda')")

    args = parser.parse_args()

    exclude = DEFAULT_EXCLUDE_PATTERNS
    if args.exclude:
        exclude = exclude + args.exclude

    quantize_state_dict_to_float8(
        input_path=args.input,
        output_path=args.output,
        exclude_patterns=exclude,
        device=args.device,
    )


if __name__ == "__main__":
    main()
