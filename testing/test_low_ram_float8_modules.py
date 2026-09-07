import os
import sys
import tempfile
import unittest
import torch
from safetensors.torch import save_file, load_file

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from toolkit.tools.quantize_model import quantize_state_dict_to_float8
from toolkit.memory_management import manager_modules


class TestLowRamFloat8Modules(unittest.TestCase):
    def test_01_quantize_state_dict_to_float8(self):
        """Verify streaming float8 quantization tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "mock_model.safetensors")
            output_path = os.path.join(tmpdir, "mock_model_fp8.safetensors")

            # Create mock weights: linear weight, bias, norm weight
            tensors = {
                "transformer.layers.0.attn.qkv.weight": torch.randn(64, 64, dtype=torch.bfloat16),
                "transformer.layers.0.attn.qkv.bias": torch.randn(64, dtype=torch.bfloat16),
                "transformer.norm.weight": torch.randn(64, dtype=torch.bfloat16),
                "transformer.first.weight": torch.randn(64, 64, dtype=torch.bfloat16), # should be excluded by "first" pattern
            }
            save_file(tensors, input_path)

            quantize_state_dict_to_float8(
                input_path=input_path,
                output_path=output_path,
                target_dtype=torch.float8_e4m3fn,
                fallback_dtype=torch.bfloat16,
            )

            self.assertTrue(os.path.exists(output_path))
            loaded = load_file(output_path)

            # Check qkv.weight is float8
            self.assertEqual(loaded["transformer.layers.0.attn.qkv.weight"].dtype, torch.float8_e4m3fn)
            # Check bias is bfloat16
            self.assertEqual(loaded["transformer.layers.0.attn.qkv.bias"].dtype, torch.bfloat16)
            # Check norm is bfloat16
            self.assertEqual(loaded["transformer.norm.weight"].dtype, torch.bfloat16)
            # Check excluded "first" is bfloat16
            self.assertEqual(loaded["transformer.first.weight"].dtype, torch.bfloat16)
            print("\n[PASS] test_01_quantize_state_dict_to_float8 passed successfully.")

    def test_02_no_pin_memory(self):
        """Verify AI_TOOLKIT_NO_PIN_MEMORY prevents cloning mmap tensors to locked RAM."""
        os.environ["AI_TOOLKIT_NO_PIN_MEMORY"] = "1"
        # Reload NO_PIN_MEMORY flag
        manager_modules.NO_PIN_MEMORY = True

        dummy_tensor = torch.randn(10, 10)
        pinned = manager_modules._ensure_cpu_pinned(dummy_tensor)
        self.assertFalse(pinned.is_pinned())
        print("[PASS] test_02_no_pin_memory passed successfully.")

    def test_03_extension_registration(self):
        """Verify quantize_model, cache_vae, and cache_te are registered in process dict."""
        from toolkit.extension import get_all_extensions_process_dict
        pdict = get_all_extensions_process_dict()

        self.assertIn("quantize_model", pdict)
        self.assertIn("cache_vae", pdict)
        self.assertIn("cache_te", pdict)
        print("[PASS] test_03_extension_registration passed successfully.")


if __name__ == "__main__":
    unittest.main()
