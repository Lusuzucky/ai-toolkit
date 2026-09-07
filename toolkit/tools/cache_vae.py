import os
import sys
import argparse
from collections import OrderedDict
from typing import Union, List, Optional
import torch

from toolkit.config import get_config
from toolkit.config_modules import ModelConfig, DatasetConfig, TrainConfig, preprocess_dataset_raw_config
from toolkit.util.get_model import get_model_class
from toolkit.data_loader import AiToolkitDataset
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc
from toolkit.basic import flush


def run_cache_vae(config_path_or_dict: Union[str, dict, OrderedDict], device: Optional[str] = None):
    accelerator = get_accelerator()
    if isinstance(config_path_or_dict, (str, dict, OrderedDict)):
        config = get_config(config_path_or_dict)
    else:
        config = config_path_or_dict

    # Find the process config
    process_list = config.get("config", {}).get("process", [])
    if not process_list:
        raise ValueError("Config missing 'config.process' list.")

    # Target first process that defines model/datasets
    process = None
    for p in process_list:
        if "model" in p and "datasets" in p:
            process = p
            break
    if process is None:
        process = process_list[0]

    model_raw = process.get("model", {})
    train_raw = process.get("train", {})
    datasets_raw = process.get("datasets", [])

    if not datasets_raw:
        raise ValueError("No datasets configured in process.")

    model_config = ModelConfig(**model_raw)
    train_config = TrainConfig(**train_raw)

    # Force only VAE loading
    model_config.only_load_vae = True
    if "vae_dtype" not in model_raw:
        model_config.vae_dtype = train_config.dtype
    if not hasattr(model_config, "model_kwargs") or model_config.model_kwargs is None:
        model_config.model_kwargs = {}
    model_config.model_kwargs["only_load_vae"] = True

    ModelClass = get_model_class(model_config)

    run_device = device or accelerator.device
    dtype = train_config.dtype

    print_acc(f"[Cache VAE] Initializing {ModelClass.__name__} for VAE-only caching on {run_device} ({dtype})...")
    model = ModelClass(
        device=run_device,
        model_config=model_config,
        dtype=dtype,
    )
    model.load_model()
    flush()

    # Preprocess dataset configs
    dataset_configs = []
    for raw_ds in datasets_raw:
        split = preprocess_dataset_raw_config([raw_ds])
        for s in split:
            s["cache_latents_to_disk"] = True
            s["cache_latents"] = False
            s["cache_text_embeddings"] = False
            dataset_configs.append(DatasetConfig(**s))

    print_acc(f"[Cache VAE] Starting latents caching for {len(dataset_configs)} dataset(s)...")
    for i, ds_cfg in enumerate(dataset_configs):
        print_acc(f"[Cache VAE] Dataset {i + 1}/{len(dataset_configs)}: {ds_cfg.folder_path}")
        dataset = AiToolkitDataset(ds_cfg, sd=model)
        flush()

    print_acc("[Cache VAE] All dataset latents successfully cached to disk!")


def main():
    parser = argparse.ArgumentParser(description="Standalone VAE caching tool - caches dataset latents to disk without loading Transformer or Text Encoder.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to job yaml config")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device to run VAE on (default: accelerator device)")

    args = parser.parse_args()
    run_cache_vae(args.config, device=args.device)


if __name__ == "__main__":
    main()
