#!/usr/bin/env python3
"""Single-process sequential pre-generator for CIFAR10-DVS frame datasets across timesteps T."""

import argparse
import os
import time

from spikingjelly.datasets import cifar10_dvs


def main():
    parser = argparse.ArgumentParser(description="Pre-generate CIFAR10-DVS frame caches safely in one process")
    parser.add_argument("--data-path", default="/data/rboone/datasets/wg_dvst")
    parser.add_argument("--timesteps", default="4,8,10,24,32")
    cli = parser.parse_args()

    timesteps = [int(x) for x in cli.timesteps.split(",") if x.strip()]
    print(f"Pre-generating CIFAR10-DVS frames in {cli.data_path} for T={timesteps}...", flush=True)

    header = f"{'T':>4s} | {'samples':>8s} | {'shape':>22s} | {'elapsed_s':>9s} | {'cache_dir'}"
    print(header)
    print("-" * len(header))

    for T in timesteps:
        t0 = time.perf_counter()
        ds = cifar10_dvs.CIFAR10DVS(
            root=cli.data_path,
            data_type="frame",
            frames_number=T,
            split_by="number",
        )
        n_samples = len(ds)
        img, _ = ds[0]
        elapsed = time.perf_counter() - t0
        expected_shape = (T, 2, 128, 128)
        assert n_samples == 10000, f"T={T}: expected 10000 samples, got {n_samples}"
        assert tuple(img.shape) == expected_shape, f"T={T}: expected shape {expected_shape}, got {tuple(img.shape)}"
        cache_dir = os.path.join(cli.data_path, f"frames_number_{T}_split_by_number")
        assert os.path.isdir(cache_dir), f"Expected cache directory missing: {cache_dir}"
        print(f"{T:4d} | {n_samples:8d} | {str(tuple(img.shape)):>22s} | {elapsed:9.2f} | {cache_dir}", flush=True)

    print("All CIFAR10-DVS frame caches ready and verified.")


if __name__ == "__main__":
    main()
