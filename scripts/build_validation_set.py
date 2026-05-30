#!/usr/bin/env python3
"""
Build a validation set from a CodeV-R1-format dataset using 2%-percentile
stratified sampling by RTL token length.

Outputs
-------
--val-output    50-sample validation parquet (CodeV-R1 format)
--train-output  remaining single-top samples excluding val (optional)

Usage (inside paladin llm4cov_slime container):
    docker exec llm4cov_slime bash -c \
      "cd /root/slime/third_party/llm4cov_oss && \
       uv run scripts/build_validation_set.py \
         --dataset hez2024/CodeV-R1-dataset-RL-test \
         --val-output /mnt/raid0_ssd/sheng/stress_test_dataset/val_codev_rl_test.parquet \
         --train-output /mnt/raid0_ssd/sheng/stress_test_dataset/train_codev_rl_test.parquet"
"""
import argparse
import random
from pathlib import Path

import datasets as ds
import pandas as pd

from llm4cov.datasets.filter import filter_single_top_data
from llm4cov.datasets.load import load_dataset_by_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a validation set via percentile-stratified RTL-length sampling."
    )
    p.add_argument("--dataset", required=True,
                   help="HF dataset name, e.g. hez2024/CodeV-R1-dataset-RL-test")
    p.add_argument("--split", default="train")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of validation samples (= number of percentile buckets).")
    p.add_argument("--val-output", required=True, help="Output path for validation parquet.")
    p.add_argument("--train-output", default=None,
                   help="Output path for remaining train parquet (single_top minus val).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # ── Step 1: load via llm4cov to obtain rtl_tokens ────────────────────────
    print(f"[1/4] Loading {args.dataset!r} split={args.split!r} ...")
    all_contexts = load_dataset_by_name(args.dataset, split=args.split)
    print(f"      raw: {len(all_contexts)} samples")

    single_top = filter_single_top_data(all_contexts)
    print(f"      after filter_single_top: {len(single_top)} samples")

    # ── Step 2: percentile-bucket sampling ────────────────────────────────────
    print(f"\n[2/4] Sampling {args.n_samples} samples via {args.n_samples} percentile buckets ...")
    sorted_ctx = sorted(single_top, key=lambda c: c.rtl_tokens)
    n = len(sorted_ctx)
    bucket_size = n / args.n_samples

    sampled_ids: set[str] = set()
    for i in range(args.n_samples):
        lo = int(i * bucket_size)
        hi = int((i + 1) * bucket_size)
        bucket = sorted_ctx[lo:hi]
        chosen = random.choice(bucket)
        sampled_ids.add(chosen.id)

    # ── Step 3: load original HF DataFrame (uses cache, fast) ────────────────
    print("\n[3/4] Loading HF DataFrame for column export ...")
    hf_df = pd.DataFrame(ds.load_dataset(args.dataset, split=args.split))
    id_series = hf_df["problem_id"].astype(str)
    single_top_ids = {c.id for c in single_top}

    # ── Step 4: write outputs ─────────────────────────────────────────────────
    print("\n[4/4] Writing output parquets ...")

    # validation parquet
    df_val = hf_df[id_series.isin(sampled_ids)].reset_index(drop=True)
    if len(df_val) != args.n_samples:
        print(f"  WARNING: expected {args.n_samples} val rows, got {len(df_val)}")
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
    df_val.to_parquet(args.val_output, index=False)
    print(f"  val:   {len(df_val):5d} rows → {args.val_output}")

    # train parquet (single_top minus val)
    if args.train_output:
        train_ids = single_top_ids - sampled_ids
        df_train = hf_df[id_series.isin(train_ids)].reset_index(drop=True)
        Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
        df_train.to_parquet(args.train_output, index=False)
        print(f"  train: {len(df_train):5d} rows → {args.train_output}")

    # summary stats
    val_contexts = [c for c in single_top if c.id in sampled_ids]
    rtl_lens = sorted(c.rtl_tokens for c in val_contexts)
    mid = len(rtl_lens) // 2
    print(f"\nRTL token lengths in val set:")
    print(f"  min={rtl_lens[0]}  p25={rtl_lens[len(rtl_lens)//4]}  "
          f"median={rtl_lens[mid]}  p75={rtl_lens[3*len(rtl_lens)//4]}  max={rtl_lens[-1]}")
    print(f"\nRemaining for training (before contamination check): {n - args.n_samples}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
