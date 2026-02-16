import argparse

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


def resize_dataframes(
    dataframes: list[pd.DataFrame], exp_size: int, seed: int = 42
) -> list[pd.DataFrame]:
    if exp_size == -1:
        # no resizing
        return dataframes
    if exp_size <= 0:
        raise ValueError("exp_size must be positive.")

    lengths = [len(df) for df in dataframes]
    total = sum(lengths)

    ratio = exp_size / total
    sizes_float = [length * ratio for length in lengths]
    sizes_int = [int(np.floor(x)) for x in sizes_float]

    # distribute remainder by largest decimal
    remainder = exp_size - sum(sizes_int)
    decimals = np.argsort([x - int(x) for x in sizes_float])[::-1]

    for i in decimals[:remainder]:
        sizes_int[i] += 1

    rng = np.random.default_rng(seed)

    resized = []
    for df, target in zip(dataframes, sizes_int, strict=True):
        n = len(df)
        if target <= n:
            # downsample (no replacement, already shuffled)
            sampled = df.sample(
                n=target,
                replace=False,
                random_state=rng.integers(1e9),
            )
        else:
            # upsample: full copies first, then remainder
            k, r = divmod(target, n)
            sampled = pd.concat(
                [df] * k
                + [
                    df.sample(
                        n=r,
                        replace=False,
                        random_state=rng.integers(1e9),
                    )
                ],
                ignore_index=True,
            ).sample(
                frac=1.0,
                random_state=rng.integers(1e9),
            )
        resized.append(sampled.reset_index(drop=True))

    assert sum(len(df) for df in resized) == exp_size
    return resized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter synthetic data JSONL files and emit a cleaned dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input JSONL file(s) to filter.",
    )
    parser.add_argument(
        "--dst-dataset",
        type=str,
        required=True,  # example: "hez2024/cvdp_ecov_train_sampled_dagger_react_r3"
        help="Destination dataset name on Hugging Face Hub.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to upload to the Hub.",
    )
    parser.add_argument("--exp-size", type=int, default=-1, help="Maximum dataset size.")
    args = parser.parse_args()

    input_paths = args.input
    print(f"Loading data from {input_paths}...")
    print("Note: file order will affect shuffling and truncation.")
    dataframes = []
    for path in input_paths:
        df = pd.read_json(path, lines=True)
        dataframes.append(df)
        assert len(df) > 0, f"No data loaded from {path}!"
        print(f"Loaded {len(df)} rows from {path}.")

    print(f"Total dataset size before truncation: {sum(len(df) for df in dataframes)}")
    if args.exp_size > 0:
        print(f"Resizing dataframes to total size {args.exp_size}...")
        dataframes = resize_dataframes(dataframes, exp_size=args.exp_size)

    df = pd.concat(dataframes, ignore_index=True)
    print(f"Total dataset size after truncation: {len(df)}")
    # shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    ds_dict = DatasetDict(
        {
            args.split: Dataset.from_pandas(
                df,
                preserve_index=False,
            )
        }
    )

    # Push to Hugging Face Hub (private)
    ds_dict.push_to_hub(
        args.dst_dataset,
        private=True,
    )


if __name__ == "__main__":
    main()
