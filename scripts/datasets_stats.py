import numpy as np

from llm4cov.datasets.filter import filter_data_by_length, filter_single_top_data
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import DataContext


def check_dataset_stats(dataset_in: list[DataContext]) -> None:
    # Print reports about the dataset:
    # 1. total count
    # 2. rtl token length distribution: min, max, mean, quartiles
    length = len(dataset_in)
    with_spec_length = sum(1 for ctx in dataset_in if len(ctx.spec_files) > 0)
    print(f"Total dataset entries: {length}, with spec files: {with_spec_length}")
    rtl_lengths = np.array([ctx.rtl_tokens for ctx in dataset_in])
    print(f"RTL token lengths: min={rtl_lengths.min()}, max={rtl_lengths.max()}")
    print(f"RTL token lengths: mean={rtl_lengths.mean():.2f}")
    print(
        f"RTL token lengths: Q1={np.percentile(rtl_lengths, 25)}, "
        f"Q2={np.percentile(rtl_lengths, 50)}, Q3={np.percentile(rtl_lengths, 75)}"
    )


def check_dataset_stats_by_name(dataset_name: str, max_rtl_length: int = 1000) -> None:
    contexts = load_dataset_by_name(dataset_name, split="train")
    print(f"Dataset: {dataset_name}")
    check_dataset_stats(contexts)

    single_top_contexts = filter_single_top_data(contexts)
    print("----After filtering single top module:")
    check_dataset_stats(single_top_contexts)

    short_contexts = filter_data_by_length(single_top_contexts, max_rtl_length, keep_long=False)
    print(f"----After filtering RTL length <= {max_rtl_length}:")
    check_dataset_stats(short_contexts)

    long_contexts = filter_data_by_length(single_top_contexts, max_rtl_length, keep_long=True)
    print(f"----After filtering RTL length > {max_rtl_length}:")
    check_dataset_stats(long_contexts)


def main() -> None:
    dataset_names = [
        "zhuyaoyu/CodeV-R1-dataset",
        "wilyub/VeriThoughtsTrainSet",
    ]
    max_rtl_length = 1000
    for dataset_name in dataset_names:
        check_dataset_stats_by_name(dataset_name, max_rtl_length)
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
