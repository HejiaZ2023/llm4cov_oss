from tqdm import tqdm

from llm4cov.datasets.filter import filter_single_top_data
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.rtl_preprocess import context_filter_top_extract
from llm4cov.datasets.types import DataContext


def extract_potential_top_from_context(
    context_list: list[DataContext], detect_mode: bool = False
) -> list[str]:
    error_cases: list[str] = []
    for context in tqdm(context_list, desc="Processing contexts"):
        if detect_mode:
            try:
                context_filter_top_extract(context)
            except ValueError:
                error_cases.append(context.id)
        else:
            context_filter_top_extract(context, debug=True)  # confirm no error
    return error_cases


def test_extract_potential_top_from_dataset(detect_mode: bool = False) -> None:
    dataset_name = "hez2024/cvdp_ecov_eval"
    contexts = load_dataset_by_name(dataset_name, split="eval")
    assert len(contexts) > 0, "No contexts loaded from dataset"
    contexts = filter_single_top_data(contexts)
    ret = extract_potential_top_from_context(contexts, detect_mode=detect_mode)
    if detect_mode and len(ret) > 0:
        print(f"Detected {len(ret)} error cases in dataset {dataset_name}: {ret}")

    dataset_name = "zhuyaoyu/CodeV-R1-dataset"
    contexts = load_dataset_by_name(dataset_name, split="train")
    assert len(contexts) > 0, "No contexts loaded from dataset"
    contexts = filter_single_top_data(contexts)
    ret = extract_potential_top_from_context(contexts, detect_mode=detect_mode)
    if detect_mode and len(ret) > 0:
        print(f"Detected {len(ret)} error cases in dataset {dataset_name}: {ret}")

    dataset_name = "wilyub/VeriThoughtsTrainSet"
    contexts = load_dataset_by_name(dataset_name, split="train")
    assert len(contexts) > 0, "No contexts loaded from dataset"
    contexts = filter_single_top_data(contexts)
    ret = extract_potential_top_from_context(contexts, detect_mode=detect_mode)
    if detect_mode and len(ret) > 0:
        print(f"Detected {len(ret)} error cases in dataset {dataset_name}: {ret}")


# Note: More detailed tests can be added with specific known RTL files and expected top modules.
def main() -> None:
    test_extract_potential_top_from_dataset(detect_mode=False)
    print("All tests passed.")


if __name__ == "__main__":
    main()
