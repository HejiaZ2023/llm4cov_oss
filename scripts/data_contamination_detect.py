#!/usr/bin/env python3

import argparse
from collections.abc import Iterable
from dataclasses import dataclass

from tqdm import tqdm

from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import DataContext, DataFile


@dataclass(frozen=True)
class FileTokens:
    name: str
    tokens: list[str]

    @property
    def length(self) -> int:
        return len(self.tokens)


def tokenize(text: str) -> list[str]:
    return text.split()


def iter_context_files(context: DataContext) -> Iterable[DataFile]:
    yield from context.spec_files
    yield from context.rtl_files


def build_context_index(contexts: list[DataContext]) -> dict[str, list[FileTokens]]:
    index: dict[str, list[FileTokens]] = {}
    for context in contexts:
        files = []
        for data_file in iter_context_files(context):
            files.append(FileTokens(name=data_file.name, tokens=tokenize(data_file.content)))
        index[context.id] = files
    return index


def lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a
    prev = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        curr = [0]
        for j, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(curr[-1] if curr[-1] >= prev[j] else prev[j])
        prev = curr
    return prev[-1]


def rouge_l_f1(tokens_a: list[str], tokens_b: list[str]) -> float:
    total = len(tokens_a) + len(tokens_b)
    if total == 0:
        return 1.0
    lcs = lcs_length(tokens_a, tokens_b)
    if lcs == 0:
        return 0.0
    return 2.0 * lcs / total


def contexts_similar(
    train_files: list[FileTokens],
    eval_files: list[FileTokens],
    threshold: float,
) -> bool:
    for eval_file in eval_files:
        for train_file in train_files:
            min_len = (
                eval_file.length if eval_file.length < train_file.length else train_file.length
            )
            max_possible = (
                2.0 * min_len / (eval_file.length + train_file.length) if min_len else 0.0
            )
            if max_possible < threshold:
                continue
            if rouge_l_f1(eval_file.tokens, train_file.tokens) >= threshold:
                return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect data contamination by Rouge-L similarity between eval and train items."
        )
    )
    parser.add_argument(
        "--eval-dataset",
        required=True,
        help="Eval dataset name (e.g. hez2024/cvdp_ecov_eval).",
    )
    parser.add_argument(
        "--train-dataset",
        required=True,
        help="Train dataset name (e.g. zhuyaoyu/CodeV-R1-dataset).",
    )
    parser.add_argument(
        "--eval-split",
        default="eval",
        help="Eval dataset split name.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Train dataset split name.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Rouge-L F1 similarity threshold.",
    )
    parser.add_argument(
        "--output",
        default="data_contamination.tsv",
        help="Output file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_contexts = load_dataset_by_name(args.eval_dataset, split=args.eval_split)
    train_contexts = load_dataset_by_name(args.train_dataset, split=args.train_split)

    eval_index = build_context_index(eval_contexts)
    train_index = build_context_index(train_contexts)

    with open(args.output, "w") as fout:
        for train_context in tqdm(train_contexts, desc="Train contexts"):
            train_files = train_index[train_context.id]
            for eval_context in tqdm(eval_contexts, desc="Eval contexts", leave=False):
                eval_files = eval_index[eval_context.id]
                if contexts_similar(train_files, eval_files, args.threshold):
                    line = f"{train_context.id}\t{eval_context.id}"
                    print(line)
                    fout.write(line + "\n")
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
