import argparse

import pandas as pd
from transformers import AutoTokenizer

CONTAMINATION_RECORD = """
7890    cvdp_copilot_static_branch_predict_0035
56047   cvdp_copilot_cdc_pulse_synchronizer_0017
67690   cvdp_agentic_bcd_adder_0006
108839  cvdp_agentic_bcd_adder_0006
90776   cvdp_agentic_bcd_adder_0006
119799  cvdp_agentic_bcd_adder_0006
126448  cvdp_agentic_bcd_adder_0006
139665  cvdp_agentic_bcd_adder_0006
152965  cvdp_agentic_bcd_adder_0006
156770  cvdp_agentic_bcd_adder_0006
164818  cvdp_agentic_bcd_adder_0006
"""

CONTAMINATION_IDS = [
    7890,
    56047,
    67690,
    108839,
    90776,
    119799,
    126448,
    139665,
    152965,
    156770,
    164818,
]


def filter_invalid_rows(row: pd.Series) -> bool:
    return (
        (row["cov_result"] is not None)
        and (row["prev_cov_result"] is not None)
        and row["cov_result"]["has_coverage"]
    )


def has_absolute_improvement(row: pd.Series, threshold: float = 0.01) -> bool:
    # return    (xrun fail -> pass)
    #           or  (coverage increase > threshold and misc not decrease)
    prev_cov = row["prev_cov_result"]
    curr_cov = row["cov_result"]
    if (curr_cov is None) or (not curr_cov["has_coverage"]):
        return False
    if not prev_cov["has_coverage"]:
        # xrun fixed
        return True
    if curr_cov["overall_coverage"] <= prev_cov["overall_coverage"] + threshold:
        return False
    prev_misc = prev_cov["misc"]
    curr_misc = curr_cov["misc"]
    for key in prev_misc:
        if prev_misc[key] is None:
            continue
        if key not in curr_misc or curr_misc[key] is None:
            return False
        if curr_misc[key] < prev_misc[key]:
            # Each misc item should at least not decrease
            return False
    return True


def get_new_run_type(valid_row: pd.Series) -> str:
    if not valid_row["prev_cov_result"]["has_coverage"]:
        return "agentic_xrun"
    else:
        return "agentic_coverage"


def _get_system(row: pd.Series) -> str:
    assert row["messages_input"][0]["role"] == "system"
    ret = row["messages_input"][0]["content"]
    assert isinstance(ret, str), "System message content should be a string."
    return ret


def _get_history(row: pd.Series) -> list[list[str]]:
    history: list[list[str]] = []
    current_pair: list[str] = []
    for msg in row["messages_input"][1:]:
        assert len(current_pair) < 2
        if len(current_pair) == 0:
            assert msg["role"] == "user"
            current_pair.append(msg["content"])
            continue
        if msg["role"] == "user":  # continue user message, merge by concat
            current_pair[0] += "\n" + msg["content"]
            continue
        assert msg["role"] == "assistant"
        current_pair.append(msg["content"])
        history.append(current_pair)
        current_pair = []
    return history


def _get_instruction(row: pd.Series) -> str:
    # Reserve search to last non-system non-assistant message
    user_start_idx = None
    for idx in range(len(row["messages_input"]) - 1, -1, -1):
        if (
            row["messages_input"][idx]["role"] == "system"
            or row["messages_input"][idx]["role"] == "assistant"
        ):
            user_start_idx = idx + 1
            break
    assert user_start_idx is not None and user_start_idx < len(row["messages_input"])
    content = "\n".join([x["content"] for x in row["messages_input"][user_start_idx:]])
    return content


def _get_output(row: pd.Series) -> str:
    assert row["message_output"]["role"] == "assistant"
    ret = row["message_output"]["content"]
    assert isinstance(ret, str), "Output content should be a string."
    return ret


def check_max_length_context(df_sampled_final: pd.DataFrame) -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    def gen_context(row: pd.Series) -> str:
        ret = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["system"]},
                *[
                    {"role": "user", "content": pair[0]}
                    if idx % 2 == 0
                    else {"role": "assistant", "content": pair[1]}
                    for pair in row["history"]
                    for idx in range(2)
                ],
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(ret, str), "Generated context should be a string."
        return ret

    contexts = df_sampled_final.apply(lambda row: gen_context(row), axis=1)
    str_counts = contexts.apply(lambda x: len(x))
    top_1_percent_contexts = contexts[str_counts >= str_counts.quantile(0.99)]
    token_counts = top_1_percent_contexts.apply(lambda x: len(tokenizer.encode(x)))
    print(f"Max token count in top 1% longest chars: {token_counts.max()}")

    word_counts = contexts.apply(lambda x: len(x.split()))
    top_1_percent_contexts = contexts[word_counts >= word_counts.quantile(0.99)]
    token_counts = top_1_percent_contexts.apply(lambda x: len(tokenizer.encode(x)))
    print(f"Max token count in top 1% longest words: {token_counts.max()}")


def parse_dataframe(
    df: pd.DataFrame, threshold: float, context_length_label: str = "long"
) -> pd.DataFrame:
    print(f"Initial rows count: {len(df)}")
    df = df[~df["context_id"].isin(CONTAMINATION_IDS)].copy()
    df = df[df.apply(filter_invalid_rows, axis=1)]
    print(f"Valid rows count: {len(df)}")
    df = df[df.apply(lambda row: has_absolute_improvement(row, threshold), axis=1)]
    print(f"Rows with absolute improvement count: {len(df)}")
    assert df["dataset"].nunique() == 1, "Only one dataset is expected per file."
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Context_ids: {df['context_id'].nunique()}")
    df["cmp_cov"] = df.apply(
        lambda row: row["cov_result"]["overall_coverage"]
        - row["prev_cov_result"]["overall_coverage"],
        axis=1,
    )
    df = df.loc[df.groupby(["dataset", "context_id"])["cmp_cov"].idxmax()].copy()
    print(f"Rows after selecting max coverage improvement per context_id: {len(df)}")
    df.loc[:, "run_type"] = df.apply(get_new_run_type, axis=1)
    df.loc[:, "context_length_label"] = context_length_label
    df.rename(columns={"input": "messages_input", "output": "message_output"}, inplace=True)
    df["system"] = df.apply(lambda row: _get_system(row), axis=1)
    df["history"] = df.apply(lambda row: _get_history(row), axis=1)
    df["instruction"] = df.apply(lambda row: _get_instruction(row), axis=1)
    df["output"] = df.apply(lambda row: _get_output(row), axis=1)

    df_final = df[
        [
            "dataset",
            "context_id",
            "run_type",
            "context_length_label",
            "system",
            "history",
            "instruction",
            "output",
        ]
    ]
    check_max_length_context(df_final)
    return df_final


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
        "--context-length-label",
        type=str,
        default="long",
        help="Context length label to attach to each row.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="Coverage improvement threshold."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    input_paths = args.input
    dataframes = []
    for path in input_paths:
        df = pd.read_json(path, lines=True)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    df_final = parse_dataframe(
        df, threshold=args.threshold, context_length_label=args.context_length_label
    )
    df_final.to_json(args.output, lines=True, orient="records")


if __name__ == "__main__":
    main()
