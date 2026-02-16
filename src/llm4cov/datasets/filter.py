from llm4cov.datasets.types import DataContext


def filter_single_top_data(contexts: list[DataContext]) -> list[DataContext]:
    """Filter out DataContext entries that do not have exactly one top module."""
    return [ctx for ctx in contexts if len(ctx.potential_top) == 1]


def filter_data_by_length(
    contexts: list[DataContext], max_rtl_length: int, keep_long: bool
) -> list[DataContext]:
    """Filter out DataContext entries that have RTL files exceeding max_rtl_length."""
    if keep_long:
        return [ctx for ctx in contexts if ctx.rtl_tokens > max_rtl_length]
    else:
        return [ctx for ctx in contexts if ctx.rtl_tokens <= max_rtl_length]
