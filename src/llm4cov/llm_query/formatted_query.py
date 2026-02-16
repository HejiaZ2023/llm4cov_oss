import asyncio

from llm4cov.datasets.types import DataFile
from llm4cov.llm_query.parse import extract_filename_from_text, extract_verilog_content
from llm4cov.llm_query.patch_apply import Patch, parse_patch
from llm4cov.llm_query.types import LLMQueryStats, OpenAIQueryArgs, openai_query_one_plain


async def query_patchs(
    messages: list[dict],
    query_args: OpenAIQueryArgs,
    max_retries: int = 3,
    debug: bool = False,
) -> tuple[str, Patch, LLMQueryStats]:
    """Send async request and return raw patch content, parsed Patch, and stats."""

    async def query_patchs_inner() -> tuple[str, Patch, LLMQueryStats]:
        content, stats = await openai_query_one_plain(query_args, messages)
        if debug:
            print(content)
        patch = parse_patch(content)
        if not patch.hunks and content.strip() != "NO_CHANGES":
            raise ValueError("Parsed zero hunks from patch response")
        return content, patch, stats

    for attempt in range(max_retries):
        try:
            return await query_patchs_inner()
        except Exception:
            backoff = 2 ** (attempt - 1)
            await asyncio.sleep(backoff)
    return await query_patchs_inner()


async def query_one_file(
    messages: list[dict], query_args: OpenAIQueryArgs, max_retries: int = 3, debug: bool = False
) -> tuple[DataFile, LLMQueryStats]:
    """Send async request and write output to file, with retry/backoff."""

    async def query_one_file_inner() -> tuple[DataFile, LLMQueryStats]:
        # -----------------------------
        # Call vLLM server
        # -----------------------------
        content, stats = await openai_query_one_plain(query_args, messages, debug=debug)
        if debug:
            print(content)
        # -----------------------------
        # Parse tool call
        # -----------------------------
        filename = extract_filename_from_text(content)
        if not filename:
            error_msg = "Failed to extract filename from text"
            if debug:
                error_msg += f":\n{content}"
                print(error_msg)
            raise ValueError(error_msg)
        file_content = extract_verilog_content(content)
        if not file_content:
            error_msg = "Failed to extract Verilog content from text"
            if debug:
                error_msg += f":\n{file_content}"
                print(error_msg)
            raise ValueError(error_msg)

        return DataFile(name=filename, content=file_content), stats  # success

    for attempt in range(max_retries):
        try:
            return await query_one_file_inner()
        except Exception as _:
            # ---------- Retry ----------
            backoff = 2 ** (attempt - 1)
            await asyncio.sleep(backoff)
    # Run one last time, let exception propagate
    return await query_one_file_inner()
