import argparse
import asyncio
import json
import logging
import random as rd
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from llm4cov.datasets.eval import display_eval_stats, eval_cov_result_against_expectations
from llm4cov.datasets.filter import filter_data_by_length, filter_single_top_data
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import (
    CovResult,
    DataFile,
    LlmGenTbContext,
    data_context_to_llm_gen_tb_context,
)
from llm4cov.eda_client.remote_exec import run_remote_cov_job_pipeline
from llm4cov.eda_client.remote_sync import LOCAL_TMP_DIR
from llm4cov.llm_query.formatted_query import OpenAIQueryArgs, query_one_file
from llm4cov.llm_query.prompt_build import build_initial_prompt_from_context
from llm4cov.llm_query.types import LLMQueryStats
from llm4cov.syn_data_gen.framework import RunContext, run_pipeline_queue_workers
from llm4cov.syn_data_gen.prompt_inject import inject_prompt_into_tb_generation

SERVER = "paladin_centos"
MODEL_SIZE = "S"
# MODEL = (
#    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
#    if MODEL_SIZE == "L"
#    else "Qwen/Qwen3-Coder-30B-A3B-Instruct"
# )
MODEL = "qwen3_4b_full_8g_3b_1e"
TOKENIZER_DIR = "/mnt/raid0_ssd/hejia/sft_backup/qwen3_4b_full_8g_3b_1e/checkpoint-1268"
REMOTE_REPO_DIR = "/workspace/llm4cov_eda"

BASE_URL = "http://localhost"
API_KEY = "EMPTY"

LLM_SINGLE_QUERY_TIMEOUT_S = 720
LLM_RETRIES = 3
LLM_JOB_TIMEOUT_S = LLM_SINGLE_QUERY_TIMEOUT_S * (LLM_RETRIES + 1) + 30
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
# LLM_MAX_COMPLETION_TOKENS = 131072
LLM_MAX_COMPLETION_TOKENS = 8192

GEN_REPETITIONS = 2

EDA_SINGLE_STAGE_TIMEOUT_S = 30
EDA_STAGES = 3
EDA_JOB_TIMEOUT_S = EDA_SINGLE_STAGE_TIMEOUT_S * EDA_STAGES + 30

DEBUG = False
DEBUG_SAMPLE = 4

MANUAL = False

EVAL = False

GEN_USE_CODEV_R1 = True
GEN_USE_VERITHOUGHT = False
GEN_THRESHOLD = 1000
GEN_KEEP_LONG = True
GEN_SAMPLE = 0  # 0 means no limit

LOG_DEBUG = False

SYN_PROMPT_INJECTION = False

PORT = 11451
DEFAULT_ORCHESTRATOR_WORKERS = 256
DEFAULT_LLM_CONCURRENCY = 192
DEFAULT_EDA_CONCURRENCY = 32

logging.basicConfig(
    level=logging.DEBUG if LOG_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
if not LOG_DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------
def load_sample_dataset(
    limit: int, start: int | None, end: int | None, syn_prompt_injection: bool
) -> list[LlmGenTbContext]:
    if MANUAL:
        codev_r1_dataset = load_dataset_by_name("zhuyaoyu/CodeV-R1-dataset", split="train")
        target_dataset_total = filter_single_top_data(codev_r1_dataset)
        target_dataset_short = filter_data_by_length(
            target_dataset_total, max_rtl_length=1000, keep_long=False
        )
        target_dataset_short = rd.sample(target_dataset_short, k=1000)
        # print(len(target_dataset_short))
        target_dataset_long = filter_data_by_length(
            target_dataset_total, max_rtl_length=1000, keep_long=True
        )
        target_dataset_long = rd.sample(target_dataset_long, k=1000)
        # print(len(target_dataset_long))
        target_dataset_raw = target_dataset_short + target_dataset_long
        # print(len(target_dataset_raw))
    elif EVAL:
        cvdp_ecov_dataset = load_dataset_by_name("hez2024/cvdp_ecov_eval", split="eval")
        # (Hejia, 12/21/2025): Actually you can just return cvdp_ecov_dataset here
        # Just kept code structure for possible future extension
        target_dataset_raw = filter_single_top_data(cvdp_ecov_dataset)
    else:
        target_dataset_raw = []
        if GEN_USE_CODEV_R1:
            codev_r1_dataset = load_dataset_by_name("zhuyaoyu/CodeV-R1-dataset", split="train")
            target_dataset_raw += filter_single_top_data(codev_r1_dataset)
        if GEN_USE_VERITHOUGHT:
            verithought_dataset = load_dataset_by_name("wilyub/VeriThoughtsTrainSet", split="train")
            target_dataset_raw += filter_single_top_data(verithought_dataset)
        target_dataset_raw = filter_data_by_length(
            target_dataset_raw, max_rtl_length=GEN_THRESHOLD, keep_long=GEN_KEEP_LONG
        )
    if limit > 0 and len(target_dataset_raw) > limit:
        target_dataset_raw = rd.sample(target_dataset_raw, k=limit)
    if start is not None or end is not None:
        target_dataset_raw = target_dataset_raw[start:end]

    target_dataset = [data_context_to_llm_gen_tb_context(ctx) for ctx in target_dataset_raw]
    if syn_prompt_injection:
        target_dataset = [inject_prompt_into_tb_generation(ctx) for ctx in target_dataset]
    return target_dataset


# ----------------------------------------------------------------------
# Workflow (parallelized per item)
# ----------------------------------------------------------------------
async def vanilla_workflow(ctx: RunContext, context: LlmGenTbContext) -> dict[str, Any]:
    """
    One item end-to-end:
      LLM → TB generation → remote coverage job
    """

    # --- LLM step ---
    messages = build_initial_prompt_from_context(context)

    query_args = OpenAIQueryArgs(
        client=ctx.shared["client"],
        model=ctx.shared["model"],
        temperature=ctx.shared["llm_temperature"],
        top_p=ctx.shared["llm_top_p"],
        max_completion_tokens=ctx.shared["llm_max_completion_tokens"],
        timeout_seconds=ctx.shared["llm_single_query_timeout_s"],
        tokenizer_dir=ctx.shared["tokenizer_dir"],
    )

    tb_file, stats = await ctx.run_llm(
        query_one_file,
        messages,
        query_args,
        max_retries=LLM_RETRIES,
        debug=False,
        timeout_s=ctx.shared["llm_job_timeout_s"],
        label="LLM:gen_tb",
    )
    assert isinstance(stats, LLMQueryStats)

    if not isinstance(tb_file, DataFile):
        # Semantic failure: let it propagate and be recorded by control plane
        raise RuntimeError(f"TB generation failed for {context.dataset_name}:{context.id}")

    # --- EDA step ---
    result: dict[str, Any] = await ctx.run_eda(
        run_remote_cov_job_pipeline,
        server=ctx.shared["server"],
        eda_repo_dir=REMOTE_REPO_DIR,
        context=context,
        tb_file=tb_file,
        timeout=EDA_SINGLE_STAGE_TIMEOUT_S,
        timeout_s=EDA_JOB_TIMEOUT_S,
        label="EDA:cov",
    )

    assert "local_work_dir" in result, "local_work_dir should be in the result"
    local_work_dir = result.get("local_work_dir")
    assert isinstance(local_work_dir, Path), "local_work_dir should be a Path"
    # Save reasoning content in local_work_dir
    if stats.reasoning_content:
        reasoning_file = local_work_dir / "llm_reasoning.txt"
        with open(reasoning_file, "w") as f:
            f.write(stats.reasoning_content)
    # local_work_dir = local_id_dir / task_id
    local_id_dir = local_work_dir.parent
    task_hash_id = local_work_dir.name

    eval_result: CovResult | None = None
    eval_result = eval_cov_result_against_expectations(context, result)

    return {
        "context_id": context.id,
        "dataset": context.dataset_name,
        "tb_file": str(tb_file),
        "llm_stats": stats,
        "eda_result": result,
        "eval_result": eval_result,
        "local_id_dir": str(local_id_dir),
        "task_hash_id": task_hash_id,
    }


# ----------------------------------------------------------------------
# Main (parallel execution)
# ----------------------------------------------------------------------
async def main() -> None:
    logging.info(f"Writing to {LOCAL_TMP_DIR}...")
    parser = argparse.ArgumentParser(description="Run vanilla batch queries.")
    parser.add_argument("--server", type=str, default=SERVER, help="Remote server.")
    parser.add_argument("--model", type=str, default=MODEL, help="Model name.")
    parser.add_argument("--tokenizer-dir", type=str, default=TOKENIZER_DIR, help="Tokenizer path.")
    parser.add_argument("--base-url", type=str, default=BASE_URL, help="Base URL.")
    parser.add_argument("--port", type=int, default=PORT, help="API port.")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="API key.")
    parser.add_argument(
        "--llm-single-query-timeout-s",
        type=float,
        default=LLM_SINGLE_QUERY_TIMEOUT_S,
        help="LLM single query timeout in seconds.",
    )
    parser.add_argument(
        "--llm-max-completion-tokens",
        type=int,
        default=LLM_MAX_COMPLETION_TOKENS,
        help="Max completion tokens per LLM request.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=LLM_TEMPERATURE,
        help="LLM sampling temperature.",
    )
    parser.add_argument("--llm-top-p", type=float, default=LLM_TOP_P, help="LLM top-p.")
    parser.add_argument(
        "--limit",
        type=int,
        default=GEN_SAMPLE,
        help="Limit number of items for debugging (0 means no limit).",
    )
    parser.add_argument("--start", type=int, default=None, help="Start index for slicing items.")
    parser.add_argument("--end", type=int, default=None, help="End index for slicing items.")
    parser.add_argument(
        "--gen-repetitions",
        type=int,
        default=GEN_REPETITIONS,
        help="Repeat each item N times for best-of-N selection.",
    )
    parser.add_argument(
        "--orchestrator-workers",
        type=int,
        default=DEFAULT_ORCHESTRATOR_WORKERS,
        help="Number of orchestrator workers.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=DEFAULT_LLM_CONCURRENCY,
        help="LLM concurrency.",
    )
    parser.add_argument(
        "--eda-concurrency",
        type=int,
        default=DEFAULT_EDA_CONCURRENCY,
        help="EDA concurrency.",
    )
    parser.add_argument(
        "--syn-prompt-injection",
        action="store_true",
        help="Use synthetic prompt injection for TB generation.",
    )
    args = parser.parse_args()

    if args.limit != 0 and (args.start is not None or args.end is not None):
        raise ValueError("Cannot use --limit with --start/--end slicing.")

    target_dataset = load_sample_dataset(
        args.limit, args.start, args.end, args.syn_prompt_injection
    )

    # Repeat each item multiple times for better stats; keep order
    if args.gen_repetitions > 1:
        target_dataset = sum([[ctx] * args.gen_repetitions for ctx in target_dataset], [])

    llm_job_timeout_s = args.llm_single_query_timeout_s * (LLM_RETRIES + 1) + 30
    client = AsyncOpenAI(
        base_url=f"{args.base_url}:{args.port}/v1",
        api_key=args.api_key,
    )

    results, stats = await run_pipeline_queue_workers(
        target_dataset,
        workflow=vanilla_workflow,
        orchestrator_workers=args.orchestrator_workers,
        llm_concurrency=args.llm_concurrency,
        eda_concurrency=args.eda_concurrency,
        llm_timeout_s=llm_job_timeout_s,
        eda_timeout_s=EDA_JOB_TIMEOUT_S,
        shared={
            "client": client,
            "server": args.server,
            "model": args.model,
            "tokenizer_dir": args.tokenizer_dir,
            "llm_single_query_timeout_s": args.llm_single_query_timeout_s,
            "llm_job_timeout_s": llm_job_timeout_s,
            "llm_max_completion_tokens": args.llm_max_completion_tokens,
            "llm_temperature": args.llm_temperature,
            "llm_top_p": args.llm_top_p,
        },
        include_traceback=False,
        log_stage_timing=DEBUG,
    )

    # --- Post processing ---
    print("\n=== Summary ===")
    cov_results: list[tuple[str, str, CovResult]] = []
    for r in results:
        if not r.ok:
            print(f"❌ FAILED: {r.error}")
        else:
            assert r.value is not None
            result: dict[str, Any] = r.value["eda_result"]
            eda_status = result.get("status")
            assert isinstance(r.value["eval_result"], CovResult)
            cov_results.append(
                (r.value["local_id_dir"], r.value["task_hash_id"], r.value["eval_result"])
            )
            if DEBUG:
                print(f"✅ OK: {r.value['dataset']}:{r.value['context_id']} | cov={eda_status}")

    print(
        f"\nFinished {stats.finished}/{stats.total}, "
        f"failed={stats.failed}, "
        f"elapsed={stats.elapsed():.1f}s"
    )

    cov_stats = display_eval_stats(cov_results)
    states_file = LOCAL_TMP_DIR / "eval_stats.json"
    with open(states_file, "w") as f:
        json.dump(cov_stats, f, indent=4)
    print(f"Eval stats written to {states_file}")


if __name__ == "__main__":
    asyncio.run(main())
