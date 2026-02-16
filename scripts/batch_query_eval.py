import argparse
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from llm4cov.datasets.eval import eval_cov_result_against_expectations
from llm4cov.datasets.filter import filter_single_top_data
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
from llm4cov.llm_query.prompt_build import (
    build_initial_prompt_from_context,
    build_react_followup_prompt,
)
from llm4cov.llm_query.types import AsyncOpenAICredentialsRefresher
from llm4cov.syn_data_gen.framework import RunContext, run_pipeline_queue_workers
from llm4cov.syn_data_gen.prompt_inject import inject_prompt_into_tb_generation

DEFAULT_SERVER = "paladin_centos"
DEFAULT_REMOTE_REPO_DIR = "/workspace/llm4cov_eda"

DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_COMPLETION_TOKENS = 8192
DEFAULT_LLM_SINGLE_QUERY_TIMEOUT_S = 240
DEFAULT_LLM_RETRIES = 3
DEFAULT_REACT_ROUNDS = 3

DEFAULT_EDA_SINGLE_STAGE_TIMEOUT_S = 30
DEFAULT_EDA_STAGES = 3
DEFAULT_EDA_JOB_TIMEOUT_S = DEFAULT_EDA_SINGLE_STAGE_TIMEOUT_S * DEFAULT_EDA_STAGES + 30

DEFAULT_ORCHESTRATOR_WORKERS = 256
DEFAULT_LLM_CONCURRENCY = 128
DEFAULT_EDA_CONCURRENCY = 16

LOG_DEBUG = False

logging.basicConfig(
    level=logging.DEBUG if LOG_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
if not LOG_DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)


@dataclass(frozen=True)
class EvalItem:
    context: LlmGenTbContext
    repeat_index: int


@dataclass(frozen=True)
class ResumeEvalItem(EvalItem):
    vanilla_local_work_dir: Path


OUTPUT_FORMAT_REQ = """
OUTPUT REQUIREMENTS:

1. You MUST explicitly state the filename for the testbench in plain text using format:
     filename: tb_xxxx.sv
2. After stating the filename, you MUST output the complete testbench
    inside a fenced SystemVerilog code block:
```systemverilog
module tb_example;
  ...
endmodule
```
"""


def _summarize_eda_result(result: dict[str, Any]) -> dict[str, str]:
    status = str(result.get("status", "unknown"))
    err_msg = str(result.get("err_msg", ""))
    if status == "xrun_failed":
        return {"status": status, "stage": "xrun", "log": err_msg}
    if status != "success":
        return {"status": status, "stage": "imc", "log": err_msg}
    cov_info = result.get("cov_info", {})
    detail = ""
    if isinstance(cov_info, dict):
        detail = str(cov_info.get("detail", ""))
    return {
        "status": status,
        "stage": "success",
        "coverage": detail,
    }


def _accumulate_llm_stats(
    stats_items: list[dict[str, int | float] | None],
) -> dict[str, int | float]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }
    for stats in stats_items:
        if not stats:
            continue
        totals["prompt_tokens"] += int(stats.get("prompt_tokens", 0))
        totals["completion_tokens"] += int(stats.get("completion_tokens", 0))
        totals["reasoning_tokens"] += int(stats.get("reasoning_tokens", 0))
        totals["total_tokens"] += int(stats.get("total_tokens", 0))
        totals["latency_seconds"] += float(stats.get("latency_seconds", 0.0))
    return totals


def _build_client(base_url: str, port: int, api_key: str) -> AsyncOpenAI:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    full_url = f"{base_url}:{port}/v1"
    return AsyncOpenAI(base_url=full_url, api_key=api_key)


def _load_dataset(
    dataset_name: str,
    split: str,
    *,
    limit: int,
    start_idx: int,
    end_idx: int,
    prompt_injection: bool,
) -> list[LlmGenTbContext]:
    dataset = load_dataset_by_name(dataset_name, split=split)
    dataset = filter_single_top_data(dataset)
    items = list(dataset)
    if start_idx or end_idx:
        start_idx = max(start_idx, 0)
        end_idx = end_idx if end_idx > 0 else len(items)
        items = items[start_idx:end_idx]
    if limit > 0 and len(items) > limit:
        items = random.sample(items, limit)
    contexts = [data_context_to_llm_gen_tb_context(ctx) for ctx in items]
    if prompt_injection:
        contexts = [inject_prompt_into_tb_generation(ctx) for ctx in contexts]
    return contexts


def _find_vanilla_tb(task_dir: Path, context: LlmGenTbContext) -> DataFile:
    assert task_dir.is_dir()
    rtl_names = {f.name for f in context.rtl_files}
    candidates = [p for p in task_dir.iterdir() if p.is_file() and p.name not in rtl_names]
    candidates = [p for p in candidates if p.suffix in {".sv", ".v", ".vh"}]
    assert len(candidates) == 1, f"Expected one TB file in {task_dir}, found: {candidates}"
    tb_path = candidates[0]
    with tb_path.open("r", encoding="utf-8") as f:
        content = f.read()
    return DataFile(name=tb_path.name, content=content)


def _find_vanilla_result_json(task_dir: Path) -> Path:
    assert task_dir.is_dir()
    candidates = [p for p in task_dir.iterdir() if p.is_file() and p.name.endswith("_result.json")]
    assert len(candidates) == 1, f"Expected one result json in {task_dir}, found: {candidates}"
    return candidates[0]


def _load_vanilla_result_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid result json: {path}")
    return data


def _discover_vanilla_runs(vanilla_run_dir: Path) -> dict[tuple[str, str], list[Path]]:
    run_map: dict[tuple[str, str], list[Path]] = {}
    for result_path in vanilla_run_dir.rglob("*_result.json"):
        run_dir = result_path.parent
        context_dir = run_dir.parent
        dataset_rel = context_dir.relative_to(vanilla_run_dir)
        assert len(dataset_rel.parts) == 3, (
            f"Unexpected structure: {dataset_rel}, should be 3 parts"
        )
        dataset_name = "/".join(dataset_rel.parts[:-1])
        context_id = context_dir.name
        run_map.setdefault((dataset_name, context_id), []).append(run_dir)
    return run_map


async def eval_workflow(ctx: RunContext, item: EvalItem) -> dict[str, Any]:
    context = item.context
    query_args: OpenAIQueryArgs = ctx.shared["query_args"]
    react_query_args: OpenAIQueryArgs = ctx.shared["react_query_args"]
    use_separate_react_client: bool = ctx.shared["use_separate_react_client"]
    llm_retries: int = ctx.shared["llm_retries"]
    llm_job_timeout_s: float = ctx.shared["llm_job_timeout_s"]
    eda_job_timeout_s: float = ctx.shared["eda_job_timeout_s"]
    eda_single_stage_timeout_s: float = ctx.shared["eda_single_stage_timeout_s"]
    server: str = ctx.shared["server"]
    remote_repo_dir: str = ctx.shared["remote_repo_dir"]
    query_debug: bool = ctx.shared["query_debug"]
    react_mode: str = ctx.shared["react_mode"]
    react_rounds: int = ctx.shared["react_rounds"]
    save_round_inputs: bool = ctx.shared["save_round_inputs"]

    messages = build_initial_prompt_from_context(context)
    best_cov_result = CovResult(id=context.id)
    stats_history: list[dict[str, int | float] | None] = []
    latest_eda_result: dict[str, Any] | None = None
    latest_tb_file: DataFile | None = None
    local_id_dir: Path | None = None
    task_hash_id: str | None = None
    round_debug_info: list[dict[str, Any]] = []
    total_rounds = 1 if react_mode == "none" else (1 + react_rounds)
    markov_snapshot: list[dict[str, str]] = []
    use_markov_snapshot = react_mode in ["markov-react", "markov-react-only"]

    try:
        for round_idx in range(total_rounds):
            if react_mode not in ["none", "react", "markov-react", "markov-react-only"]:
                raise ValueError(f"Unknown react mode: {react_mode}")
            if use_markov_snapshot and round_idx > 1:
                messages_snapshot = list(markov_snapshot)
            else:
                messages_snapshot = list(messages)

            round_debug_info.append(
                {
                    "round_index": round_idx,
                    "messages": messages_snapshot,
                    "best_cov_result": best_cov_result.model_dump(),
                }
            )

            if react_mode == "markov-react-only" and round_idx == 0:
                assert isinstance(item, ResumeEvalItem), (
                    f"Vanilla run missing for {context.dataset_name}:{context.id}"
                )
                vanilla_dir = item.vanilla_local_work_dir
                tb_file = _find_vanilla_tb(vanilla_dir, context)
                result_path = _find_vanilla_result_json(vanilla_dir)
                latest_eda_result = _load_vanilla_result_json(result_path)
                latest_eda_result["local_work_dir"] = str(vanilla_dir)
                stats_history.append(None)
            else:
                round_query_args = (
                    react_query_args
                    if (use_separate_react_client and round_idx > 0)
                    else query_args
                )

                tb_file, stats = await ctx.run_llm(
                    query_one_file,
                    messages_snapshot,
                    round_query_args,
                    max_retries=llm_retries,
                    debug=query_debug,
                    timeout_s=llm_job_timeout_s,
                    label=f"LLM:gen_tb_round_{round_idx}",
                )
                stats_history.append(stats.model_dump() if stats else None)

                if not isinstance(tb_file, DataFile):
                    raise RuntimeError(
                        f"TB generation failed for {context.dataset_name}:{context.id}"
                    )

                latest_eda_result = await ctx.run_eda(
                    run_remote_cov_job_pipeline,
                    server=server,
                    eda_repo_dir=remote_repo_dir,
                    context=context,
                    tb_file=tb_file,
                    timeout=eda_single_stage_timeout_s,
                    timeout_s=eda_job_timeout_s,
                    label=f"EDA:cov_round_{round_idx}",
                )

            latest_tb_file = tb_file
            response_text = f"filename: {tb_file.name}\n```systemverilog\n{tb_file.content}\n```"
            messages.append({"role": "assistant", "content": response_text})

            local_work_dir = latest_eda_result.get("local_work_dir")
            if isinstance(local_work_dir, Path):
                local_id_dir = local_work_dir.parent
                task_hash_id = local_work_dir.name
                latest_eda_result["local_work_dir"] = str(local_work_dir)
            elif isinstance(item, ResumeEvalItem) and round_idx == 0:
                local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
                local_id_dir.mkdir(parents=True, exist_ok=True)
                task_hash_id = item.vanilla_local_work_dir.name
            elif local_id_dir is None:
                local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
                local_id_dir.mkdir(parents=True, exist_ok=True)
                task_hash_id = task_hash_id or "unknown"

            eval_result = eval_cov_result_against_expectations(context, latest_eda_result)

            updated_best = False
            if best_cov_result <= eval_result:
                best_cov_result = eval_result
                updated_best = True

            if react_mode == "none" or round_idx >= total_rounds - 1:
                continue

            instruction = "Improve coverage toward 100% by editing the testbench."
            eda_status = (
                _summarize_eda_result(latest_eda_result) if latest_eda_result is not None else None
            )
            if eda_status:
                if eda_status.get("status") == "xrun_failed":
                    instruction = "Fix the xrun failure by editing the testbench."
                elif eda_status.get("status") != "success":
                    instruction = (
                        "Fix the coverage run failure (imc stage) by editing the testbench."
                    )

            followup_messages = build_react_followup_prompt(
                parse_status={"status": "success", "type": "file"},  # Exception handled above
                apply_status=None,
                eda_status=eda_status,
                tb_content=None,
                instruction=instruction + OUTPUT_FORMAT_REQ,
                is_single_message=True,
            )
            assert len(followup_messages) == 1
            messages.extend(followup_messages)

            if use_markov_snapshot and updated_best:
                assert len(messages) >= 4, "Not enough messages for Markov snapshot"
                # Only update markov snapshot when we have a new best result
                markov_snapshot = [messages[0], messages[1], messages[-2], messages[-1]]
            if eval_result.overall_coverage >= 1.0:
                assert eval_result.is_pass_xrun, "100% coverage must pass xrun"
                break  # Stop early if we have perfect coverage
    except Exception as e:
        if local_id_dir is None:
            local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
            local_id_dir.mkdir(parents=True, exist_ok=True)
        if save_round_inputs and round_debug_info:
            round_debug_info.append(
                {"final_best_cov_result": best_cov_result.model_dump(), "error": str(e)}
            )
            round_inputs_file = local_id_dir / f"react_eval_round_inputs_{item.repeat_index}.json"
            with round_inputs_file.open("w", encoding="utf-8") as f:
                json.dump(round_debug_info, f, indent=2, ensure_ascii=True)
        return {
            "context_id": context.id,
            "dataset": context.dataset_name,
            "repeat_index": item.repeat_index,
            "tb_file": str(latest_tb_file) if latest_tb_file else None,
            "llm_stats": _accumulate_llm_stats(stats_history) if stats_history else None,
            "eda_result": latest_eda_result,
            "eval_result": best_cov_result,
            "round_debug_info": round_debug_info,
            "local_id_dir": str(local_id_dir),
            "task_hash_id": task_hash_id or "unknown",
        }

    if local_id_dir is None:
        local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
        local_id_dir.mkdir(parents=True, exist_ok=True)

    if save_round_inputs and round_debug_info:
        round_inputs_file = local_id_dir / f"react_eval_round_inputs_{item.repeat_index}.json"
        round_debug_info.append({"final_best_cov_result": best_cov_result.model_dump()})
        with round_inputs_file.open("w", encoding="utf-8") as f:
            json.dump(round_debug_info, f, indent=2, ensure_ascii=True)

    return {
        "context_id": context.id,
        "dataset": context.dataset_name,
        "repeat_index": item.repeat_index,
        "tb_file": str(latest_tb_file) if latest_tb_file else None,
        "llm_stats": _accumulate_llm_stats(stats_history) if stats_history else None,
        "eda_result": latest_eda_result,
        "eval_result": best_cov_result,
        "round_debug_info": round_debug_info,
        "local_id_dir": str(local_id_dir),
        "task_hash_id": task_hash_id or "unknown",
    }


def _metric_kind(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (float, int)):
        return "float"
    return None


def _score_group(
    grouped: dict[str, list[CovResult]],
    *,
    pass_k: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    metrics_summary: dict[str, dict[str, float]] = {}
    metrics_per_task: dict[str, dict[str, dict[str, float]]] = {}

    if not grouped:
        return metrics_summary, metrics_per_task

    sample_metrics = grouped[next(iter(grouped))][0].model_dump()
    for metric_name in sample_metrics:
        values_by_task: dict[str, list[Any]] = {}
        for task_id, samples in grouped.items():
            values = [getattr(sample, metric_name) for sample in samples]
            values_by_task[task_id] = values

        metric_kind = _metric_kind(values_by_task[next(iter(values_by_task))][0])
        if metric_kind is None:
            continue
        per_task: dict[str, dict[str, float]] = {}
        for task_id, values in values_by_task.items():
            if not values:
                per_task[task_id] = {"@1": 0.0, f"@{pass_k}": 0.0}
                continue
            if metric_kind == "bool":
                pass_at_1 = sum(1 for v in values if v) / len(values)
                pass_at_k = 1.0 if any(values) else 0.0
                per_task[task_id] = {"@1": pass_at_1, f"@{pass_k}": pass_at_k}
            else:
                best_at_1 = sum(float(v) for v in values) / len(values)
                best_at_k = max(float(v) for v in values)
                per_task[task_id] = {"@1": best_at_1, f"@{pass_k}": best_at_k}

        metrics_per_task[metric_name] = per_task
        overall_at_1 = sum(v["@1"] for v in per_task.values()) / len(per_task)
        overall_at_k = sum(v[f"@{pass_k}"] for v in per_task.values()) / len(per_task)
        metrics_summary[metric_name] = {"@1": overall_at_1, f"@{pass_k}": overall_at_k}

    return metrics_summary, metrics_per_task


async def main() -> None:
    logging.info(f"Writing to {LOCAL_TMP_DIR}...")
    parser = argparse.ArgumentParser(description="Batch eval with pass@k/best@k summaries.")
    parser.add_argument("--model", type=str, required=True, help="Model name (required).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hez2024/cvdp_ecov_eval",
        help="Dataset name to load via HuggingFace datasets.",
    )
    parser.add_argument("--split", type=str, default="eval", help="Dataset split name.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0=all).")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end-idx", type=int, default=0, help="End index (exclusive).")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of repeated evals per task (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_TMP_DIR / "eval_debug.json",
        help="Output JSON path for debug info.",
    )

    parser.add_argument("--base-url", type=str, default="http://localhost", help="Base URL.")
    parser.add_argument("--port", type=int, default=8000, help="API port.")
    parser.add_argument("--api-key", type=str, default="dummy", help="API key.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_LLM_MAX_COMPLETION_TOKENS,
        help="Max completion tokens.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="LLM request timeout in seconds.",
    )
    parser.add_argument(
        "--use-responses-api",
        action="store_true",
        help="Use Responses API instead of Completion API.",
    )
    parser.add_argument(
        "--query-debug",
        action="store_true",
        help="Enable debug printing for LLM queries.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Tokenizer path (for reasoning token stats).",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=DEFAULT_LLM_RETRIES,
        help="Retry count for LLM query.",
    )

    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="Remote server.")
    parser.add_argument(
        "--remote-repo-dir",
        type=str,
        default=DEFAULT_REMOTE_REPO_DIR,
        help="Remote repo dir for EDA job.",
    )
    parser.add_argument(
        "--eda-single-stage-timeout-s",
        type=float,
        default=DEFAULT_EDA_SINGLE_STAGE_TIMEOUT_S,
        help="EDA timeout per stage.",
    )
    parser.add_argument(
        "--eda-stages",
        type=int,
        default=DEFAULT_EDA_STAGES,
        help="Number of EDA stages (for total timeout).",
    )
    parser.add_argument(
        "--eda-job-timeout-s",
        type=float,
        default=0.0,
        help="Total EDA timeout; 0 means compute from stages.",
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
        "--prompt-injection",
        action="store_true",
        help="Enable prompt injection for testbench generation.",
    )
    parser.add_argument("--debug", action="store_true", help="Log extra info.")
    parser.add_argument(
        "--react-mode",
        type=str,
        default="none",
        choices=["none", "react", "markov-react", "markov-react-only"],
        help="React mode for follow-up rounds.",
    )
    parser.add_argument(
        "--vanilla-run-dir",
        type=Path,
        default=None,
        help="Vanilla run dir (required for markov-react-only).",
    )
    parser.add_argument(
        "--react-rounds",
        type=int,
        default=DEFAULT_REACT_ROUNDS,
        help="Number of react follow-up rounds (default: 3).",
    )
    # A whole group of react client args
    parser.add_argument(
        "--use-separate-react-client", action="store_true", help="Use separate client."
    )
    parser.add_argument(
        "--react-base-url", type=str, default="http://localhost", help="React Base URL."
    )
    parser.add_argument("--react-port", type=int, default=8000, help="React API port.")
    parser.add_argument("--react-api-key", type=str, default="dummy", help="React API key.")
    parser.add_argument("--react-model", type=str, default=None, help="React model.")
    parser.add_argument(
        "--react-max-completion-tokens",
        type=int,
        default=DEFAULT_LLM_MAX_COMPLETION_TOKENS,
        help="React max completion tokens.",
    )
    parser.add_argument(
        "--react-tokenizer-dir", type=str, default=None, help="React tokenizer directory."
    )
    parser.add_argument("--react-temperature", type=float, default=None, help="React temperature.")
    parser.add_argument("--react-top-p", type=float, default=None, help="React top-p.")
    parser.add_argument(
        "--react-timeout-seconds", type=float, default=None, help="React timeout seconds."
    )
    parser.add_argument(
        "--react-use-responses-api", action="store_true", help="Use Responses API for React."
    )
    #
    parser.add_argument("--use-vertex", action="store_true", help="Use Vertex AI client.")
    parser.add_argument(
        "--vertex-project-id", type=str, default="ucsd-cse-stable-gcp", help="Vertex project ID."
    )
    parser.add_argument("--vertex-location", type=str, default="global", help="Vertex location.")

    args = parser.parse_args()

    contexts = _load_dataset(
        args.dataset,
        args.split,
        limit=args.limit,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prompt_injection=args.prompt_injection,
    )
    if args.react_mode == "markov-react-only":
        assert args.vanilla_run_dir, "markov-react-only requires --vanilla-run-dir"
        vanilla_run_dir = args.vanilla_run_dir
        assert vanilla_run_dir.is_dir(), f"Missing vanilla run dir: {vanilla_run_dir}"
        run_map = _discover_vanilla_runs(vanilla_run_dir)
        items: list[EvalItem] = []
        for ctx in contexts:
            key = (ctx.dataset_name, ctx.id)
            run_dirs = run_map.get(key, [])
            assert len(run_dirs) <= args.n_samples, (
                f"Too many vanilla runs for {ctx.dataset_name}:{ctx.id}"
            )
            run_dirs = sorted(run_dirs, key=lambda p: p.name)
            for repeat_index in range(min(len(run_dirs), args.n_samples)):
                local_work_dir = run_dirs[repeat_index]
                context_dir = local_work_dir.parent
                dataset_rel = context_dir.relative_to(vanilla_run_dir)
                dataset_name = "/".join(dataset_rel.parts[:-1])
                assert dataset_name == ctx.dataset_name, f"Dataset mismatch for {local_work_dir}"
                assert context_dir.name == ctx.id, f"Context mismatch for {local_work_dir}"
                items.append(
                    ResumeEvalItem(
                        context=ctx,
                        repeat_index=repeat_index,
                        vanilla_local_work_dir=local_work_dir,
                    )
                )
        exp_ctx_id_cnt = len(contexts)
        got_ctx_id_cnt = len(run_map.keys())
        exp_run_id_cnt = exp_ctx_id_cnt * args.n_samples
        got_run_id_cnt = len(items)
        print(
            f"Discovered {got_ctx_id_cnt}/{exp_ctx_id_cnt} contexts with vanilla runs, "
            f"total {got_run_id_cnt}/{exp_run_id_cnt} runs for markov-react-only."
        )
    else:
        items = [
            EvalItem(context=ctx, repeat_index=repeat_index)
            for ctx in contexts
            for repeat_index in range(args.n_samples)
        ]

    if not items:
        print("No items to evaluate.")
        return

    client = _build_client(args.base_url, args.port, args.api_key)
    query_args_fields: dict[str, Any] = {
        "client": client,
        "model": args.model,
        "max_completion_tokens": args.max_completion_tokens,
    }
    if args.temperature is not None:
        query_args_fields["temperature"] = args.temperature
    if args.top_p is not None:
        query_args_fields["top_p"] = args.top_p
    if args.timeout_seconds is not None:
        query_args_fields["timeout_seconds"] = args.timeout_seconds
    if args.tokenizer_dir is not None:
        query_args_fields["tokenizer_dir"] = args.tokenizer_dir
    if args.use_responses_api:
        query_args_fields["use_responses_api"] = True
    if args.use_vertex:
        # check GOOGLE_APPLICATION_CREDENTIALS is set
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            raise OSError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        vertex_client_refresher = AsyncOpenAICredentialsRefresher(
            project_id=args.vertex_project_id, location=args.vertex_location
        )
        query_args_fields["use_vertex"] = True
        query_args_fields["vertex_client"] = vertex_client_refresher
    query_args = OpenAIQueryArgs(**query_args_fields)
    # log query args
    query_args_log = query_args.model_dump()
    query_args_log.pop("client", None)
    print(f"Using query args: {query_args_log}")

    if args.use_separate_react_client:
        assert args.react_model is not None, (
            "React model must be specified if using separate client"
        )
        react_client = _build_client(args.react_base_url, args.react_port, args.react_api_key)
        react_query_args_fields: dict[str, Any] = {
            "client": react_client,
            "model": args.react_model,
            "max_completion_tokens": args.react_max_completion_tokens,
        }
        if args.react_temperature is not None:
            react_query_args_fields["temperature"] = args.react_temperature
        if args.react_top_p is not None:
            react_query_args_fields["top_p"] = args.react_top_p
        if args.react_timeout_seconds is not None:
            react_query_args_fields["timeout_seconds"] = args.react_timeout_seconds
        if args.react_tokenizer_dir is not None:
            react_query_args_fields["tokenizer_dir"] = args.react_tokenizer_dir
        if args.react_use_responses_api:
            react_query_args_fields["use_responses_api"] = True
        react_query_args = OpenAIQueryArgs(**react_query_args_fields)
        react_query_args_log = react_query_args.model_dump()
        react_query_args_log.pop("client", None)
        print(f"Using separate React query args: {react_query_args_log}")

    llm_request_timeout_s = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else DEFAULT_LLM_SINGLE_QUERY_TIMEOUT_S
    )
    llm_job_timeout_s = llm_request_timeout_s * (args.llm_retries + 1) + 30
    eda_job_timeout_s = (
        args.eda_job_timeout_s
        if args.eda_job_timeout_s > 0
        else args.eda_single_stage_timeout_s * args.eda_stages + 30
    )

    results, stats = await run_pipeline_queue_workers(
        items,
        workflow=eval_workflow,
        orchestrator_workers=args.orchestrator_workers,
        llm_concurrency=args.llm_concurrency,
        eda_concurrency=args.eda_concurrency,
        llm_timeout_s=llm_job_timeout_s,
        eda_timeout_s=eda_job_timeout_s,
        shared={
            "query_args": query_args,
            "use_separate_react_client": args.use_separate_react_client,
            "react_query_args": react_query_args if args.use_separate_react_client else query_args,
            "llm_retries": args.llm_retries,
            "llm_job_timeout_s": llm_job_timeout_s,
            "eda_job_timeout_s": eda_job_timeout_s,
            "eda_single_stage_timeout_s": args.eda_single_stage_timeout_s,
            "server": args.server,
            "remote_repo_dir": args.remote_repo_dir,
            "query_debug": args.query_debug,
            "react_mode": args.react_mode,
            "react_rounds": args.react_rounds,
            "save_round_inputs": args.debug,
        },
        include_traceback=args.debug,
        log_stage_timing=args.debug,
    )

    grouped: dict[str, list[CovResult]] = {}
    samples_output: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for item, result in zip(items, results, strict=True):
        context = item.context
        if result.ok and result.value is not None:
            eval_result = result.value["eval_result"]
            llm_stats = result.value.get("llm_stats")
        else:
            eval_result = CovResult(id=context.id)
            llm_stats = None

        if isinstance(llm_stats, dict):
            total_prompt_tokens += int(llm_stats.get("prompt_tokens", 0))
            total_completion_tokens += int(llm_stats.get("completion_tokens", 0))

        grouped.setdefault(context.id, []).append(eval_result)
        samples_output.append(
            {
                "context_id": context.id,
                "dataset": context.dataset_name,
                "repeat_index": item.repeat_index,
                "ok": result.ok,
                "error": result.error,
                "eval_result": eval_result.model_dump(),
                "llm_stats": llm_stats,
            }
        )

    pad_missing_runs = args.react_mode == "markov-react-only"
    if pad_missing_runs:
        existing_keys = {
            (sample["context_id"], sample["repeat_index"]) for sample in samples_output
        }
        for ctx in contexts:
            grouped.setdefault(ctx.id, [])
            for repeat_index in range(args.n_samples):
                repeat_key = (ctx.id, repeat_index)
                if repeat_key in existing_keys:
                    continue
                zero_result = CovResult(id=ctx.id)
                grouped[ctx.id].append(zero_result)
                samples_output.append(
                    {
                        "context_id": ctx.id,
                        "dataset": ctx.dataset_name,
                        "repeat_index": repeat_index,
                        "ok": False,
                        "error": "missing_vanilla_run",
                        "eval_result": zero_result.model_dump(),
                        "llm_stats": None,
                    }
                )

    pass_k = min(args.n_samples, 5)
    metrics_summary, metrics_per_task = _score_group(grouped, pass_k=pass_k)
    pass_targets_per_task = metrics_per_task.get("is_pass_targets", {})
    non_agentic_task_ids = [tid for tid in grouped if tid.startswith("cvdp_copilot_")]
    agentic_task_ids = [tid for tid in grouped if tid.startswith("cvdp_agentic_")]
    non_agentic_scores = [
        pass_targets_per_task[tid] for tid in non_agentic_task_ids if tid in pass_targets_per_task
    ]
    agentic_scores = [
        pass_targets_per_task[tid] for tid in agentic_task_ids if tid in pass_targets_per_task
    ]

    def _avg_scores(values: list[dict[str, float]]) -> dict[str, float]:
        if not values:
            return {"@1": 0.0, f"@{pass_k}": 0.0}
        return {
            "@1": sum(v["@1"] for v in values) / len(values),
            f"@{pass_k}": sum(v[f"@{pass_k}"] for v in values) / len(values),
        }

    pass_targets_non_agentic = _avg_scores(non_agentic_scores)
    pass_targets_agentic = _avg_scores(agentic_scores)
    round_cov: dict[int, list[float]] = {}
    if pad_missing_runs:
        eval_by_key = {
            (sample["context_id"], sample["repeat_index"]): sample["eval_result"]
            for sample in samples_output
        }
        for repeat_index in range(args.n_samples):
            values: list[float] = []
            for ctx in contexts:
                eval_dict = eval_by_key.get((ctx.id, repeat_index))
                if not eval_dict:
                    values.append(0.0)
                    continue
                eval_result = CovResult(**eval_dict)
                values.append(eval_result.overall_coverage if eval_result.is_pass_xrun else 0.0)
            round_cov[repeat_index] = values
        per_round_avg = [
            sum(values) / len(values) if values else 0.0 for _, values in sorted(round_cov.items())
        ]
    else:
        for item, result in zip(items, results, strict=True):
            eval_result = (
                result.value["eval_result"]
                if result.ok and result.value is not None
                else CovResult(id=item.context.id)
            )
            if eval_result.is_pass_xrun:
                round_cov.setdefault(item.repeat_index, []).append(eval_result.overall_coverage)
        per_round_avg = [
            sum(values) / len(values) if values else 0.0 for _, values in sorted(round_cov.items())
        ]
    mean_sim_pass_cov_at_1 = sum(per_round_avg) / len(per_round_avg) if per_round_avg else 0.0
    mean_sim_pass_cov_at_k = max(per_round_avg) if per_round_avg else 0.0
    metrics_summary["mean_sim_pass_cov"] = {
        "@1": mean_sim_pass_cov_at_1,
        f"@{pass_k}": mean_sim_pass_cov_at_k,
    }

    print("\n=== Eval Summary ===")
    for metric_name, metric_scores in metrics_summary.items():
        if metric_name == "mean_sim_pass_cov":
            print(
                f"{metric_name}: Best@1= {metric_scores['@1'] * 100:.1f} % "
                f"Best@{pass_k}= {metric_scores[f'@{pass_k}'] * 100:.1f} %"
            )
            continue
        metric_kind = _metric_kind(getattr(next(iter(grouped.values()))[0], metric_name))
        if metric_kind == "bool":
            print(
                f"{metric_name}: Pass@1= {metric_scores['@1'] * 100:.1f} % "
                f"Pass@{pass_k}= {metric_scores[f'@{pass_k}'] * 100:.1f} %"
            )
        else:
            print(
                f"{metric_name}: Best@1= {metric_scores['@1'] * 100:.1f} % "
                f"Best@{pass_k}= {metric_scores[f'@{pass_k}'] * 100:.1f} %"
            )

    if non_agentic_task_ids:
        print(
            "pass_targets_non_agentic: "
            f"Pass@1= {pass_targets_non_agentic['@1'] * 100:.1f} % "
            f"Pass@{pass_k}= {pass_targets_non_agentic[f'@{pass_k}'] * 100:.1f} %"
        )
    if agentic_task_ids:
        print(
            "pass_targets_agentic: "
            f"Pass@1= {pass_targets_agentic['@1'] * 100:.1f} % "
            f"Pass@{pass_k}= {pass_targets_agentic[f'@{pass_k}'] * 100:.1f} %"
        )

    print(
        f"Token stats: prompt_tokens= {total_prompt_tokens} , "
        f"completion_tokens= {total_completion_tokens} "
    )
    print(
        f"\nFinished {stats.finished}/{stats.total}, "
        f"failed={stats.failed}, "
        f"elapsed={stats.elapsed():.1f}s"
    )

    output_payload = {
        "summary": {
            "pass_k": pass_k,
            "metrics": metrics_summary,
            "pass_targets_non_agentic": pass_targets_non_agentic,
            "pass_targets_agentic": pass_targets_agentic,
            "token_stats": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            },
            "stats": {
                "finished": stats.finished,
                "total": stats.total,
                "failed": stats.failed,
                "elapsed_s": round(stats.elapsed(), 1),
            },
        },
        "per_task": metrics_per_task,
        "samples": samples_output,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=True)
    print(f"Debug output written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
