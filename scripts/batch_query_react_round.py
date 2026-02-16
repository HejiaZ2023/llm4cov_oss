import argparse
import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from llm4cov.datasets.eval import display_eval_stats, eval_cov_result_against_expectations
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import (
    CovResult,
    DataContext,
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
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
# MODEL = "qwen3_4b_full_8g_3b_1e"
# TOKENIZER_DIR = "/mnt/raid0_ssd/hejia/sft_backup/qwen3_4b_full_8g_3b_1e/checkpoint-1268"
TOKENIZER_DIR = None
REMOTE_REPO_DIR = "/workspace/llm4cov_eda"

BASE_URL = "http://localhost"
# BASE_URL = "http://69.63.236.190"
PORT = 11451
# PORT = 26558
API_KEY = "dummy"

LLM_SINGLE_QUERY_TIMEOUT_S = 1440 if MODEL_SIZE == "L" else 600
LLM_RETRIES = 3
# LLM_JOB_TIMEOUT_S = LLM_SINGLE_QUERY_TIMEOUT_S * (LLM_RETRIES + 1) + 30
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
LLM_MAX_COMPLETION_TOKENS = 16384


GEN_REPETITIONS = 2
DEFAULT_REACT_ROUNDS = 3
DEFAULT_REACT_MODE = "markov-react"

EDA_SINGLE_STAGE_TIMEOUT_S = 30
EDA_STAGES = 3
EDA_JOB_TIMEOUT_S = EDA_SINGLE_STAGE_TIMEOUT_S * EDA_STAGES + 30

DEFAULT_ORCHESTRATOR_WORKERS = 96
DEFAULT_LLM_CONCURRENCY = 96
DEFAULT_EDA_CONCURRENCY = 16

DEBUG = False
LOG_DEBUG = False
QUERY_DEBUG = False

SYN_PROMPT_INJECTION = False

logging.basicConfig(
    level=logging.DEBUG if LOG_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
if not LOG_DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)


@dataclass
class ReactRoundItem:
    context: LlmGenTbContext
    prev_tb: DataFile | None
    prev_cov: CovResult | None
    prev_run_id: str | None
    prev_run_dir: str | None
    prev_result_json_raw: str | None
    separation_id: int
    resume_history: dict[int, tuple[Path, dict[str, Any]]]


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
HISTORY_FILE = "react_round_history.json"


def _parse_eval_key(key: str, run_dir: Path) -> Path | None:
    key_path = Path(key)
    if key_path.is_absolute():
        try:
            return key_path.relative_to(run_dir)
        except ValueError:
            run_name = run_dir.name
            if run_name in key_path.parts:
                idx = key_path.parts.index(run_name)
                return Path(*key_path.parts[idx + 1 :])
            return None
    return key_path


def _load_eval_stats(run_dir: Path) -> dict[str, Any]:
    eval_stats_path = run_dir / "eval_stats.json"
    if not eval_stats_path.is_file():
        raise FileNotFoundError(f"Missing eval_stats.json in {run_dir}")
    with eval_stats_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("eval_stats.json is not a dict")
    return data


def _best_entries_from_eval_stats(
    eval_stats: dict[str, Any], run_dir: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    best_entries: dict[tuple[str, str], dict[str, Any]] = {}
    raw_entries = eval_stats.get("best_results_output")
    if isinstance(raw_entries, dict):
        for key, entry in raw_entries.items():
            rel_path = _parse_eval_key(key, run_dir)
            if rel_path is None or len(rel_path.parts) < 2:
                logging.warning("Skipping invalid eval_stats key: %s", key)
                continue
            dataset_name = "/".join(rel_path.parts[:-1])
            context_id = rel_path.parts[-1]
            entry["run_id"] = entry.get("run_id")
            best_entries[(dataset_name, context_id)] = entry
        return best_entries

    raw_entries = eval_stats.get("best_rids")
    if isinstance(raw_entries, dict):
        for key, run_id in raw_entries.items():
            rel_path = _parse_eval_key(key, run_dir)
            if rel_path is None or len(rel_path.parts) < 2:
                logging.warning("Skipping invalid eval_stats key: %s", key)
                continue
            dataset_name = "/".join(rel_path.parts[:-1])
            context_id = rel_path.parts[-1]
            best_entries[(dataset_name, context_id)] = {"run_id": run_id}
        return best_entries

    raise ValueError("eval_stats.json missing best_results_output or best_rids")


def _find_prev_tb(task_dir: Path, context: LlmGenTbContext) -> DataFile | None:
    if not task_dir.is_dir():
        return None
    rtl_names = {f.name for f in context.rtl_files}
    candidates = [p for p in task_dir.iterdir() if p.is_file() and p.name not in rtl_names]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.suffix != ".sv", p.name))
    tb_path = candidates[0]
    with tb_path.open("r", encoding="utf-8") as f:
        content = f.read()
    return DataFile(name=tb_path.name, content=content)


def _find_prev_result_json(task_dir: Path) -> str | None:
    if not task_dir.is_dir():
        return None
    candidates = [p for p in task_dir.iterdir() if p.is_file() and p.name.endswith("_result.json")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    result_path = candidates[0]
    with result_path.open("r", encoding="utf-8") as f:
        return f.read()


def _cov_from_eval_entry(context_id: str, entry: dict[str, Any]) -> CovResult | None:
    if "is_pass_xrun" not in entry:
        return None
    return CovResult(
        id=context_id,
        is_pass_xrun=bool(entry.get("is_pass_xrun", False)),
        has_coverage=bool(entry.get("has_coverage", False)),
        overall_coverage=float(entry.get("overall_coverage", 0.0)),
        is_pass_targets=bool(entry.get("is_pass_targets", False)),
        misc=entry.get("misc", {}),
    )


def _summarize_prev_cov(prev_cov: CovResult | None) -> dict[str, str] | None:
    if prev_cov is None:
        return None
    if not prev_cov.is_pass_xrun:
        return {"status": "xrun_failed", "stage": "xrun", "log": "xrun failed"}
    if not prev_cov.has_coverage:
        return {"status": "imc_failed", "stage": "imc", "log": "imc failed"}
    return {
        "status": "success",
        "stage": "success",
        "coverage": f"overall={prev_cov.overall_coverage:.4f}",
    }


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


def _summarize_prev_result_json(
    prev_result_json_raw: str | None, prev_cov: CovResult | None
) -> dict[str, str] | None:
    if not prev_result_json_raw:
        return _summarize_prev_cov(prev_cov)
    try:
        parsed = json.loads(prev_result_json_raw)
    except json.JSONDecodeError:
        return _summarize_prev_cov(prev_cov)
    if not isinstance(parsed, dict):
        return _summarize_prev_cov(prev_cov)
    return _summarize_eda_result(parsed)


def _parse_tb_from_response(response_text: str) -> DataFile | None:
    if not response_text:
        return None
    match_name = re.search(r"filename:\s*([^\n\r]+)", response_text)
    match_block = re.search(
        r"```systemverilog\s*(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE
    )
    if not match_name or not match_block:
        return None
    return DataFile(name=match_name.group(1).strip(), content=match_block.group(1).strip())


def _round_history_dir(
    round_index: int,
    *,
    local_work_dir: Path | None,
    prev_run_dir: str | None,
    local_id_dir: Path,
    run_id: str | None,
) -> Path:
    if local_work_dir is not None:
        base_dir = local_work_dir
    elif prev_run_dir:
        base_dir = Path(prev_run_dir)
    else:
        base_dir = local_id_dir / (run_id or "unknown_run")
    history_dir = base_dir / f"round_{round_index}"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def _parse_history_context(history_path: Path, run_dir: Path) -> tuple[str, str] | None:
    try:
        rel = history_path.relative_to(run_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    if len(parts) >= 5 and parts[-2].startswith("round_"):
        dataset_name = "/".join(parts[:-4])
        context_id = parts[-4]
    elif len(parts) >= 5:
        dataset_name = "/".join(parts[:-3])
        context_id = parts[-3]
    else:
        dataset_name = "/".join(parts[:-2])
        context_id = parts[-2]
    return dataset_name, context_id


def _load_history_entry(history_path: Path) -> dict[str, Any] | None:
    try:
        with history_path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(history, list) or not history:
        return None
    entry = history[-1]
    if not isinstance(entry, dict):
        return None
    return entry


def _cov_categories_from_summary(summary_entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary_entry, dict):
        return None
    return {k: v for k, v in summary_entry.items() if k not in ("name", "level")}


def _extract_cov_categories_from_result(
    result: dict[str, Any] | None, context: LlmGenTbContext
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    cov_info = result.get("cov_info")
    if not isinstance(cov_info, dict):
        return None
    summary = cov_info.get("summary")
    if not isinstance(summary, list):
        return None
    for entry in summary:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == context.dut_top_module_name and entry.get("level") == 0:
            return _cov_categories_from_summary(entry)
    return None


def _extract_cov_categories_from_raw(
    result_json_raw: str | None, context: LlmGenTbContext
) -> dict[str, Any] | None:
    if not result_json_raw:
        return None
    try:
        parsed = json.loads(result_json_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _extract_cov_categories_from_result(parsed, context)


async def react_round_workflow(ctx: RunContext, item: ReactRoundItem) -> dict[str, Any]:
    context = item.context
    react_rounds: int = ctx.shared["react_rounds"]
    react_mode: str = ctx.shared["react_mode"]
    resume_history = item.resume_history
    messages = build_initial_prompt_from_context(context)
    prev_tb: DataFile | None = item.prev_tb
    prev_cov: CovResult | None = item.prev_cov
    prev_result_json_raw: str | None = item.prev_result_json_raw
    local_id_dir: Path | None = None
    task_hash_id: str | None = None
    best_cov_result = prev_cov or CovResult(id=context.id)
    markov_snapshot: list[dict[str, str]] = []
    last_run_id = item.prev_run_id
    last_run_dir = item.prev_run_dir
    total_rounds = max(react_rounds, 1)
    history_file: Path | None = None

    parse_status = {"status": "failed", "type": "file"}
    if prev_tb is not None:
        parse_status = {"status": "success", "type": "file"}
        response_text = f"filename: {prev_tb.name}\n```systemverilog\n{prev_tb.content}\n```"
        messages.append({"role": "assistant", "content": response_text})

    query_args = OpenAIQueryArgs(
        client=ctx.shared["client"],
        model=ctx.shared["model"],
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
        timeout_seconds=ctx.shared["llm_single_query_timeout_s"],
        tokenizer_dir=ctx.shared["tokenizer_dir"],
    )

    prev_eda_status = _summarize_prev_result_json(prev_result_json_raw, prev_cov)
    instruction = "Improve coverage toward 100% by editing the testbench."
    if prev_eda_status:
        if prev_eda_status.get("status") == "xrun_failed":
            instruction = "Fix the xrun failure by editing the testbench."
        elif prev_eda_status.get("status") != "success":
            instruction = "Fix the coverage run failure (imc stage) by editing the testbench."

    cov_result: CovResult | None = None
    cov_result_categories: dict[str, Any] | None = None
    prev_cov_categories = _extract_cov_categories_from_raw(prev_result_json_raw, context)
    improved = False

    if (
        prev_cov is not None
        and prev_cov.is_pass_xrun
        and prev_cov.has_coverage
        and prev_cov.overall_coverage >= 1.0
        and prev_tb is not None
    ):
        cov_result_categories = (
            prev_cov.misc if prev_cov_categories is None else prev_cov_categories
        )
        if item.prev_run_dir:
            prev_id_dir = Path(item.prev_run_dir).parent
            local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
            local_id_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(prev_id_dir, local_id_dir, dirs_exist_ok=True)
        response_text = f"filename: {prev_tb.name}\n```systemverilog\n{prev_tb.content}\n```"
        history = [
            {
                "round_index": 0,
                "separation_id": item.separation_id,
                "messages": messages,
                "llm_response": response_text,
                "llm_stats": None,
                "parse_status": {"status": "success", "type": "file"},
                "eda_result": None,
                "cov_result": prev_cov.model_dump(),
                "cov_result_categories": cov_result_categories,
                "improved": False,
                "prev_cov_result": prev_cov.model_dump(),
                "prev_cov_result_categories": prev_cov_categories or prev_cov.misc,
                "prev_tb": prev_tb.model_dump(),
                "prev_run_id": last_run_id,
                "prev_run_dir": last_run_dir,
                "prev_result_json_raw": prev_result_json_raw,
                "task_hash_id": last_run_id,
                "run_id": last_run_id,
            }
        ]
        if local_id_dir is None:
            local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
            local_id_dir.mkdir(parents=True, exist_ok=True)
        history_dir = _round_history_dir(
            0,
            local_work_dir=None,
            prev_run_dir=last_run_dir,
            local_id_dir=local_id_dir,
            run_id=last_run_id,
        )
        history_file = history_dir / HISTORY_FILE
        with history_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {
            "context_id": context.id,
            "dataset": context.dataset_name,
            "history_file": str(history_file),
            "local_id_dir": str(local_id_dir),
            "prev_cov": prev_cov,
            "cov_result": prev_cov,
            "cov_result_categories": cov_result_categories,
            "prev_cov_result_categories": prev_cov_categories or prev_cov.misc,
            "improved": False,
            "prev_run_id": last_run_id,
            "prev_run_dir": last_run_dir,
            "prev_result_json_raw": prev_result_json_raw,
            "task_hash_id": last_run_id,
        }

    followup_messages = build_react_followup_prompt(
        parse_status=parse_status,
        apply_status=None,
        eda_status=prev_eda_status,
        tb_content=None,  # Only 'patch' mode uses tb_content
        instruction=instruction + OUTPUT_FORMAT_REQ,
        is_single_message=True,
    )
    messages.extend(followup_messages)

    for round_idx in range(total_rounds):
        if round_idx in resume_history:
            history_path, entry = resume_history[round_idx]
            history_file = history_path
            messages_snapshot = entry.get("messages")
            if isinstance(messages_snapshot, list):
                messages = list(messages_snapshot)
            response_text_raw = entry.get("llm_response")
            response_text = response_text_raw if isinstance(response_text_raw, str) else ""
            if response_text:
                tb_file = _parse_tb_from_response(response_text)
                assert isinstance(tb_file, DataFile), "Parsed tb_file is not DataFile"
                prev_tb = tb_file
                messages.append({"role": "assistant", "content": response_text})
            resume_eda_result = (
                entry.get("eda_result") if isinstance(entry.get("eda_result"), dict) else None
            )
            cov_result_raw = entry.get("cov_result")
            cov_result = CovResult(**cov_result_raw) if isinstance(cov_result_raw, dict) else None
            cov_result_categories = entry.get("cov_result_categories")
            improved = bool(entry.get("improved", False))
            last_run_id = (
                entry.get("run_id")
                or entry.get("task_hash_id")
                or entry.get("prev_run_id")
                or last_run_id
            )
            if last_run_id is not None and not isinstance(last_run_id, str):
                last_run_id = str(last_run_id)
            if local_id_dir is None:
                local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
                local_id_dir.mkdir(parents=True, exist_ok=True)
            run_id = last_run_id or f"unknown_run_{item.separation_id}_{round_idx}"
            source_run_dir = history_path.parent.parent
            target_run_dir = local_id_dir / run_id
            target_run_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_run_dir, target_run_dir, dirs_exist_ok=True)
            history_file = target_run_dir / f"round_{round_idx}" / HISTORY_FILE
            task_hash_id = run_id
            last_run_dir = str(target_run_dir)

            # each run_dir is expected to contain only 1 round history
            for round_dir in target_run_dir.iterdir():
                if not round_dir.is_dir():
                    continue
                if not round_dir.name.startswith("round_"):
                    logging.warning(
                        "Unexpected dir in run_dir %s: %s", target_run_dir, round_dir.name
                    )
                    continue
                try:
                    round_num = int(round_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    logging.warning(
                        "Unexpected round dir name in run_dir %s: %s",
                        target_run_dir,
                        round_dir.name,
                    )
                    continue
                if round_num != round_idx:
                    shutil.rmtree(round_dir)
                    if round_num != round_idx + 1:  # error dir will only be next round
                        logging.warning(
                            "Removed unexpected round dir in run_dir %s: %s",
                            target_run_dir,
                            round_dir.name,
                        )

            if resume_eda_result is not None:
                prev_result_json_raw = json.dumps(resume_eda_result, ensure_ascii=True)
                prev_cov_categories = _extract_cov_categories_from_raw(
                    prev_result_json_raw, context
                )
            prev_cov = cov_result

            updated_best = False
            if cov_result is not None and best_cov_result <= cov_result:
                best_cov_result = cov_result
                updated_best = True

            if response_text:
                next_eda_status = _summarize_eda_result(resume_eda_result or {})
                next_instruction = "Improve coverage toward 100% by editing the testbench."
                if next_eda_status.get("status") == "xrun_failed":
                    next_instruction = "Fix the xrun failure by editing the testbench."
                elif next_eda_status.get("status") != "success":
                    next_instruction = (
                        "Fix the coverage run failure (imc stage) by editing the testbench."
                    )
                parse_status_entry = entry.get("parse_status")
                if not isinstance(parse_status_entry, dict):
                    parse_status_entry = {"status": "failed", "type": "file"}
                next_followup = build_react_followup_prompt(
                    parse_status=parse_status_entry,
                    apply_status=None,
                    eda_status=next_eda_status,
                    tb_content=None,
                    instruction=next_instruction + OUTPUT_FORMAT_REQ,
                    is_single_message=True,
                )
                messages.extend(next_followup)

            if react_mode == "markov-react" and updated_best:
                assert len(messages) >= 4, "Not enough messages for Markov snapshot"
                markov_snapshot = [messages[0], messages[1], messages[-2], messages[-1]]
            elif react_mode == "markov-react" and not markov_snapshot and len(messages) >= 4:
                markov_snapshot = [messages[0], messages[1], messages[-2], messages[-1]]

            if cov_result is not None and cov_result.overall_coverage >= 1.0:
                break
            continue

        if react_mode not in ["react", "markov-react"]:
            raise ValueError(f"Unknown react mode: {react_mode}")
        if react_mode == "markov-react" and round_idx > 0:
            messages_snapshot = list(markov_snapshot)
        else:
            messages_snapshot = list(messages)
            if react_mode == "markov-react" and round_idx == 0:
                # initialize
                markov_snapshot = list(messages)

        if react_mode == "markov-react":
            assert len(messages_snapshot) == 4, "Markov snapshot must have 4 messages"

        try:
            tb_file, stats = await ctx.run_llm(
                query_one_file,
                messages_snapshot,
                query_args,
                max_retries=ctx.shared["llm_max_retries"],
                debug=QUERY_DEBUG,
                timeout_s=ctx.shared["llm_job_timeout_s"],
                label=f"LLM:react_round_{round_idx}",
            )
            assert isinstance(stats, LLMQueryStats), "LLM stats is not LLMQueryStats"
            assert isinstance(tb_file, DataFile), "Returned tb_file is not DataFile"
        except Exception as exc:
            if isinstance(exc, asyncio.TimeoutError):
                exc = TimeoutError(
                    f"LLM query timed out for context {context.id} at round {round_idx}"
                )
            logging.warning(exc)
            if local_id_dir is None:
                local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
                local_id_dir.mkdir(parents=True, exist_ok=True)
            history = [
                {
                    "round_index": round_idx,
                    "separation_id": item.separation_id,
                    "messages": messages_snapshot,
                    "llm_response": "",
                    "llm_stats": None,
                    "parse_status": {"status": "failed", "type": "file"},
                    "eda_result": None,
                    "cov_result": None,
                    "cov_result_categories": None,
                    "improved": False,
                    "prev_cov_result": prev_cov.model_dump() if prev_cov else None,
                    "prev_cov_result_categories": prev_cov_categories,
                    "prev_tb": prev_tb.model_dump() if prev_tb else None,
                    "prev_run_id": last_run_id,
                    "prev_run_dir": last_run_dir,
                    "prev_result_json_raw": prev_result_json_raw,
                    "task_hash_id": last_run_id,
                    "run_id": last_run_id,
                    "error": str(exc),
                }
            ]
            history_dir = _round_history_dir(
                round_idx,
                local_work_dir=None,
                prev_run_dir=last_run_dir,
                local_id_dir=local_id_dir,
                run_id=last_run_id,
            )
            history_file = history_dir / HISTORY_FILE
            with history_file.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            return {
                "context_id": context.id,
                "dataset": context.dataset_name,
                "history_file": str(history_file),
                "local_id_dir": str(local_id_dir),
                "prev_cov": prev_cov,
                "cov_result": None,
                "cov_result_categories": None,
                "prev_cov_result_categories": prev_cov_categories,
                "improved": False,
                "prev_run_id": last_run_id,
                "prev_run_dir": last_run_dir,
                "task_hash_id": last_run_id,
            }

        response_text = f"filename: {tb_file.name}\n```systemverilog\n{tb_file.content}\n```"

        parse_status_new: dict[str, str] = {"status": "failed", "type": "file"}
        eda_result: dict[str, Any] | None = None
        cov_result = None
        local_work_dir: Path | None = None

        parse_status_new = {"status": "success", "type": "file"}
        eda_result = await ctx.run_eda(
            run_remote_cov_job_pipeline,
            server=ctx.shared["server"],
            eda_repo_dir=REMOTE_REPO_DIR,
            context=context,
            tb_file=tb_file,
            skip_detail=False,
            timeout=EDA_SINGLE_STAGE_TIMEOUT_S,
            timeout_s=EDA_JOB_TIMEOUT_S,
            label=f"EDA:cov_round_{round_idx}",
        )
        assert "local_work_dir" in eda_result, "Missing local_work_dir in EDA result"
        local_work_dir = eda_result["local_work_dir"]
        assert isinstance(local_work_dir, Path), "local_work_dir should be a Path"
        local_id_dir = local_work_dir.parent
        task_hash_id = local_work_dir.name
        cov_result = eval_cov_result_against_expectations(context, eda_result)
        eda_result["local_work_dir"] = str(local_work_dir)

        if local_id_dir is None:
            local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
            local_id_dir.mkdir(parents=True, exist_ok=True)

        improved = False
        if cov_result is not None and prev_cov is not None:
            improved = prev_cov < cov_result

        cov_result_categories_extract = _extract_cov_categories_from_result(eda_result, context)
        if cov_result_categories_extract is None and cov_result is not None:
            cov_result_categories = cov_result.misc
        else:
            cov_result_categories = cov_result_categories_extract
        if prev_cov_categories is None and prev_cov is not None:
            prev_cov_categories = prev_cov.misc

        history = [
            {
                "round_index": round_idx,
                "separation_id": item.separation_id,
                "messages": messages_snapshot,
                "llm_response": response_text,
                "llm_stats": stats.model_dump() if stats else None,
                "parse_status": parse_status_new,
                "eda_result": eda_result,
                "cov_result": cov_result.model_dump() if cov_result else None,
                "cov_result_categories": cov_result_categories,
                "improved": improved,
                "prev_cov_result": prev_cov.model_dump() if prev_cov else None,
                "prev_cov_result_categories": prev_cov_categories,
                "prev_tb": prev_tb.model_dump() if prev_tb else None,
                "prev_run_id": last_run_id,
                "prev_run_dir": last_run_dir,
                "prev_result_json_raw": prev_result_json_raw,
                "task_hash_id": task_hash_id or last_run_id,
                "run_id": task_hash_id or last_run_id,
            }
        ]

        history_dir = _round_history_dir(
            round_idx,
            local_work_dir=local_work_dir,
            prev_run_dir=last_run_dir,
            local_id_dir=local_id_dir,
            run_id=task_hash_id or last_run_id,
        )
        history_file = history_dir / HISTORY_FILE
        with history_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if stats.reasoning_content:
            reasoning_dir = history_dir
            reasoning_file = reasoning_dir / "llm_reasoning.txt"
            with reasoning_file.open("w", encoding="utf-8") as f:
                f.write(stats.reasoning_content or "")

        updated_best = False
        if cov_result is not None and best_cov_result <= cov_result:
            best_cov_result = cov_result
            updated_best = True

        last_run_id = task_hash_id or last_run_id
        last_run_dir = str(local_work_dir) if local_work_dir else last_run_dir

        messages.append({"role": "assistant", "content": response_text})
        next_eda_status = _summarize_eda_result(eda_result or {})
        next_instruction = "Improve coverage toward 100% by editing the testbench."
        if next_eda_status.get("status") == "xrun_failed":
            next_instruction = "Fix the xrun failure by editing the testbench."
        elif next_eda_status.get("status") != "success":
            next_instruction = "Fix the coverage run failure (imc stage) by editing the testbench."
        next_followup = build_react_followup_prompt(
            parse_status=parse_status_new,
            apply_status=None,
            eda_status=next_eda_status,
            tb_content=None,
            instruction=next_instruction + OUTPUT_FORMAT_REQ,
            is_single_message=True,
        )
        messages.extend(next_followup)

        prev_tb = tb_file
        prev_cov = cov_result
        if eda_result is not None:
            prev_result_json_raw = json.dumps(eda_result, ensure_ascii=True)
        prev_cov_categories = _extract_cov_categories_from_raw(prev_result_json_raw, context)

        if react_mode == "markov-react" and updated_best:
            assert len(messages) >= 4, "Not enough messages for Markov snapshot"
            markov_snapshot = [messages[0], messages[1], messages[-2], messages[-1]]

        if cov_result is not None and cov_result.overall_coverage >= 1.0:
            break

    if local_id_dir is None:
        local_id_dir = LOCAL_TMP_DIR / context.dataset_name / context.id
        local_id_dir.mkdir(parents=True, exist_ok=True)
    if history_file is None:
        history_file = local_id_dir / HISTORY_FILE

    return {
        "context_id": context.id,
        "dataset": context.dataset_name,
        "history_file": str(history_file),
        "local_id_dir": str(local_id_dir),
        "prev_cov": prev_cov,
        "cov_result": cov_result,
        "cov_result_categories": cov_result_categories,
        "prev_cov_result_categories": prev_cov_categories,
        "improved": improved if cov_result is not None else False,
        "prev_run_id": last_run_id,
        "prev_run_dir": last_run_dir,
        "task_hash_id": task_hash_id or last_run_id,
    }


def _dataset_split(dataset_name: str) -> str:
    if dataset_name == "hez2024/cvdp_ecov_eval":
        return "eval"
    return "train"


def _load_dataset_contexts(dataset_name: str, ids: set[str]) -> dict[str, LlmGenTbContext]:
    dataset = load_dataset_by_name(dataset_name, split=_dataset_split(dataset_name))
    contexts: dict[str, LlmGenTbContext] = {}
    for item in dataset:
        if isinstance(item, DataContext) and (item.id in ids and item.id not in contexts):
            ctx = data_context_to_llm_gen_tb_context(item)
            if SYN_PROMPT_INJECTION:
                ctx = inject_prompt_into_tb_generation(ctx)
            contexts[item.id] = ctx
    return contexts


async def main() -> None:
    global GEN_REPETITIONS
    logging.info(f"Writing to {LOCAL_TMP_DIR}...")
    parser = argparse.ArgumentParser(description="Run single react round from eval output.")
    parser.add_argument(
        "--eval-run-dir",
        type=Path,
        required=True,
        help="Path to eval run dir under eda_intermediate.",
    )
    parser.add_argument("--server", type=str, default=SERVER, help="Remote server.")
    parser.add_argument("--model", type=str, default=MODEL, help="Model name.")
    parser.add_argument("--tokenizer-dir", type=str, default=TOKENIZER_DIR, help="Tokenizer path.")
    parser.add_argument("--base-url", type=str, default=BASE_URL, help="Base URL.")
    parser.add_argument("--port", type=int, default=PORT, help="API port.")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="API key.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
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
        "--llm-single-query-timeout-s",
        type=float,
        default=LLM_SINGLE_QUERY_TIMEOUT_S,
        help="LLM single query timeout in seconds.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=LLM_RETRIES,
        help="Max retries for each LLM query.",
    )
    parser.add_argument(
        "--react-rounds",
        type=int,
        default=DEFAULT_REACT_ROUNDS,
        help="Number of react rounds per repetition (default: 3).",
    )
    parser.add_argument(
        "--react-mode",
        type=str,
        default=DEFAULT_REACT_MODE,
        choices=["react", "markov-react"],
        help="React mode for multi-round sampling.",
    )
    parser.add_argument(
        "--resume-from-breakpoint",
        action="store_true",
        help="Reuse existing react_round_history.json entries from a previous run.",
    )
    parser.add_argument(
        "--previous-run-dir",
        type=Path,
        default=None,
        help="Previous run dir containing react_round_history.json files.",
    )
    args = parser.parse_args()
    if (args.start is not None or args.end is not None) and args.limit:
        raise ValueError("--limit cannot be used together with --start/--end.")

    eval_run_dir: Path = args.eval_run_dir
    if not eval_run_dir.is_dir():
        raise FileNotFoundError(f"Eval run dir does not exist: {eval_run_dir}")

    eval_stats = _load_eval_stats(eval_run_dir)
    best_entries = _best_entries_from_eval_stats(eval_stats, eval_run_dir)

    dataset_ids: dict[str, set[str]] = {}
    for dataset_name, context_id in best_entries:
        dataset_ids.setdefault(dataset_name, set()).add(context_id)

    base_items: list[ReactRoundItem] = []
    for dataset_name, ids in dataset_ids.items():
        contexts = _load_dataset_contexts(dataset_name, ids)
        for context_id in ids:
            context = contexts.get(context_id)
            if context is None:
                logging.warning("Missing context for %s:%s", dataset_name, context_id)
                continue
            entry = best_entries.get((dataset_name, context_id), {})
            run_id = entry.get("run_id")
            prev_cov = _cov_from_eval_entry(context_id, entry) if isinstance(entry, dict) else None
            prev_tb = None
            prev_run_dir = None
            prev_result_json_raw = None
            if run_id:
                rel_path = Path(*dataset_name.split("/")) / context_id
                task_dir = eval_run_dir / rel_path / str(run_id)
                prev_tb = _find_prev_tb(task_dir, context)
                prev_result_json_raw = _find_prev_result_json(task_dir)
                prev_run_dir = str(task_dir)
            base_items.append(
                ReactRoundItem(
                    context=context,
                    prev_tb=prev_tb,
                    prev_cov=prev_cov,
                    prev_run_id=str(run_id) if run_id else None,
                    prev_run_dir=prev_run_dir,
                    prev_result_json_raw=prev_result_json_raw,
                    separation_id=0,
                    resume_history={},
                )
            )

    if args.start is not None or args.end is not None:
        for item in base_items:
            if item.prev_run_id is None:
                raise ValueError("prev_run_id is required when slicing with --start/--end.")
        base_items.sort(
            key=lambda item: (
                item.context.dataset_name,
                item.context.id,
                item.prev_run_id,
            )
        )
        base_items = base_items[slice(args.start, args.end)]

    if args.limit > 0:
        base_items = base_items[: args.limit]

    GEN_REPETITIONS = max(args.gen_repetitions, 1)
    items: list[ReactRoundItem] = []
    item_index: dict[tuple[str, str, int], ReactRoundItem] = {}
    for item in base_items:
        for rep in range(GEN_REPETITIONS):
            react_item = ReactRoundItem(
                context=item.context,
                prev_tb=item.prev_tb,
                prev_cov=item.prev_cov,
                prev_run_id=item.prev_run_id,
                prev_run_dir=item.prev_run_dir,
                prev_result_json_raw=item.prev_result_json_raw,
                separation_id=rep,
                resume_history={},
            )
            items.append(react_item)
            item_index[(item.context.dataset_name, item.context.id, rep)] = react_item

    if args.resume_from_breakpoint:
        if args.previous_run_dir is None:
            raise ValueError("--previous-run-dir is required with --resume-from-breakpoint")
        prev_dir = args.previous_run_dir
        if not prev_dir.is_dir():
            raise FileNotFoundError(f"Previous run dir does not exist: {prev_dir}")
        global_max_round = -1
        for history_path in prev_dir.rglob(HISTORY_FILE):
            context_info = _parse_history_context(history_path, prev_dir)
            if context_info is None:
                continue
            entry_raw = _load_history_entry(history_path)
            if entry_raw is None:
                continue
            entry = entry_raw
            assert isinstance(entry, dict)
            if "error" in entry:  # skip failed entries
                continue
            round_index = entry.get("round_index")
            separation_id = entry.get("separation_id", 0)
            if round_index is None:
                continue
            if not isinstance(round_index, int):
                try:
                    round_index = int(round_index)
                except (TypeError, ValueError):
                    continue
            if not isinstance(separation_id, int):
                try:
                    separation_id = int(separation_id)
                except (TypeError, ValueError):
                    separation_id = 0
            dataset_name, context_id = context_info
            item_key = (dataset_name, context_id, separation_id)
            target_item = item_index.get(item_key)
            if target_item is None:
                continue
            target_item.resume_history[round_index] = (history_path, entry)
            global_max_round = max(global_max_round, round_index)
        separation_rounds = [0 for _ in range(global_max_round + 1)]
        for (dataset_name, context_id, separation_id), target_item in item_index.items():
            if not target_item.resume_history:
                continue
            history_paths = set(hp for hp, _ in target_item.resume_history.values())
            run_paths = set(hp.parent.parent for hp in history_paths)
            if len(run_paths) != len(history_paths):
                logging.warning(
                    "Inconsistent resume history for %s:%s separation_id=%d: "
                    "multiple history files from the same run dir",
                    dataset_name,
                    context_id,
                    separation_id,
                )
                target_item.resume_history.clear()
                continue
            round_keys = sorted(target_item.resume_history.keys())
            max_round = round_keys[-1]
            expected = list(range(max_round + 1))
            if round_keys != expected:
                logging.warning(
                    "Non-contiguous resume history for "
                    f"{dataset_name}:{context_id} separation_id={separation_id}: "
                    f"got {round_keys}, expected {expected}"
                )
                target_item.resume_history.clear()
                continue
            separation_rounds[max_round] += 1
        resume_tasks: set[tuple[str, str]] = set()
        resume_repetitions: set[tuple[str, str, int]] = set()
        total_rounds = 0
        max_round = -1
        for key, target_item in item_index.items():
            if not target_item.resume_history:
                continue
            dataset_name, context_id, separation_id = key
            resume_tasks.add((dataset_name, context_id))
            resume_repetitions.add((dataset_name, context_id, separation_id))
            round_keys = list(target_item.resume_history.keys())
            total_rounds += len(round_keys)
            max_round = max(max_round, max(round_keys))
        logging.info(
            "Resume detected: tasks=%d repetitions=%d rounds=%d max_round=%d",
            len(resume_tasks),
            len(resume_repetitions),
            total_rounds,
            max_round,
        )
        for i in range(max_round + 1):
            logging.info("  Round %d: %d repetitions to resume", i, separation_rounds[i])
        items.sort(
            key=lambda item: max(item.resume_history.keys(), default=-1),
            reverse=True,
        )

    llm_job_timeout_s = args.llm_single_query_timeout_s * (args.llm_max_retries + 1) + 30

    client = AsyncOpenAI(
        base_url=f"{args.base_url}:{args.port}/v1",
        api_key=args.api_key,
    )

    results, stats = await run_pipeline_queue_workers(
        items,
        workflow=react_round_workflow,
        orchestrator_workers=args.orchestrator_workers,
        llm_concurrency=args.llm_concurrency,
        eda_concurrency=args.eda_concurrency,
        llm_timeout_s=llm_job_timeout_s,
        eda_timeout_s=EDA_JOB_TIMEOUT_S,
        shared={
            "client": client,
            "react_rounds": max(args.react_rounds, 1),
            "react_mode": args.react_mode,
            "server": args.server,
            "model": args.model,
            "tokenizer_dir": args.tokenizer_dir,
            "llm_single_query_timeout_s": args.llm_single_query_timeout_s,
            "llm_job_timeout_s": llm_job_timeout_s,
            "llm_max_retries": args.llm_max_retries,
        },
        include_traceback=False,
        log_stage_timing=DEBUG,
    )

    logging.info("\n=== Summary ===")
    cov_results: list[tuple[str, str, CovResult]] = []
    improved_rounds: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    history_index: list[dict[str, Any]] = []

    improved_xrun = 0
    improved_cov = 0
    for r in results:
        if not r.ok:
            logging.error(f"❌ FAILED: {r.error}")
            continue
        assert r.value is not None
        cov_result = r.value.get("cov_result")
        prev_cov = r.value.get("prev_cov")
        if isinstance(cov_result, CovResult):
            cov_results.append(
                (r.value["local_id_dir"], r.value.get("task_hash_id", "react_round"), cov_result)
            )
        history_index.append(
            {
                "dataset": r.value["dataset"],
                "context_id": r.value["context_id"],
                "history_file": r.value["history_file"],
                "improved": r.value.get("improved", False),
            }
        )
        comparisons.append(
            {
                "dataset": r.value["dataset"],
                "context_id": r.value["context_id"],
                "prev_cov_result": prev_cov.model_dump()
                if isinstance(prev_cov, CovResult)
                else None,
                "cov_result": cov_result.model_dump()
                if isinstance(cov_result, CovResult)
                else None,
                "prev_cov_result_categories": r.value.get("prev_cov_result_categories"),
                "cov_result_categories": r.value.get("cov_result_categories"),
                "improved": r.value.get("improved", False),
            }
        )
        if r.value.get("improved"):
            improved_rounds.append(
                {
                    "dataset": r.value["dataset"],
                    "context_id": r.value["context_id"],
                    "history_file": r.value["history_file"],
                    "cov_result": cov_result.model_dump()
                    if isinstance(cov_result, CovResult)
                    else None,
                }
            )
            assert isinstance(prev_cov, CovResult)
            if not prev_cov.is_pass_xrun:
                improved_xrun += 1
            else:
                improved_cov += 1

    logging.info(
        f"\nFinished {stats.finished}/{stats.total}, "
        f"failed={stats.failed}, "
        f"elapsed={stats.elapsed():.1f}s"
    )

    cov_stats = display_eval_stats(cov_results)
    total_items = cov_stats.get("total_items", 0)
    pass_xrun_items = round(cov_stats.get("pass_xrun", 0) * total_items)
    cov_stats["improved_rounds"] = len(improved_rounds)
    cov_stats["improve_canditates"] = total_items
    cov_stats["improved_xrun"] = improved_xrun
    cov_stats["improve_xrun_canditates"] = total_items - pass_xrun_items
    cov_stats["improved_cov"] = improved_cov
    cov_stats["improve_cov_canditates"] = pass_xrun_items
    cov_stats["improved_rounds_list"] = improved_rounds

    cov_stats["cov_comparisons"] = comparisons
    cov_stats["react_history_index"] = history_index
    cov_stats["source_eval_run_dir"] = str(eval_run_dir)

    states_file = LOCAL_TMP_DIR / "eval_stats.json"
    with states_file.open("w", encoding="utf-8") as f:
        json.dump(cov_stats, f, indent=4)
    logging.info(f"Eval stats written to {states_file}")


if __name__ == "__main__":
    asyncio.run(main())
