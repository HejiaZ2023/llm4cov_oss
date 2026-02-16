import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm4cov.datasets.eval import eval_cov_result_against_expectations
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import (
    DataContext,
    DataFile,
    LlmGenTbContext,
    data_context_to_llm_gen_tb_context,
)
from llm4cov.llm_query.prompt_build import build_initial_prompt_from_context
from llm4cov.syn_data_gen.prompt_inject import (
    PROMPT_TO_BE_INJECTED,
    inject_prompt_into_tb_generation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

ENABLE_COMPRESS_SYSTEM_PROMPT = False  # Just find out it's unnecessary


def _strip_prompt_injection(text: str) -> str:
    if PROMPT_TO_BE_INJECTED in text:
        text = text.replace(PROMPT_TO_BE_INJECTED, "")
    return text


def _strip_prompt_injection_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            content = _strip_prompt_injection(content)
        cleaned.append({**msg, "content": content})
    return cleaned


def _compress_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ENABLE_COMPRESS_SYSTEM_PROMPT:
        return messages
    compressed = []
    for msg in messages:
        if msg.get("role") == "system":
            compressed.append(
                {
                    **msg,
                    "content": (
                        "You are a SystemVerilog testbench generator. "
                        "Output: filename line + one systemverilog block. "
                        "No assertions/checks. No text inside the block besides code."
                    ),
                }
            )
        else:
            compressed.append(msg)
    return compressed


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


def _build_direct_infer_messages(context: LlmGenTbContext) -> list[dict[str, str]]:
    llm_context = inject_prompt_into_tb_generation(context)
    messages = build_initial_prompt_from_context(llm_context)
    messages = _strip_prompt_injection_messages(messages)
    return _compress_system_prompt(messages)


def _format_tb_output(tb_file: DataFile) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": f"filename: {tb_file.name}\n```systemverilog\n{tb_file.content}\n```",
    }


def _iter_agentic_histories(run_dir: Path) -> Iterable[tuple[str, str, str | None, Path]]:
    for history_path in run_dir.rglob("agentic_round_history.json"):
        try:
            rel = history_path.relative_to(run_dir)
        except ValueError:
            continue
        if len(rel.parts) < 4:
            continue
        if len(rel.parts) >= 5 and rel.parts[-2].startswith("round_"):
            dataset_name = "/".join(rel.parts[:-4])
            context_id = rel.parts[-4]
            run_id = rel.parts[-3]
        elif len(rel.parts) >= 5:
            dataset_name = "/".join(rel.parts[:-3])
            context_id = rel.parts[-3]
            run_id = rel.parts[-2]
        else:  # len(rel.parts) == 4
            dataset_name = "/".join(rel.parts[:-2])
            context_id = rel.parts[-2]
            # run id should be a folder at same level of agentic_round_history.json
            run_id_candidates = [p for p in history_path.parent.iterdir() if p.is_dir()]
            if len(run_id_candidates) != 1:
                logging.warning(
                    "Cannot determine run_id for %s: got %s, skipping",
                    history_path,
                    run_id_candidates,
                )
                run_id = None
            else:
                run_id = run_id_candidates[0].name
        yield dataset_name, context_id, run_id, history_path


def _cov_label_from_entry(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    overall = entry.get("overall_coverage")
    if isinstance(overall, (int, float)):
        overall = round(float(overall), 4)
    return {
        "is_pass_xrun": entry.get("is_pass_xrun"),
        "has_coverage": entry.get("has_coverage"),
        "overall_coverage": overall,
        "misc": entry.get("misc"),
    }


def _write_jsonl(output_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_split(dataset_name: str) -> str:
    if dataset_name == "hez2024/cvdp_ecov_eval":
        return "eval"
    return "train"


def _load_dataset_contexts(dataset_name: str, ids: set[str]) -> dict[str, LlmGenTbContext]:
    dataset = load_dataset_by_name(dataset_name, split=_dataset_split(dataset_name))
    contexts: dict[str, LlmGenTbContext] = {}
    for item in dataset:
        if isinstance(item, DataContext) and (item.id in ids and item.id not in contexts):
            contexts[item.id] = data_context_to_llm_gen_tb_context(item)
    return contexts


def _find_tb_from_task_dir(
    task_dir: Path, rtl_names: set[str], spec_names: set[str]
) -> DataFile | None:
    candidates: list[DataFile] = []
    for path in sorted(task_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith("_result.json"):
            continue
        if path.suffix != ".sv" and path.suffix != ".v":
            continue
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
        data_file = DataFile(name=path.name, content=content)
        if data_file.name in rtl_names or data_file.name in spec_names:
            continue
        candidates.append(data_file)
    if not candidates:
        return None
    candidates.sort(key=lambda f: (not f.name.startswith("tb_"), f.name))
    return candidates[0]


def _iter_direct_infer_task_dirs(run_dir: Path) -> Iterable[tuple[str, str, str, Path, Path]]:
    for result_path in run_dir.rglob("*_result.json"):
        try:
            rel = result_path.relative_to(run_dir)
        except ValueError:
            continue
        if len(rel.parts) < 4:
            continue
        run_id = rel.parts[-2]
        context_id = rel.parts[-3]
        dataset_name = "/".join(rel.parts[:-3])
        task_dir = result_path.parent
        yield dataset_name, context_id, run_id, task_dir, result_path


def _collect_direct_infer_rows(run_dir: Path, fetch_best: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_entries: dict[tuple[str, str], dict[str, Any]] = {}
    if fetch_best:
        eval_stats = _load_eval_stats(run_dir)
        best_entries = _best_entries_from_eval_stats(eval_stats, run_dir)
    dataset_ids: dict[str, set[str]] = {}
    if fetch_best:
        for dataset_name, context_id in best_entries:
            dataset_ids.setdefault(dataset_name, set()).add(context_id)
    else:
        for dataset_name, context_id, _, _, _ in _iter_direct_infer_task_dirs(run_dir):
            dataset_ids.setdefault(dataset_name, set()).add(context_id)

    contexts_by_dataset: dict[str, dict[str, LlmGenTbContext]] = {}
    for dataset_name, ids in dataset_ids.items():
        contexts_by_dataset[dataset_name] = _load_dataset_contexts(dataset_name, ids)

    if fetch_best:
        entries = list(best_entries.items())
        iterable = tqdm(entries, desc="Processing direct_infer entries", total=len(entries))
        for (dataset_name, context_id), entry in iterable:
            if not isinstance(entry, dict):
                logging.warning("Skipping invalid entry for %s:%s", dataset_name, context_id)
                continue
            run_id = entry.get("run_id")
            if not run_id:
                logging.warning("Missing run_id for %s:%s", dataset_name, context_id)
                continue
            context = contexts_by_dataset.get(dataset_name, {}).get(context_id)
            if context is None:
                logging.warning("Missing dataset context for %s:%s", dataset_name, context_id)
                continue
            task_dir = run_dir / Path(*dataset_name.split("/")) / context_id / str(run_id)
            if not task_dir.is_dir():
                logging.warning("Missing task dir: %s", task_dir)
                continue
            rtl_names = {f.name for f in context.rtl_files}
            spec_names = {f.name for f in context.spec_files}
            tb_file = _find_tb_from_task_dir(task_dir, rtl_names, spec_names)
            if tb_file is None:
                logging.warning("Missing TB file in %s", task_dir)
                continue
            messages = _build_direct_infer_messages(context)
            rows.append(
                {
                    "input": messages,
                    "output": _format_tb_output(tb_file),
                    "dataset": dataset_name,
                    "context_id": context_id,
                    "run_type": "direct_infer",
                    "cov_result": _cov_label_from_entry(entry),
                    "prev_cov_result": None,
                    "run_id": str(run_id),
                }
            )
        return rows

    tasks = list(_iter_direct_infer_task_dirs(run_dir))
    for dataset_name, context_id, run_id, task_dir, result_path in tqdm(
        tasks, desc="Processing direct_infer runs", total=len(tasks)
    ):
        context = contexts_by_dataset.get(dataset_name, {}).get(context_id)
        if context is None:
            logging.warning("Missing dataset context for %s:%s", dataset_name, context_id)
            continue
        rtl_names = {f.name for f in context.rtl_files}
        spec_names = {f.name for f in context.spec_files}
        tb_file = _find_tb_from_task_dir(task_dir, rtl_names, spec_names)
        if tb_file is None:
            logging.warning("Missing TB file in %s", task_dir)
            continue
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        if not isinstance(result, dict):
            logging.warning("Invalid result json in %s", result_path)
            continue
        try:
            cov_result = eval_cov_result_against_expectations(context, result)
        except Exception as e:
            logging.warning(
                "Failed to evaluate cov_result for %s:%s run %s: %s",
                dataset_name,
                context_id,
                run_id,
                e,
            )
            continue
        messages = _build_direct_infer_messages(context)
        rows.append(
            {
                "input": messages,
                "output": _format_tb_output(tb_file),
                "dataset": dataset_name,
                "context_id": context_id,
                "run_type": "direct_infer",
                "cov_result": _cov_label_from_entry(cov_result.model_dump()),
                "prev_cov_result": None,
                "run_id": str(run_id),
            }
        )
    return rows


def _collect_agentic_rows(run_dir: Path, fetch_best: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    histories = list(_iter_agentic_histories(run_dir))
    best_entries: dict[tuple[str, str], dict[str, Any]] = {}
    if fetch_best:
        eval_stats = _load_eval_stats(run_dir)
        best_entries = _best_entries_from_eval_stats(eval_stats, run_dir)
    for dataset_name, context_id, run_id, history_path in tqdm(
        histories, desc="Processing agentic histories", total=len(histories)
    ):
        with history_path.open("r", encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list) or not history:
            logging.warning("Invalid history file: %s", history_path)
            continue
        entry = history[-1]
        if not isinstance(entry, dict):
            logging.warning("Invalid history entry in %s", history_path)
            continue
        messages = entry.get("messages")
        output = entry.get("llm_response")
        if not isinstance(messages, list) or not isinstance(output, str):
            logging.warning("Missing messages/output in %s", history_path)
            continue
        history_run_id = (
            entry.get("run_id") or entry.get("task_hash_id") or entry.get("prev_run_id") or run_id
        )
        if history_run_id is not None:
            history_run_id = str(history_run_id)
        if fetch_best:
            best_entry = best_entries.get((dataset_name, context_id))
            best_run_id = best_entry.get("run_id") if isinstance(best_entry, dict) else None
            if not best_run_id:
                logging.warning("Missing best run_id for %s:%s", dataset_name, context_id)
                continue
            if history_run_id != str(best_run_id):
                continue
        messages = _strip_prompt_injection_messages(messages)
        messages = _compress_system_prompt(messages)
        rows.append(
            {
                "input": messages,
                "output": {"role": "assistant", "content": output},
                "dataset": dataset_name,
                "context_id": context_id,
                "run_type": "agentic",
                "cov_result": _cov_label_from_entry(entry.get("cov_result")),
                "prev_cov_result": _cov_label_from_entry(entry.get("prev_cov_result")),
                "run_id": history_run_id,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an eda_intermediate run dir into a synthetic JSONL dataset."
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Run dir under eda_intermediate."
    )
    parser.add_argument(
        "--run-type",
        choices=["direct_infer", "agentic"],
        required=True,
        help="Whether the run dir is direct_infer or agentic.",
    )
    parser.add_argument(
        "--fetch-best",
        action="store_true",
        help="For agentic runs, use eval_stats.json to select the best run per context.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output jsonl path.")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")

    if args.run_type == "direct_infer":
        rows = _collect_direct_infer_rows(run_dir, args.fetch_best)
    else:
        rows = _collect_agentic_rows(run_dir, args.fetch_best)

    _write_jsonl(args.output, rows)
    logging.info("Wrote %d rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
