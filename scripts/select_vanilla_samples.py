import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm4cov.datasets.eval import eval_cov_result_against_expectations
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import CovResult, LlmGenTbContext, data_context_to_llm_gen_tb_context


@dataclass
class RunEntry:
    dataset_rel: Path
    dataset_name: str
    context_id: str
    run_id: str
    run_dir: Path
    eda_result: dict[str, Any]
    cov_result: CovResult


def _load_result_raw(result_path: Path) -> dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid result json: {result_path}")
    return data


def _discover_runs(vanilla_run_dir: Path) -> dict[tuple[Path, str], list[RunEntry]]:
    run_map: dict[tuple[Path, str], list[RunEntry]] = {}
    for result_path in vanilla_run_dir.rglob("*_result.json"):
        run_dir = result_path.parent
        context_dir = run_dir.parent
        try:
            dataset_rel = context_dir.relative_to(vanilla_run_dir)
        except ValueError:
            continue
        if len(dataset_rel.parts) < 2:
            continue
        dataset_name = "/".join(dataset_rel.parts[:-1])
        context_id = context_dir.name
        run_id = run_dir.name
        try:
            eda_result = _load_result_raw(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logging.warning("Skipping %s: %s", result_path, exc)
            continue
        entry = RunEntry(
            dataset_rel=dataset_rel,
            dataset_name=dataset_name,
            context_id=context_id,
            run_id=run_id,
            run_dir=run_dir,
            eda_result=eda_result,
            cov_result=CovResult(id=context_id),
        )
        run_map.setdefault((dataset_rel, context_id), []).append(entry)
    return run_map


def _select_xrun_fail(entries: list[RunEntry]) -> RunEntry | None:
    candidates = [e for e in entries if not e.cov_result.is_pass_xrun]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.run_id)


def _select_candidates(entries: list[RunEntry]) -> list[RunEntry]:
    return [
        e
        for e in entries
        if e.cov_result.is_pass_xrun
        and e.cov_result.has_coverage
        and e.cov_result.overall_coverage < 1.0
    ]


def _select_median_from_candidates(candidates: list[RunEntry]) -> RunEntry:
    coverage_map: dict[float, list[RunEntry]] = {}
    for entry in candidates:
        coverage_map.setdefault(entry.cov_result.overall_coverage, []).append(entry)
    unique_covs = sorted(coverage_map)
    median_idx = (len(unique_covs) - 1) // 2
    median_cov = unique_covs[median_idx]
    return min(coverage_map[median_cov], key=lambda e: e.run_id)


def _select_worst_from_candidates(candidates: list[RunEntry]) -> RunEntry:
    worst_cov = min(e.cov_result.overall_coverage for e in candidates)
    worst_candidates = [e for e in candidates if e.cov_result.overall_coverage == worst_cov]
    return min(worst_candidates, key=lambda e: e.run_id)


def _has_cov_gap(median_entry: RunEntry, worst_entry: RunEntry, threshold: float) -> bool:
    return (
        median_entry.cov_result.overall_coverage - worst_entry.cov_result.overall_coverage
    ) > threshold


def _build_eval_stats(selected: list[RunEntry]) -> dict[str, Any]:
    results = [e.cov_result for e in selected]
    total_items = len(results)
    if total_items == 0:
        return {"total_items": 0}
    pass_xrun = sum(1 for r in results if r.is_pass_xrun)
    have_coverage = sum(1 for r in results if r.has_coverage)
    pass_targets = sum(1 for r in results if r.is_pass_targets)
    mean_overall_cov = sum(r.overall_coverage for r in results) / total_items
    mean_overall_cov_has_cov = (
        sum(r.overall_coverage for r in results if r.has_coverage) / have_coverage
        if have_coverage > 0
        else 0.0
    )
    non_agentic_items = [x for x in results if x.id.startswith("cvdp_copilot_")]
    agentic_items = [x for x in results if x.id.startswith("cvdp_agentic_")]
    pass_targets_non_agentic = sum(1 for r in non_agentic_items if r.is_pass_targets)
    pass_targets_agentic = sum(1 for r in agentic_items if r.is_pass_targets)

    best_results_output = {
        str(entry.dataset_rel): {
            "run_id": entry.run_id,
            "is_pass_xrun": entry.cov_result.is_pass_xrun,
            "has_coverage": entry.cov_result.has_coverage,
            "overall_coverage": entry.cov_result.overall_coverage,
            "is_pass_targets": entry.cov_result.is_pass_targets,
            "misc": entry.cov_result.misc,
        }
        for entry in selected
    }
    return {
        "total_items": total_items,
        "pass_xrun": pass_xrun / total_items if total_items > 0 else 0.0,
        "have_coverage": have_coverage / total_items if total_items > 0 else 0.0,
        "pass_targets": pass_targets / total_items if total_items > 0 else 0.0,
        "mean_overall_cov": mean_overall_cov,
        "mean_overall_cov_has_cov": mean_overall_cov_has_cov,
        "pass_targets_non_agentic": (
            pass_targets_non_agentic / len(non_agentic_items) if len(non_agentic_items) > 0 else 0.0
        ),
        "pass_targets_agentic": (
            pass_targets_agentic / len(agentic_items) if len(agentic_items) > 0 else 0.0
        ),
        "best_results_output": best_results_output,
    }


def _write_output_dir(output_dir: Path, selected: list[RunEntry], overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in tqdm(selected, desc=f"Copying to {output_dir}"):
        target_dir = output_dir / entry.dataset_rel / entry.run_id
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(entry.run_dir, target_dir)
    eval_stats = _build_eval_stats(selected)
    eval_path = output_dir / "eval_stats.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_stats, f, indent=4)


def _load_dataset_contexts(
    dataset_name: str, split: str, target_ids: list[str]
) -> dict[str, LlmGenTbContext]:
    dataset_raw = load_dataset_by_name(dataset_name, split=split)
    return {
        ctx.id: data_context_to_llm_gen_tb_context(ctx)
        for ctx in dataset_raw
        if ctx.id in target_ids
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select runs from a vanilla batch run.")
    parser.add_argument("--vanilla-run-dir", required=True, type=Path)
    parser.add_argument(
        "--worst-threshold",
        type=float,
        default=0.1,
        help="Min coverage gap between median and worst to keep worst run.",
    )
    parser.add_argument(
        "--prioritize-worst",
        action="store_true",
        help="Select worst run first; keep median only if coverage gap exceeds threshold.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to load when evaluating coverage targets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dirs.")
    args = parser.parse_args()

    vanilla_run_dir: Path = args.vanilla_run_dir
    if not vanilla_run_dir.is_dir():
        raise FileNotFoundError(f"Missing vanilla run dir: {vanilla_run_dir}")

    run_map = _discover_runs(vanilla_run_dir)
    dataset_cache: dict[str, dict[str, LlmGenTbContext]] = {}

    context_id_map: dict[str, list[str]] = {}
    for (dataset_rel, context_id), _ in run_map.items():
        dataset_name = "/".join(dataset_rel.parts[:-1])
        context_id_map.setdefault(dataset_name, []).append(context_id)

    for dataset_name, context_ids in context_id_map.items():
        dataset_cache[dataset_name] = _load_dataset_contexts(
            dataset_name, args.dataset_split, context_ids
        )

    context_to_delete: list[tuple[Path, str]] = []
    entries_id_to_delete: dict[tuple[Path, str], list[str]] = {}
    for (dataset_rel, context_id), entries in tqdm(run_map.items(), desc="Processing datasets"):
        dataset_name = "/".join(dataset_rel.parts[:-1])
        context_map = dataset_cache[dataset_name]
        context = context_map.get(context_id)
        if context is None:
            logging.warning("Missing context %s in dataset %s", context_id, dataset_name)
            context_to_delete.append((dataset_rel, context_id))
            continue
        for entry in entries:
            try:
                entry.cov_result = eval_cov_result_against_expectations(context, entry.eda_result)
            except Exception as e:
                logging.warning(
                    "Failed to evaluate coverage result for entry %s: %s", entry.run_id, e
                )
                entries_id_to_delete.setdefault((dataset_rel, context_id), []).append(entry.run_id)

    for key in context_to_delete:
        del run_map[key]
    for key, entry_ids in entries_id_to_delete.items():
        entries = run_map[key]
        run_map[key] = [e for e in entries if e.run_id not in entry_ids]

    xrun_selected: list[RunEntry] = []
    median_selected: list[RunEntry] = []
    worst_selected: list[RunEntry] = []

    for entries in tqdm(run_map.values(), desc="Selecting runs"):
        entries.sort(key=lambda e: e.run_id)
        xrun_entry = _select_xrun_fail(entries)
        if xrun_entry is not None:
            xrun_selected.append(xrun_entry)
        candidates = _select_candidates(entries)
        if not candidates:
            continue

        if args.prioritize_worst:
            worst_entry = _select_worst_from_candidates(candidates)
            worst_selected.append(worst_entry)
            median_entry = _select_median_from_candidates(candidates)
            if _has_cov_gap(median_entry, worst_entry, args.worst_threshold):
                median_selected.append(median_entry)
        else:
            median_entry = _select_median_from_candidates(candidates)
            median_selected.append(median_entry)
            worst_entry = _select_worst_from_candidates(candidates)
            if _has_cov_gap(median_entry, worst_entry, args.worst_threshold):
                worst_selected.append(worst_entry)

    xrun_dir = Path(f"{vanilla_run_dir}_xrun_fail")
    median_dir = Path(f"{vanilla_run_dir}_median")
    worst_dir = Path(f"{vanilla_run_dir}_worst")

    _write_output_dir(xrun_dir, xrun_selected, args.overwrite)
    _write_output_dir(median_dir, median_selected, args.overwrite)
    _write_output_dir(worst_dir, worst_selected, args.overwrite)

    logging.info("xrun_fail: %d run_ids -> %s", len(xrun_selected), xrun_dir)
    logging.info("median: %d run_ids -> %s", len(median_selected), median_dir)
    logging.info("worst: %d run_ids -> %s", len(worst_selected), worst_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    main()
