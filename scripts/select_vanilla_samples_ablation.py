import argparse
import logging
import math
import random
from pathlib import Path

from select_vanilla_samples import (
    RunEntry,
    _build_eval_stats,
    _discover_runs,
    _load_dataset_contexts,
    _write_output_dir,
)
from tqdm import tqdm

from llm4cov.datasets.eval import eval_cov_result_against_expectations
from llm4cov.datasets.types import LlmGenTbContext


def _select_best(entries: list[RunEntry]) -> RunEntry | None:
    if not entries:
        return None
    return max(entries, key=lambda e: (e.cov_result, e.run_id))


def _select_random(entries: list[RunEntry], rng: random.Random) -> RunEntry | None:
    if not entries:
        return None
    return rng.choice(entries)


def _filter_full_coverage(entries: list[RunEntry]) -> list[RunEntry]:
    return [e for e in entries if e.cov_result.overall_coverage < 1.0]


def _write_round_output(
    vanilla_run_dir: Path,
    round_id: int,
    selected: list[RunEntry],
    select_mode: str,
    overwrite: bool,
) -> None:
    output_dir = Path(f"{vanilla_run_dir}_{select_mode}_{round_id}")
    _write_output_dir(output_dir, selected, overwrite)
    eval_stats = _build_eval_stats(selected)
    logging.info(
        "round_%d: %d run_ids -> %s (mean_cov=%.4f)",
        round_id,
        len(selected),
        output_dir,
        float(eval_stats.get("mean_overall_cov", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best-of runs from a vanilla batch run.")
    parser.add_argument("--vanilla-run-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to load when evaluating coverage targets.",
    )
    parser.add_argument(
        "--select-cnt",
        type=int,
        help="Total number of runs to select across rounds.",
    )
    parser.add_argument(
        "--select-ratio",
        type=float,
        help="Ratio of total candidates to select (0-1], computed after filtering.",
    )
    parser.add_argument(
        "--select-mode",
        type=str,
        choices=("best", "random"),
        default="best",
        help="Selection mode per context id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for selecting context ids.",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        help="Output directory for selected runs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dirs.")
    args = parser.parse_args()

    vanilla_run_dir: Path = args.vanilla_run_dir
    output_run_dir: Path | None = args.output_run_dir
    if not vanilla_run_dir.is_dir():
        raise FileNotFoundError(f"Missing vanilla run dir: {vanilla_run_dir}")
    if output_run_dir is not None:
        if not output_run_dir.parent.exists():
            raise FileNotFoundError(f"Missing output run dir parent: {output_run_dir.parent}")
        if output_run_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Output run dir already exists: {output_run_dir}")
    else:
        output_run_dir = vanilla_run_dir
    if args.select_cnt is None and args.select_ratio is None:
        raise ValueError("Must set --select-cnt or --select-ratio")
    if args.select_cnt is not None and args.select_ratio is not None:
        raise ValueError("Set only one of --select-cnt or --select-ratio")
    if args.select_cnt is not None and args.select_cnt <= 0:
        raise ValueError("--select-cnt must be positive")
    if args.select_ratio is not None and not (0.0 < args.select_ratio <= 1.0):
        raise ValueError("--select-ratio must be in (0, 1]")

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
    # remove empty entries
    keys_deleted = []
    for key in run_map:
        if not run_map[key]:
            keys_deleted.append(key)
        else:
            run_map[key].sort(key=lambda e: e.run_id)
    for key in keys_deleted:
        del run_map[key]

    # apply select ratio before filtering missing contexts
    if args.select_ratio is not None:
        total_entries = sum(len(entries) for entries in run_map.values())
        new_run_map: dict[tuple[Path, str], list[RunEntry]] = {}
        remaining = math.ceil(total_entries * args.select_ratio)
        rng = random.Random(args.seed)

        while remaining > 0 and run_map:
            keys = list(run_map.keys())
            len_keys = len(keys)
            if remaining >= len_keys:
                rng.shuffle(keys)
                chosen_keys = keys
                remaining -= len_keys
            else:
                chosen_keys = rng.sample(keys, k=remaining)
                remaining = 0
            logging.info(
                "Select-ratio round: choosing %d context ids, %d remaining.",
                len(chosen_keys),
                remaining,
            )
            for key in chosen_keys:
                if key not in new_run_map:
                    new_run_map[key] = list()
                new_run_map[key].append(run_map[key][-1])
                run_map[key].pop()
                if not run_map[key]:
                    del run_map[key]
        run_map = new_run_map

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
            except Exception as exc:
                logging.warning(
                    "Failed to evaluate coverage result for entry %s: %s", entry.run_id, exc
                )
                entries_id_to_delete.setdefault((dataset_rel, context_id), []).append(entry.run_id)

    for key in context_to_delete:
        del run_map[key]
    for key, entry_ids in entries_id_to_delete.items():
        entries = run_map[key]
        run_map[key] = [e for e in entries if e.run_id not in entry_ids]

    candidates: dict[tuple[Path, str], list[RunEntry]] = {}
    for key, entries in run_map.items():
        filtered = _filter_full_coverage(entries)
        if not filtered:
            continue
        filtered.sort(key=lambda e: e.run_id)
        candidates[key] = filtered

    total_candidates = sum(len(entries) for entries in candidates.values())
    logging.info("Total candidates after filtering missing contexts: %d", total_candidates)
    # select-ratio computation was done above
    select_cnt = total_candidates if args.select_ratio is not None else args.select_cnt
    assert select_cnt is not None
    assert total_candidates >= select_cnt, f"Not enough candidates ({total_candidates})"
    logging.info("Selecting %d candidates (mode=%s).", select_cnt, args.select_mode)

    rng = random.Random(args.seed)
    remaining = select_cnt
    round_id = 0

    while remaining > 0 and candidates:
        context_keys = sorted(candidates.keys(), key=lambda k: (str(k[0]), k[1]))
        if remaining >= len(context_keys):
            chosen_keys = context_keys
            remaining -= len(context_keys)
        else:
            chosen_keys = rng.sample(context_keys, k=remaining)
            remaining = 0

        selected: list[RunEntry] = []
        for key in chosen_keys:
            entries = candidates[key]
            if args.select_mode == "best":
                entry = _select_best(entries)
            else:
                entry = _select_random(entries, rng)
            if entry is None:
                continue
            selected.append(entry)
            entries.remove(entry)
            if not entries:
                del candidates[key]

        logging.info(
            "Round %d: selected %d entries (mode=%s).",
            round_id,
            len(selected),
            args.select_mode,
        )
        _write_round_output(output_run_dir, round_id, selected, args.select_mode, args.overwrite)
        round_id += 1

    if remaining > 0:
        logging.warning("Stopped early: %d selections remaining but no candidates.", remaining)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    main()
